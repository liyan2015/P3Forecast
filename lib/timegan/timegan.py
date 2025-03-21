'''
Author: yooki(yooki.k613@gmail.com)
LastEditTime: 2025-03-20 21:22:38
Description: A pytorch implementation of TimeGAN (Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, "Time-series Generative Adversarial Networks," Neural Information Processing Systems (NeurIPS), 2019.), the code is modified from https://github.com/pakornv/timegan-pytorch.
'''

from itertools import chain
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import os
from data.parameters import Paths,ID
from lib.timegan.visualization_metrics import visualization
from lib.timegan.time_dataset import TimeDataset

class Net(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        rnn=nn.GRU,
        activation_fn=torch.sigmoid,
    ):
        super().__init__()
        self.rnn = rnn(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.activation_fn = activation_fn

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.linear(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return x


def to_tensor(data):
    return torch.from_numpy(data).float()

def batch_generator(dataset, batch_size):
    dataset_size = len(dataset)
    idx = torch.randperm(dataset_size)
    batch_idx = idx[:batch_size]
    batch = torch.stack([to_tensor(dataset[i]) for i in batch_idx])
    return batch

class TimeGAN():
    def __init__(self, input_size, hidden_size, device, name, load_name = None, is_pre = False, gan_index=0, save_model=True, gamma=1, num_layers=3, batch_size=128, id=ID):
        """Initialize the TimeGAN model

        Args:
        -----------------
            input_size: input size
            hidden_size: hidden size, in another way dim
            device: device
            name: name
            load_name: load model path name
            is_pre: if True, load the pre-trained model
            gan_index: gan index
            save_model: if True, save the model
            gamma: gamma
            num_layers: number of layers
            batch_size: batch size
            id: id
        """
        self.id = id
        self.batch_size = batch_size
        self.device = device
        self.gamma = gamma
        self.gan_index = gan_index
        self.save_model = save_model
        self.name = name
        self.load_name = name if load_name is None else load_name
        self.model = {
            'embedder': Net(input_size, hidden_size, hidden_size, num_layers).to(device),
            'recovery': Net(hidden_size, hidden_size, input_size, num_layers).to(device),
            'generator': Net(input_size, hidden_size, hidden_size, num_layers).to(device),
            'supervisor': Net(hidden_size, hidden_size, hidden_size, num_layers - 1).to(device),
            'discriminator': Net(hidden_size, hidden_size, 1, num_layers, activation_fn=None).to(device)
        }
        if is_pre and os.path.exists(Paths['timegan_model'] + "/" + self.load_name):
            try:
                self.load(True)
            except FileNotFoundError as e:
                print("Models not found:"+e.filename)
                pass
    def load(self,is_first):
        """Load the model

        Args:
        ------------
            is_first: if True, is the first training of local training
        """
        for key in self.model.keys():
            if is_first: # load global GAN model
                path = Paths['timegan_model'] + "/%s/%s_%d.pt"%(self.load_name,key,self.id)
            else: # load local GAN model
                path = Paths['timegan_model'] + "/%s/%s_%d.pt"%(self.name,key,self.id)
            self.model[key].load_state_dict(torch.load(path))
        print(f'load TIMEGAN({path}) success!')

    def train(
        self,
        iteration,
        dataset,
        interval=1000,
        lrs = [0.001,0.0005,0.0005],
    ):
        """Train the model

        Args:
        -----------
            iteration: number of iterations
            dataset: dataset
            interval: interval of printing loss
            lrs: three learning rates

        Returns:
        -----------
            loss:list loss of GAN [loss_g,loss_g_u,loss_g_v,loss_d,loss_s]
        """
        
        loss = [[],[],[],[],[]]
        # Discriminator loss
        def _loss_d(y_real, y_fake, y_fake_e):
            loss_d_real = F.binary_cross_entropy_with_logits(y_real, torch.ones_like(y_real))
            loss_d_fake = F.binary_cross_entropy_with_logits(y_fake, torch.zeros_like(y_fake))
            loss_d_fake_e = F.binary_cross_entropy_with_logits(y_fake_e, torch.zeros_like(y_fake_e))
            return loss_d_real + loss_d_fake + self.gamma * loss_d_fake_e

        # Generator loss
        ## Adversarial loss
        def _loss_g_u(y_fake):  
            return F.binary_cross_entropy_with_logits(y_fake, torch.ones_like(y_fake))

        def _loss_g_u_e(y_fake_e):
            return F.binary_cross_entropy_with_logits(y_fake_e, torch.ones_like(y_fake_e))

        ## Supervised loss
        def _loss_s(h_hat_supervise, h):
            # return F.mse_loss(h[:, 1:, :], h_hat_supervise[:, 1:, :])
            return F.mse_loss(h[:, 1:, :], h_hat_supervise[:, :-1, :])

        ## Two moments
        def _loss_g_v(x_hat, x):
            loss_g_v1 = torch.mean(
                torch.abs(torch.sqrt(torch.var(x_hat, 0) + 1e-6) - torch.sqrt(torch.var(x, 0) + 1e-6))
            )
            loss_g_v2 = torch.mean(torch.abs(torch.mean(x_hat, 0) - torch.mean(x, 0)))
            return loss_g_v1 + loss_g_v2
        ## Summation
        def _loss_g(loss_g_u, loss_g_u_e, loss_s, loss_g_v):
            return loss_g_u + self.gamma * loss_g_u_e + 100 * torch.sqrt(loss_s) + 100 * loss_g_v
        
        # Embedder network loss
        def _loss_e_t0(x_tilde, x):
            return F.mse_loss(x_tilde, x)

        def _loss_e_0(loss_e_t0):
            return torch.sqrt(loss_e_t0) * 10

        def _loss_e(loss_e_0, loss_s):
            return loss_e_0 + 0.1 * loss_s
        
        optimizer_er = optim.Adam(chain(self.model['embedder'].parameters(), self.model['recovery'].parameters()), lr=lrs[0])
        optimizer_gs = optim.Adam(chain(self.model['generator'].parameters(), self.model['supervisor'].parameters()), lr=lrs[1])
        optimizer_d = optim.Adam(self.model['discriminator'].parameters(),lr=lrs[2])
        for key in self.model.keys():
            self.model[key].train()
        print("Start Embedding Network Training")
        for step in range(1+self.gan_index*iteration, iteration + 1 + self.gan_index*iteration):

            x = batch_generator(dataset, self.batch_size).to(self.device)
            h = self.model['embedder'](x)
            x_tilde = self.model['recovery'](h)

            loss_e_t0 = _loss_e_t0(x_tilde, x)
            loss_e_0 = _loss_e_0(loss_e_t0)
            optimizer_er.zero_grad()
            loss_e_0.backward()
            optimizer_er.step()

            if step % interval == 0:
                print(
                    "step: "
                    + str(step)
                    + "/"
                    + str(iteration*(self.gan_index+1))
                    + ", loss_e: "
                    + str(np.round(np.sqrt(loss_e_t0.item()), 4))
                )
        print("Finish Embedding Network Training")
        print("Start Training with Supervised Loss Only")
        for step in range(1+self.gan_index*iteration, iteration + 1 + self.gan_index*iteration):
            x = batch_generator(dataset, self.batch_size).to(self.device)
            z = torch.randn(self.batch_size, x.size(1), x.size(2)).to(self.device)

            h = self.model['embedder'](x)
            h_hat_supervise = self.model['supervisor'](h)

            loss_s = _loss_s(h_hat_supervise, h)
            optimizer_gs.zero_grad()
            loss_s.backward()
            optimizer_gs.step()

            if step % interval == 0:
                print(
                    "step: "
                    + str(step)
                    + "/"
                    + str(iteration*(self.gan_index+1))
                    + ", loss_s: "
                    + str(np.round(np.sqrt(loss_s.item()), 4))
                )

        print("Finish Training with Supervised Loss Only")
        print("Start Joint Training")
        for step in range(1+self.gan_index*iteration, iteration + 1 + self.gan_index*iteration):
            for _ in range(2):
                x = batch_generator(dataset, self.batch_size).to(self.device)
                z = torch.randn(self.batch_size, x.size(1), x.size(2)).to(self.device)

                h = self.model['embedder'](x)
                e_hat = self.model['generator'](z)
                h_hat = self.model['supervisor'](e_hat)
                h_hat_supervise = self.model['supervisor'](h)
                x_hat = self.model['recovery'](h_hat)
                y_fake = self.model['discriminator'](h_hat)
                y_fake_e = self.model['discriminator'](e_hat)

                loss_s = _loss_s(h_hat_supervise, h)
                loss_g_u = _loss_g_u(y_fake)
                loss_g_u_e = _loss_g_u_e(y_fake_e)
                loss_g_v = _loss_g_v(x_hat, x)
                loss_g = _loss_g(loss_g_u, loss_g_u_e, loss_s, loss_g_v)
                optimizer_gs.zero_grad()
                loss_g.backward()
                optimizer_gs.step()

                h = self.model['embedder'](x)
                x_tilde = self.model['recovery'](h)
                h_hat_supervise = self.model['supervisor'](h)

                loss_e_t0 = _loss_e_t0(x_tilde, x)
                loss_e_0 = _loss_e_0(loss_e_t0)
                loss_s = _loss_s(h_hat_supervise, h)
                loss_e = _loss_e(loss_e_0, loss_s)
                optimizer_er.zero_grad()
                loss_e.backward()
                optimizer_er.step()

            x = batch_generator(dataset, self.batch_size).to(self.device)
            z = torch.randn(self.batch_size, x.size(1), x.size(2)).to(self.device)

            h = self.model['embedder'](x)
            e_hat = self.model['generator'](z)
            h_hat = self.model['supervisor'](e_hat)
            y_fake = self.model['discriminator'](h_hat)
            y_real = self.model['discriminator'](h)
            y_fake_e = self.model['discriminator'](e_hat)

            loss_d = _loss_d(y_real, y_fake, y_fake_e)
            if loss_d.item() > 0.15:
                optimizer_d.zero_grad()
                loss_d.backward()
                optimizer_d.step()

            if step % interval == 0:
                txt = "step: "\
                    + str(step)\
                    + "/"\
                    + str(iteration*(self.gan_index+1))\
                    + ", loss_d: "\
                    + str(np.round(loss_d.item(), 4))\
                    + ", loss_g_u: "\
                    + str(np.round(loss_g_u.item(), 4))\
                    + ", loss_g_v: "\
                    + str(np.round(loss_g_v.item(), 4))\
                    + ", loss_s: "\
                    + str(np.round(np.sqrt(loss_s.item()), 4))\
                    + ", loss_e_t0: "\
                    + str(np.round(np.sqrt(loss_e_t0.item()), 4))
                print(txt)
                
                loss[0].append(loss_g.item())
                loss[1].append(loss_g_u.item())
                loss[2].append(loss_g_v.item())
                loss[3].append(loss_d.item())
                loss[4].append(loss_s.item())
        print("Finish Joint Training")
        self.gan_index += 1
        if self.save_model:
            if not os.path.exists(Paths['timegan_model']+'/'+self.name):
                os.makedirs(Paths['timegan_model']+'/'+self.name)
            for key in self.model.keys():
                if self.load_name == self.name: # [local,center_gen]
                    torch.save(self.model[key].state_dict(), Paths['timegan_model']+"/%s/%s.pt"%(self.name,key))
                else:
                    torch.save(self.model[key].state_dict(), Paths['timegan_model']+"/%s/%s_%d.pt"%(self.name,key,self.id))
            print("Models saved")
        return loss

    def generate(self, dataset, cols, dataset_size, isEvaluate=True):
        """Visualize the generated data

        Args:
        --------------
            dataset: dataset
            isEvaluate: if True, visualize the generated data

        Returns:
        -----------
            generated_data: generated data shape(samples,input_size)
        """
        seq_len = dataset[0].shape[0]
        input_size = dataset[0].shape[1]
        self.model['generator'].eval()
        self.model['supervisor'].eval()
        self.model['recovery'].eval()

        with torch.no_grad():
            z = torch.randn(dataset_size, seq_len, input_size).to(self.device)
            e_hat = self.model['generator'](z)
            h_hat = self.model['supervisor'](e_hat)
            x_hat = self.model['recovery'](h_hat)

        generated_data_curr = x_hat.cpu().numpy()
        generated_data = list()
        for i in range(dataset_size):
            temp = generated_data_curr[i, :, :]
            generated_data.append(temp)
        if isEvaluate:
            visualization(dataset, generated_data, ["pca","tsne"], self.vis, cols, self.name)
        generated_data = np.array(generated_data)
        
        return generated_data
    


