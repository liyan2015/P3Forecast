'''
Author: yooki(yooki.k613@gmail.com)
LastEditTime: 2025-03-21 14:41:31
Description: Federated GAN Training based on Data Synthesis Quality
'''
import copy
from lib.utils import load_variants,save_variants,add_to_csv
import data.parameters as parameters
from lib.classes import ModelParas,CloudClient,torch,np,pd,os

def client(cloud: CloudClient,
        generate_paras: ModelParas = None,
        columes=['cpu_util'],
        train_type = 'ours',
        base_epoch = 0,
        is_gan_train = False,
        is_gan_pre = True,
        loss_interval = 10,
        args=None
        ):
    """
    Args:
    ------------
        cloud: CloudClient, the cloud client
        generate_paras: ModelParas, the model parameters
        columes: list, the columns of data
        train_type: str, the train type
        base_epoch: int, the base epoch
        is_gan_train: bool, whether to train the GAN model
        is_gan_pre: bool, whether to use the pre-trained GAN model
        loss_interval: int, the interval of adding loss
        args: argparse.Namespace, the other parameters

    """
    print('cloud: ',cloud.cloud_type)
    if cloud.gzs is None:
        cloud.gzs = []
    if args is not None and args.weight in ['dtw','pdtw']:
        e_methods = ('mmd',args.weight)
    else:
        e_methods = ('mmd',)
    if is_gan_train:
        train_interval = 100  # gan training interval to record evaluation
        epochs=int(np.ceil(generate_paras.num_epochs/train_interval))
        train_epochs = generate_paras.num_epochs
        losses = []
        res={}
        for method in e_methods:
            res[method]=[]
        for ii in range(epochs):
            data_np,loss = cloud.train_gan(generate_paras,
            min(train_interval, train_epochs),
            train_type,columes,interval=loss_interval,is_pre=is_gan_pre,is_train=is_gan_train,is_first=(ii==0))
            print("--------finished train gan------------")
            losses.append(loss)
            _, gz_renorm,_,_ = cloud.generate(generate_paras,columes,data_np,None,is_show=False)
            if np.isnan(gz_renorm).any():
                raise ValueError('Generated data contains NaN!')
            r = cloud.evaluate(gz_renorm,generate_paras,columes,data_np,methods=e_methods)
            print(f"{ii+1}/{epochs}: ",end=' ')
            for method in e_methods:
                res[method].append(r[method])
                print(f"{method}={res[method]}", end=', ')
            print('\n')
            train_epochs -= train_interval
        if args is not None:
            save_variants({cloud.cloud_type:r[args.weight]},f'{args.weight}_{cloud.id}',is_update=True)

        LS = {
            'epoch':[x for x in range(loss_interval+base_epoch, 1+base_epoch+generate_paras.num_epochs, loss_interval)],
            'loss_g':[y for x in losses for y in x[0]],
            'loss_g_u':[y for x in losses for y in x[1]],
            'loss_g_v':[y for x in losses for y in x[2]],
            'loss_d':[y for x in losses for y in x[3]],
            'loss_s':[y for x in losses for y in x[4]],
        }
        df = pd.DataFrame(LS)
        filename__ = parameters.Paths['results']+f'/{cloud.cloud_type}/{train_type}_{cloud.id}_loss.csv'
        add_to_csv(filename__, df)
        print(f"Saved loss to {filename__}")
        
        EV = {'epoch': np.arange(train_interval+base_epoch, 1+base_epoch+generate_paras.num_epochs, train_interval).tolist()}
        for method in e_methods:
            if method.lower() == 'mmd':
                EV[method.upper()] = res[method]
            else:
                EV[method.upper()] = [x.mean() for x in res[method]] 
        df = pd.DataFrame(EV)
        filename_ = parameters.Paths['results']+f'/{cloud.cloud_type}/{train_type}_{cloud.id}_eval.csv'
        add_to_csv(filename_, df)
        print(f"Saved eval to {filename_}")

def fedavg(w,weights):
    """Federated average

    Args:
    ------------
        w: list, the model weights
        weights: list, the weights of client

    Returns:
    ------------
        w_avg: dict, the average model weights
    """

    weights = np.array(weights)
    weights = weights/weights.sum()
    print('weights:',weights)
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w[0][k] * weights[0]
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * weights[i]
    return w_avg

def FL(generate_paras:ModelParas,train_type:str,clouds:list,columes:list, global_epoch, id_, K=7, id=parameters.ID, args=None):
    """ in Federated learning environment, train the gan model

    Args:
    ------------
        generate_paras: ModelBase, the model parameters
        train_type: str, the train type
        clouds: list, the cloud clients
        columes: list, the columns of data
        global_epoch: int, the number of FL epochs
        paras: dict, the parameters
        id_: int, the id_ (use the model in id_)
        K: int, the number of clients
        id: int, the id
        **kwargs: the other parameters
    """
    global_epoch = global_epoch
    for epoch in range(global_epoch):
        print(f"Rounds {epoch+1}/{global_epoch}")
        idx = np.random.choice(range(len(parameters.selected_Clouds)),K,replace=False)
        w_ds,w_es,w_gs,w_rs,w_ss,ns = [],[],[],[],[],[]
        for i in idx:
            cloud = clouds[i]
            client(cloud,generate_paras,columes,base_epoch=epoch*generate_paras.num_epochs,train_type = train_type,is_gan_pre=args.gan_is_pre,is_gan_train=args.gan_not_train, args=args)
            if not args.gan_not_train:
                continue
            if args is not None and args.weight == 'pdtw':
                pdtws = load_variants(f'pdtw_{id}')
                ns.append(1/pdtws[cloud.cloud_type].mean())
            elif args is not None and args.weight=='avg':
                ns.append(1)
            elif args is not None and args.weight=='dtw':
                dtws = load_variants(f'dtw_{id}')
                ns.append(1/dtws[cloud.cloud_type].mean())
            else:
                dataSizes = load_variants(f'dataSizes_{id}')
                ns.append(dataSizes[cloud.cloud_type])
            w_es.append(torch.load(parameters.Paths['timegan_model'] + "/%s/embedder_%d.pt"%(cloud.cloud_type,id_), map_location=cloud.device))
            w_gs.append(torch.load(parameters.Paths['timegan_model'] + "/%s/generator_%d.pt"%(cloud.cloud_type,id_), map_location=cloud.device))
            w_rs.append(torch.load(parameters.Paths['timegan_model'] + "/%s/recovery_%d.pt"%(cloud.cloud_type,id_), map_location=cloud.device))
            w_ss.append(torch.load(parameters.Paths['timegan_model'] + "/%s/supervisor_%d.pt"%(cloud.cloud_type,id_), map_location=cloud.device))
            w_ds.append(torch.load(parameters.Paths['timegan_model'] + "/%s/discriminator_%d.pt"%(cloud.cloud_type,id_), map_location=cloud.device))
        if not args.gan_not_train:
            continue
        w_e = fedavg(w_es,ns)
        w_g = fedavg(w_gs,ns)
        w_r = fedavg(w_rs,ns)
        w_s = fedavg(w_ss,ns)
        w_d = fedavg(w_ds,ns)
        if not os.path.exists(parameters.Paths['timegan_model'] + "/%s"%train_type):
            os.mkdir(parameters.Paths['timegan_model'] + "/%s"%train_type)
        torch.save(w_e, parameters.Paths['timegan_model'] + "/%s/embedder_%d.pt"%(train_type,id_))
        torch.save(w_g, parameters.Paths['timegan_model'] + "/%s/generator_%d.pt"%(train_type,id_))
        torch.save(w_r, parameters.Paths['timegan_model'] + "/%s/recovery_%d.pt"%(train_type,id_))
        torch.save(w_s, parameters.Paths['timegan_model'] + "/%s/supervisor_%d.pt"%(train_type,id_))
        torch.save(w_d, parameters.Paths['timegan_model'] + "/%s/discriminator_%d.pt"%(train_type,id_))

