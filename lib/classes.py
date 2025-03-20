'''
Author: yooki(yooki.k613@gmail.com)
LastEditTime: 2025-03-20 20:35:26
Description: 
(1) ModelParas: Model parameters class
(2) TimeSeriesDataset: Time series dataset class
(3) GenTimeSeriesDataset: Generated time series dataset class
(4) CloudClient: Cloud client class
(5) Center: Center class
'''
import math
from lib.models import TCN,GRU,LSTM,torch,nn
from lib.utils import np,pd,os,read_dataSet,print_time_lag,metric,save_variants
from torch.utils.data import Dataset, DataLoader
from parameters import Paths,Clouds,Clouds_,plt
import datetime
from random import shuffle
from tqdm import tqdm
from lib.timegan.timegan import TimeGAN,TimeDataset
from lib.similarity import fastdtw,euclidean,fastpdtw,mmd_rbf

class ModelParas:
    """
    Model Parameters class
    Note: model_type must be one of [``'GRU'``, ``'LSTM'``, ``'TCN'``]
    """
    def __init__(self,
                 seq_length,
                 batch_size,
                 hidden_size,
                 learning_rate,
                 num_epochs,
                 model_type,
                 criterion=nn.MSELoss(),
                 train_percentage=0.7,
                 input_size=1,
                 layer_size=1,
                 output_size=1,
                 dropout=0.2,
                 device="cpu"):
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.output_size = output_size
        if layer_size == 1:
            self.dropout = 0
        else:
            self.dropout = dropout
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.train_percentage = train_percentage
        self.criterion = criterion.to(device)
        self.model_type = model_type

    def to_dict(self):
        return {
            'seq_length': self.seq_length,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'model_type': self.model_type,
            'train_percentage': self.train_percentage,
            'hidden_size': self.hidden_size,
            'input_size': self.input_size,
            'layer_size': self.layer_size,
            'output_size': self.output_size,
        }
    @staticmethod
    def from_dict(dt:dict): 
        return ModelParas(
            seq_length=dt['seq_length'],
            batch_size=dt['batch_size'],
            hidden_size=dt['hidden_size'],
            learning_rate=dt['learning_rate'],
            num_epochs=dt['num_epochs'],
            model_type=dt['model_type'],
            train_percentage=dt['train_percentage'],
            input_size=dt['input_size'],
            layer_size=dt['layer_size'],
            output_size=dt['output_size']
        )

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        """Time series dataset

        Args:
        -------------
        data: np.ndarray, the data
        seq_length: int, the length of sequence
        """
        if type(data) is pd.core.series.Series:
            self.data = torch.Tensor(data.values)
        else:
            self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length] # (64,2)
        y = self.data[idx + self.seq_length]# (2)
        return x, y
    
class GenTimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        """Time series dataset

        Args:
        -------------
        data: np.ndarray, the data
        seq_length: int, the length of sequence
        """
        self.data = torch.Tensor(data)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx,:self.seq_length]# (64,2)
        y = self.data[idx,self.seq_length]# (2)
        return x, y

class CloudClient:
    __train_loader = None
    __test_loader = None
    __max_x = None
    __min_x = None
    ct = None
    gzs = None
    # test_threshold = 0
    train_threshold = 0
    def __init__(self, cloud_type:str, id, id_, mu=0, device="cpu"):
        """
        Args:
        -------------
        cloud_type: str, the cloud type
        id: int, the id
        id_: int, the id_ (use the TimeGAN in id_)
        mu: float, the mu of learning rate
        
        """
        self.cloud_type = cloud_type
        self.df,_ = read_dataSet(cloud_type,isRaw=False)
        self.id = id
        self.id_ = id_
        self.gan = None
        self.mu = mu
        self.device = device

    def __normalize(self,x,cs=1):
        """Normalize the data
        
        Args:
        -------------
        x: np.ndarray, the data
        
        Returns:
        -------------
        np.ndarray, the normalized data"""
        self.__max_x = np.max(x,axis=0).reshape(-1,cs)
        self.__min_x = np.min(x,axis=0).reshape(-1,cs)
        return (x - self.__min_x) / (self.__max_x - self.__min_x)

    
    def __set_dataloader(self,
                        model_paras: ModelParas,
                        columes: list,
                        generated_data=[]):
        '''Normalize data and set the data loader => ``self.train_loader``,``self.test_loader``.

        Args:
        ------------
            model_paras: ModelBase, the model parameters
            columes: list, the columes of data
            generated_data: list, the generated data
        '''           
        if self.__train_loader is None:
            datas = [self.df[x].values for x in columes]
            tsv = np.stack(datas, axis=1)
            index = int(len(tsv) * model_paras.train_percentage)
            train = tsv[0:index]
            train = self.__normalize(train,len(columes))
            train = torch.from_numpy(train).to(torch.float32)
            train_data = TimeSeriesDataset(train, model_paras.seq_length)
            self.__train_loader = [DataLoader(train_data,
                                    batch_size=model_paras.batch_size,
                                    shuffle=True)]
            self.ct = Center(self.id)
            self.__test_loader = self.ct.get_center_test_datasets(model_paras, columes)
        if generated_data is not None and len(generated_data)>0:
            self.__train_loader+=[DataLoader(GenTimeSeriesDataset(generated_data,model_paras.seq_length),batch_size=model_paras.batch_size,shuffle=True)]


    def __get_dataloader_self(self,
                        model_paras: ModelParas,
                        data,
                        is_shuffle=True):
        '''Set the normalized data loader => ``self.test_loader``.

        Args:
        ------------
            model_paras: ModelBase, the model parameters
            data: np.ndarray, the normalized data
        '''                   
        return DataLoader(GenTimeSeriesDataset(data,model_paras.seq_length),
                            batch_size=model_paras.batch_size,
                            shuffle=is_shuffle)


    @staticmethod
    def init_model(model_paras: ModelParas):
        '''Initialize the model

        Args:
        ------------
            model_paras: ModelBase, the model parameters
        
        Returns:
        ------------
            model: nn.Module, the model
        '''
        if model_paras.model_type == 'GRU':
            model = GRU(model_paras.input_size, model_paras.hidden_size,
                        model_paras.output_size, model_paras.layer_size,
                        dropout=model_paras.dropout)
        elif model_paras.model_type == 'TCN':
            model = TCN(model_paras.input_size, model_paras.output_size,num_channels=[4,4,4,4],kernel_size=4,dropout= 0.1)
        elif model_paras.model_type == 'LSTM':
            model = LSTM(model_paras.input_size, model_paras.hidden_size,
                        model_paras.output_size, model_paras.layer_size,
                        dropout=model_paras.dropout)
        else:
            raise ValueError(f'{model_paras.model_type} model is temporarily not supported!')
        return model
    
    def train(self,model_paras: ModelParas, columns: list = ['cpu_util'], base_epoch = 0, metric_ = 'rmse',train_type='ours', is_save=True, is_show=True):
        '''Train the model for one column but []

        Args:
        ------------
            model_paras: ModelBase, the model parameters
            columns: list, the columns of data
            base_epoch: int, the base epoch
            metric_: str, the evaluation metric
            train_type: str, the train type
            is_save: bool, whether to save the model
            is_show: bool, whether to show the loss

        Returns:
        ------------
            test_scores: list, the test scores
            train_scores: list, the train scores
        '''
        start_time = datetime.datetime.now()
        self.__set_dataloader(model_paras, columns, self.gzs)
        model_paras.model_type = model_paras.model_type.upper()
        model_save_path = f"{Paths['prediction_model']}/{self.cloud_type}/{model_paras.model_type}_{self.id}.pt"
        model_load_path = model_save_path
        if not os.path.exists(os.path.dirname(model_save_path)):
            os.makedirs(os.path.dirname(model_save_path))
        if not os.path.exists(os.path.dirname(model_load_path)):
            os.makedirs(os.path.dirname(model_load_path))
        model = self.init_model(model_paras)
        if os.path.exists(model_load_path):
            model.load_state_dict(torch.load(model_load_path))
        model = model.to(self.device)
        if type(model_paras.learning_rate) is dict:
            lr_ = model_paras.learning_rate[self.cloud_type]
        elif type(model_paras.learning_rate) is list:
            lr_ = model_paras.learning_rate[Clouds.index(self.cloud_type)]
        elif type(model_paras.learning_rate) is str:
            lr_  = math.exp(-self.mu*(len(self.gzs)/len(self.df)/model_paras.train_percentage))*float(model_paras.learning_rate)          
        else:
            lr_ = model_paras.learning_rate

        # txt = f'Learning rate: {lr_}, Gen Data Length: {len(self.gzs)}'
        # print(txt)
        optimizer = torch.optim.Adam(model.parameters(),lr=lr_)
        test_scores = []
        train_scores = []
        train_losses = []
        epoch_bar = tqdm(range(base_epoch,base_epoch+model_paras.num_epochs))
        for epoch in epoch_bar :
            losses,n = 0,0
            train_predictions, train_labels = [], []
            #* model train
            model.train()
            for train_loader in self.__train_loader:
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = model(inputs)
                    # Calculate loss
                    loss = model_paras.criterion(outputs, labels)
                    losses+=loss.item()
                    n+=1
                    train_predictions.append(outputs)
                    train_labels.append(labels)
                    # Backward and optimize
                    loss.backward()
                    optimizer.step()
            train_losses.append(losses/n)
            # *model eval
            model.eval()
            with torch.no_grad():
                test_predictions, test_labels = [[] for x in self.__test_loader], [[] for x in self.__test_loader]
                losses = 0
                for i,test_loader in enumerate(self.__test_loader):
                    for inputs, labels in test_loader:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        try:
                            outputs = model(inputs)
                            loss = model_paras.criterion(outputs, labels)
                            test_predictions[i].append(outputs)
                            test_labels[i].append(labels)
                            losses += loss.item()
                        except Exception as e:
                            print('ERROR: ', e)
            losses = losses / len(self.__test_loader)
            if is_save:
                torch.save(model.state_dict(), model_save_path)
            test_score = np.zeros((len(self.__test_loader),len(columns)))
            for i in range(len(self.__test_loader)):
                test_predictions[i] = (torch.cat(test_predictions[i]).cpu().numpy())# * (max_x[i] - min_x[i]) + min_x[i]
                test_labels[i] = (torch.cat(test_labels[i]).cpu().numpy()) #* (max_x[i] - min_x[i]) + min_x[i]
                test_score[i] = metric(test_predictions[i], test_labels[i], metric_, 0)

            test_scores.append(test_score.mean(axis=0))
            train_predictions = (torch.cat(train_predictions).cpu().detach().numpy())#* (self.__max_x - self.__min_x) + self.__min_x
            train_labels = (torch.cat(train_labels).cpu().detach().numpy()) #* (self.__max_x - self.__min_x) + self.__min_x
            train_score = metric(train_predictions, train_labels, metric_, 0)
            train_scores.append(train_score)
            epoch_bar.set_description(
                f"Epoch [{epoch + 1}/{model_paras.num_epochs}]\n"
                f"Loss: {losses:.4f}\n"
                f"{metric_.upper()}: {str([round(y, 4) for y in list(test_scores[-1])])} -> {test_scores[-1].mean():.4f}"
            )
        test_scores = np.array(test_scores)
        train_scores = np.array(train_scores)
        print_time_lag(start_time,datetime.datetime.now(),f'{self.cloud_type} train {model_paras.model_type} model {model_paras.num_epochs} epochs')
        if is_save:
            torch.save(model.state_dict(), model_save_path)
            
        train_losses = np.array(train_losses)
        if is_show:
            x = np.arange(1, model_paras.num_epochs + 1)
            interval = 5
            tri_x = x[::interval]
            for j,col in enumerate(columns):
                tri_y = test_scores[::interval,j]
                plt.plot(x, test_scores[:,j])
                plt.scatter(tri_x, tri_y)
            plt.title(f'Test accuracy on test datasets of {self.cloud_type} by {train_type}')
            plt.legend(columns)
            plt.xlabel('Epoch')
            plt.ylabel('Test Accuracy')
            plt.show()
        return test_scores, train_scores
    
    def test(self,model_paras: ModelParas, columns: list = ['cpu_util']):
        '''Test the model for multiple columns

        Args:
        ------------
            model_paras: ModelBase, the model parameters
            columns: list, the columns of data
        '''
        model_path = f"{Paths['prediction_model']}/{self.cloud_type}/{model_paras.model_type}_{self.id}.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'{model_path} not found!')
        if self.__test_loader is None:
            self.__set_dataloader(model_paras, columns)
        model = self.init_model(model_paras) 
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        with torch.no_grad():
            total_loss = 0
            test_predictions, test_labels = [[] for x in self.__test_loader], [[] for x in self.__test_loader]
            for i,test_loader in enumerate(self.__test_loader):
                for inputs, labels in test_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    try: 
                        outputs = model(inputs)
                        loss = model_paras.criterion(outputs, labels)
                        test_predictions[i].append(outputs)
                        test_labels[i].append(labels)
                        total_loss += loss.item() * len(inputs)
                    except Exception as e:
                        print('ERROR: ', e)
        txt = "loss on test data: {}".format(total_loss)
        print(txt)
        # max_x, min_x = self.ct.get_max_min()
        rmses = np.zeros((len(self.__test_loader),len(columns)))
        smapes = np.zeros((len(self.__test_loader),len(columns)))
        maes = np.zeros((len(self.__test_loader),len(columns)))
        for i in range(len(self.__test_loader)):
            test_predictions[i] = (torch.cat(test_predictions[i]).cpu().numpy()) #* (max_x[i] - min_x[i]) + min_x[i]
            test_labels[i] = (torch.cat(test_labels[i]).cpu().numpy()) #* (max_x[i] - min_x[i]) + min_x[i]
            rmses[i] = metric(test_predictions[i], test_labels[i], 'rmse',0)
            smapes[i] = metric(test_predictions[i], test_labels[i], 'smape',0)
            maes[i] = metric(test_predictions[i], test_labels[i], 'mae',0)
        txt = '{} on test data: {}'.format('rmse'.upper(),rmses.mean())
        print(txt)
        txt = '{} on test data: {}'.format('smape'.upper(),smapes.mean())
        print(txt)
        txt = '{} on test data: {}'.format('mae'.upper(),maes.mean())
        print(txt)
        test_predictions = np.row_stack(test_predictions)
        test_labels = np.row_stack(test_labels)
        return test_predictions, test_labels


    def test_self(self,model_paras: ModelParas, test_data,metric_='rmse'):
        '''Test the model by the given test data

        Args:
        ------------
            model_paras: ModelBase, the model parameters
            test_data: np.array, the test data
            metric_: str, the evaluation metric

        Returns:
        ------------
            scores: np.ndarray, the test result
        '''
        test_loader = self.__get_dataloader_self(model_paras,test_data,False)
        model_path = f"{Paths['prediction_model']}/{self.cloud_type}/{model_paras.model_type}_{self.id}.pt"
        model = self.init_model(model_paras) 
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        with torch.no_grad():
            total_loss = 0
            test_predictions, test_labels = [], []
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                try: 
                    outputs = model(inputs)
                    loss = model_paras.criterion(outputs, labels)
                    test_predictions.append(outputs)
                    test_labels.append(labels)
                    total_loss += loss.item() * len(inputs)
                except Exception as e:
                    print('ERROR: ', e)
        # txt = "loss on test data: {}".format(total_loss)
        # print(txt)
        test_predictions = torch.cat(test_predictions).cpu().numpy()
        test_labels = torch.cat(test_labels).cpu().numpy()
        test_predictions = np.expand_dims(test_predictions,axis=0)
        test_labels = np.expand_dims(test_labels,axis=0)
        scores = metric(test_predictions, test_labels, metric_, 0)
        return scores



    def train_gan(self,model_paras: ModelParas ,train_epochs, train_type:str,columns: list = ['cpu_util'],interval = 1000, is_pre = False,is_train = True, is_first=True):
        '''Train the GAN model or load the pre-trained model if ``is_train`` is False

        Args:
        ------------
            model_paras: ModelBase, the model parameters
            train_epochs: int, the train epochs
            train_type: str, the train type
            columns: list, the columns of data
            interval: int, the interval of printing loss
            is_pre: bool, whether to use the pre-trained model
            is_train: bool, whether to train the model
            is_first: bool, whether is the first training of local trainning

        Returns:
        ------------
            data_np: np.ndarray, shape(n_samples,input_size) ,the original data
            loss: list|None, the loss of GAN [loss_g,loss_g_u,loss_g_v,loss_d,loss_s]
        '''
        loss = []
        load_name =  train_type
        if self.gan is None:
            self.gan = TimeGAN(model_paras.input_size, model_paras.hidden_size, self.device,name = self.cloud_type, load_name=load_name, gan_index=0,is_pre=is_pre, num_layers=model_paras.layer_size,batch_size=model_paras.batch_size,id=self.id_)
        else:
            self.gan.load(is_first)
        data_np = np.column_stack([self.df[x].values[:] for x in columns])
        data_np = data_np[:int(len(data_np)*model_paras.train_percentage)]
        if is_train:
            data = TimeDataset(data_np,model_paras.seq_length)
            loss = self.gan.train(train_epochs,data,interval,lrs=model_paras.learning_rate)
        return data_np,loss

    def generate(self,model_paras: ModelParas, columns: list = ['cpu_util'], data_np:np.ndarray = None, dataset_size = None,is_show=True):
        '''Generate the data, please train the model first

        Args:
        ------------
            model_paras: ModelBase, the model parameters
            columns: list, the columns of data
            data_np: np.ndarray, the original data
            dataset_size: int, the size of generated data
            is_show: bool, whether to show the generated data evaluation visualization
        
        Returns:
        ------------
            gz_order: np.ndarray, the ordered and data
            gz_renorm: np.ndarray, the ordered and renormalized data
            data: list, TimeDataset 
            gz:   list, the generated data (data_len,seq_len,input_size)
        '''
        if data_np is None:
            data_np = np.column_stack([self.df[x].values for x in columns])
            data_np = data_np[:int(len(data_np)*model_paras.train_percentage)]
        data = TimeDataset(data_np,model_paras.seq_length)
        if self.gan is None:
            raise ValueError('Please train the GAN model first!')
        if dataset_size is None:
            dataset_size = len(data)
        gz = self.gan.generate(data,cols=columns,dataset_size=dataset_size,isEvaluate=is_show)
        gz_order = data.concat_data(gz,dataset_size=dataset_size)
        gz_order[gz_order<0] = 0 
        gz_renorm = data.renormalize(gz_order)
        return gz_order, gz_renorm,data, gz
    
    def evaluate(self,gz_renorm:np.ndarray,model_paras: ModelParas, columns: list = ['cpu_util'], data_np:np.ndarray = None, methods=('pdtw',)):
        '''Evaluate the generated data

        Args:
        ------------
            gz_renorm: np.ndarray, the ordered and renormalized data
            model_paras: ModelBase, the model parameters
            columns: list, the columns of data
            data_np: np.ndarray, the original data
            method: set, the evaluation methods

        Returns:
        ------------
            res: dict, the evaluation result
        '''
        if data_np is None:
            data_np = np.column_stack([self.df[x].values for x in columns])
            data_np = data_np[:int(len(data_np)*model_paras.train_percentage)]
        res = {}
        for method in methods:
            if method.lower() == 'mmd':
                mmd = mmd_rbf(data_np,gz_renorm)
                res[method] = mmd
                print('MMD:',mmd)
            if method.lower() == 'dtw':
                dtws = np.zeros(len(columns))
                for i in range(len(columns)):
                    dtw_,_ = fastdtw(data_np[:,i:i+1], gz_renorm[:,i:i+1],dist=euclidean)
                    dtws[i] = dtw_
                    print(columns[i],'DTW',dtw_)
                res[method] = dtws
            if method.lower() == 'pdtw':
                sims = np.zeros(len(columns))
                for i in range(len(columns)):
                    _,sim = fastpdtw(data_np[:,i],gz_renorm[:,i])
                    if type(sim) is list:
                        sim = sim[0]  
                    print(columns[i],'PDTW',sim)
                    sims[i]=sim
                res[method] = sims
        return res

    def query(self,model_paras: ModelParas,gzs:np.ndarray,  metric_='rmse',is_query=True):
        '''Evaluate the generated data

        Args:
        ------------
            model_paras: ModelBase, the model parameters
            gzs: np.ndarray, the generated data shape->(data_len,seq_len,input_size)
            metric_: str, the evaluation metric
            is_query: bool, whether to query the data

        Returns:
        ------------
            query_data: np.ndarray, the query data
        '''
        if not is_query:
            return gzs
        scores = self.test_self(model_paras, gzs, metric_) 
        scores = scores.mean(axis=1)
        mask = scores<=self.train_threshold
        return gzs[mask]

class Center(CloudClient):
    __train_loader = None
    __test_loader = None
    __max_x = None
    __min_x = None
    gan = None
    def __init__(self,id) -> None:
        self.cloud_type = 'Center'
        self.dfs = []   
        self.id = id
        self.id_ = id
        x = {}
        for cloud_type in Clouds_:
            df,_ = read_dataSet(cloud_type,False)
            self.dfs.append(df)
            x[cloud_type] = len(df)
        save_variants(x,f'dataSizes_{id}')
    
    def __set_max_min(self,xs,cs=1):
        """Set normalization of the data min max
        shape of x: (n_clients, n_features)
        
        Args:
        -------------
        xs: [np.ndarray], the data
        cs: int, the number of columes
        """
        maxs,mins = [],[]
        for x in xs:
            maxs.append(np.max(x,axis=0).reshape(-1,cs))
            mins.append(np.min(x,axis=0).reshape(-1,cs))
        self.__max_x = np.row_stack(maxs)
        self.__min_x = np.row_stack(mins)

    def __set_dataloader(self,
                        model_paras: ModelParas,
                        columes: list):
        '''Normalize data and set the data loader => ``self.train_loader``,``self.test_loader``.

        Args:
        ------------
            model_paras: ModelBase, the model parameters
            columes: list, the columes of data
        '''
        self.__train_loader = []
        self.__test_loader = []
        trains = []
        tests = []
        for df in self.dfs:               
            datas = [df[x].values for x in columes]
            tsv = np.stack(datas, axis=1)
            index = int(len(tsv) * model_paras.train_percentage)
            trains.append(tsv[0:index])
            tests.append(tsv[index + 1:])
        self.__set_max_min(trains,cs=len(columes))
        for i in range(len(trains)):      
            test = tests[i]
            test = (test - self.__min_x[i]) / (self.__max_x[i] - self.__min_x[i])
            train = (trains[i] - self.__min_x[i]) / (self.__max_x[i] - self.__min_x[i])
            train = torch.from_numpy(train).to(torch.float32)
            test = torch.from_numpy(test).to(torch.float32)
            train_data = TimeSeriesDataset(train, model_paras.seq_length)
            test_data = TimeSeriesDataset(test, model_paras.seq_length)
            # self.__train_loader.append(DataLoader(train_data,
            #                         batch_size=model_paras.batch_size,
            #                         shuffle=True))
            self.__test_loader.append(DataLoader(test_data, batch_size=model_paras.batch_size,shuffle=False))
        # shuffle(self.__train_loader)


    def get_center_test_datasets(self, model_paras, columns):
        self.__set_dataloader(model_paras,columns)
        return self.__test_loader
    
    def get_max_min(self):
        return self.__max_x, self.__min_x