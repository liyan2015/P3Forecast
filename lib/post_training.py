'''
Author: yooki(yooki.k613@gmail.com)
LastEditTime: 2025-03-20 21:22:20
Description: Post-Training of Local Workload Prediction Models
'''
from lib.utils import add_to_csv,custom_modelparas,datetime,print_time_lag
from parameters import Paths,ID,Clouds,Clouds_,plt
from lib.classes import ModelParas,CloudClient,torch,np,pd,os

def client(cloud: CloudClient,
        predict_paras: ModelParas = None,
        generate_paras: ModelParas = None,
        columes=['cpu_util'],
        train_type = 'local',
        metric_='rmse',
        base_epoch = 0,
        args=None
        ):
    """
    Args:
    ------------
        cloud: CloudClient, the cloud client
        predict_paras: ModelParas, the model parameters
        generate_paras: ModelParas, the model parameters
        columes: list, the columns of data
        train_type: str, the train type
        metric_: str, the evaluation metric
        base_epoch: int, the base epoch
        args: argparse.Namespace, the other parameters

    Returns:
    ------------
        filename: str, the filename
    """
    print('cloud: ',cloud.cloud_type)
    filename_ = ''
    if cloud.gzs is None:
        cloud.gzs = []
    data_np,_ = cloud.train_gan(generate_paras,0,train_type,columes,interval=0,is_pre=True,is_train=False,is_first=True)
    _, gz_renorm,_,gz = cloud.generate(generate_paras,columes,data_np,None,is_show=False)
    if np.isnan(gz_renorm).any():
        raise ValueError('Generated data contains NaN!')

    if len(cloud.gzs) == 0:
        if cloud.train_threshold!=0:
            cloud.gzs = gz
    else:
        cloud.gzs = np.concatenate((cloud.gzs,cloud.query(predict_paras,gz,metric_,is_query=args.not_query)),axis=0)
 
    DT = {
        'model': predict_paras.num_epochs * [predict_paras.model_type],
        'cloud': predict_paras.num_epochs * [cloud.cloud_type],
        'epoch': np.arange(1+base_epoch, predict_paras.num_epochs + 1 + base_epoch).tolist()
    }
    test_scores,  train_scores= cloud.train(predict_paras,columes,base_epoch,metric_=metric_,train_type=train_type,is_save=True,is_show=True)
    print("--------finished train model------------")
    for i in range(len(columes)):
        DT[f"test_{columes[i]}"] = test_scores[:,i].tolist()
        DT[f"train_{columes[i]}"] = train_scores[:,i].tolist()
    cloud.train_threshold = np.min(np.mean(train_scores,axis=1))
    # cloud.test_threshold = np.min(np.mean(test_scores,axis=1))
    df = pd.DataFrame(DT)
    filename_ = Paths['results']+f'/{cloud.cloud_type}/{train_type}_{cloud.id}.csv'
    add_to_csv(filename_, df)
    return filename_

def post_training(predict_paras,generate_paras,clouds,columes,train_type,paras,args,K):
    """Post training of local predictor

    Args:
    ------------
        predict_paras: ModelBase, the model parameters
        generate_paras: ModelBase, the model parameters
        clouds: list, the cloud clients
        columes: list, the columns of data
        train_type: str, the train type
        paras: dict, the custom parameters
        args: argparse.Namespace, the other parameters
        K: int, the number of clients
    """
    idx = np.random.choice(range(len(Clouds_)),K,replace=False)
    for index,i in enumerate(idx):
        cloud = clouds[i]
        print(f"Clients {index+1}/{K}: ",cloud.cloud_type)
        r = args.epochs//args.local_epochs_post
        for j in range(r):
            client(cloud,custom_modelparas(predict_paras,cloud.cloud_type,paras,train_type),generate_paras,columes,base_epoch=j*predict_paras.num_epochs,train_type = train_type,args=args)


