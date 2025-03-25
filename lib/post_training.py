'''
Author: yooki(yooki.k613@gmail.com)
LastEditTime: 2025-03-22 06:49:03
Description: Post-Training of Local Workload Prediction Models
'''
from lib.utils import add_to_csv,custom_modelparas,copy
import data.parameters as parameters
from data.parameters import plt
from lib.classes import ModelParas,CloudClient,np,pd,threading

def client(cloud: CloudClient,
        predict_paras: ModelParas = None,
        generate_paras: ModelParas = None,
        columes=['cpu_util'],
        train_type = 'ours',
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
        test_scores: np.array, the test scores
    """
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
    test_scores,  train_scores= cloud.train(predict_paras,columes,base_epoch,metric_=metric_,is_save=True)
    print(cloud.cloud_type, "--------finished train model------------")
    for i in range(len(columes)):
        DT[f"test_{columes[i]}"] = test_scores[:,i].tolist()
        DT[f"train_{columes[i]}"] = train_scores[:,i].tolist()
    cloud.train_threshold = np.min(np.mean(train_scores,axis=1))
    # cloud.test_threshold = np.min(np.mean(test_scores,axis=1))
    df = pd.DataFrame(DT)
    filename_ = parameters.Paths['results']+f'/{cloud.cloud_type}/{train_type}_{cloud.id}.csv'
    add_to_csv(filename_, df)
    print(cloud.cloud_type,f"save predictive results to {filename_}")
    return test_scores

def post_training(predict_paras,generate_paras,clouds,columes,train_type,paras,args,K, metric_='rmse',is_show=False):
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
        metric_: str, the evaluation metric
        is_show: bool, whether to show the plot
    """
    idx = np.random.choice(range(len(parameters.selected_Clouds)),K,replace=False)
    threads = []
    for index,i in enumerate(idx):
        cloud = clouds[i]
        generate_paras_ = copy.deepcopy(generate_paras)
        predict_paras_ = copy.deepcopy(predict_paras)
        generate_paras_.device = cloud.device
        predict_paras_.device = cloud.device

        def run_client(cloud,predict_paras,generate_paras,columes,train_type,metric_,args):
            print(f"Clients {index+1}/{K}: ",cloud.cloud_type)
            r = args.epochs//args.local_epochs_post
            test_scores = np.zeros((args.epochs,len(columes)))
            for j in range(r):
                test_scores[j*predict_paras.num_epochs:(j+1)*predict_paras.num_epochs,:] = client(cloud,custom_modelparas(predict_paras,cloud.cloud_type,paras,train_type),generate_paras,columes,base_epoch=j*predict_paras.num_epochs,train_type = train_type,metric_=metric_,args=args)

            if is_show:
                fig, ax = plt.subplots(1,1)
                x = np.arange(1, args.epochs + 1)
                interval = 5
                tri_x = x[::interval][1:]-1
                for j,col in enumerate(columes):
                    tri_y = test_scores[::interval,j][1:]
                    ax.plot(x, test_scores[:,j],label=col)
                    ax.scatter(tri_x, tri_y)
                # plt.title(f'Test accuracy on test datasets of {cloud.cloud_type} by {train_type}')
                ax.legend()
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric_.upper())
                fig.savefig(f'{parameters.Paths["images"]}/{cloud.cloud_type}_{train_type}_{cloud.id}.png')
                plt.close(fig)
                print(f"save plot to {parameters.Paths['images']}/{cloud.cloud_type}_{train_type}_{cloud.id}.png")
        t = threading.Thread(target=run_client,args=(cloud,predict_paras_,generate_paras_,columes,train_type,metric_,args), daemon=True)     
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
