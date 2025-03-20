'''
Author: yooki(yooki.k613@gmail.com)
LastEditTime: 2025-03-20 21:23:57
Description: main.py
'''
from parameters import Clouds,ID,SEED,Clouds_
from lib.utils import print_time_lag,datetime
from lib.fedgan_training import ModelParas,FL,np,CloudClient
from lib.post_training import post_training
from torch import device as nndevice,cuda
from lib.utils import save_json,refresh_history
import argparse
import random

st = datetime.datetime.now()
parser = argparse.ArgumentParser(description='Parameters required to run the program')

# ---------------main parameters---------------

parser.add_argument('-g','--gpu',type=int,default=0,help='gpu device, if -1, use cpu')
parser.add_argument('-c','--columes',type=str,default='cpu_util,mem_util',help='dataset columes')
parser.add_argument('-n','--note',type=str,default='',help='note of this run')
parser.add_argument('-s','--seed',type=int,help='customize seed')
parser.add_argument('-id','--id',type=int,default=None,help='choose gan models with id')
parser.add_argument('-cs','--clouds',type=str,default=None,help='cloud indexs 0-6, such as 0,1,2,3,4,5,6')
parser.add_argument('-p','--probability',type=float,default=1,help='the probability of cloud selected')

# ---------------Federated GAN parameters---------------

parser.add_argument('-ghs','--gan_hidden_size',type=int,default=256,help='hidden size of timegan')
parser.add_argument('-gln','--gan_layer_num',type=int,default=3,help='layer size of timegan')
parser.add_argument('-glr','--gan_learning_rate',type=int,default=0.001,help='learning rate of timegan')
parser.add_argument('-gle','--gan_local_epochs',type=int,default=500,help='local train epochs of Federated GAN')
parser.add_argument('-cr','--rounds',type=int,default=10,help='communication rounds of Federated GAN')
parser.add_argument('-gnt','--gan_not_train',action="store_false",help='with False not train, without True')
parser.add_argument('-gip','--gan_is_pre',action="store_true",help='with True gan pre, without False')
parser.add_argument('-w','--weight',type=str,help='aggreating weight',choices=['pdtw','dtw','datasize','avg'])

# ---------------post training parameters---------------

parser.add_argument('-m','--model',type=str,default='GRU',help='predictor: GRU or TCN or LSTM', choices=['GRU','TCN','LSTM'])
parser.add_argument('-l','--seq_len',type=int,default=64,help='sequence length')
parser.add_argument('-b','--batch_size',type=int,default=128,help='batch size of predictor')
parser.add_argument('-hd','--hidden_size',type=int,default=512,help='hidden size of predictor')
parser.add_argument('-ln','--layer_num',type=int,default=1,help='layer size of predictor')
parser.add_argument('-e','--epochs',type=int,default=30,help='total train epochs of predictor')
parser.add_argument('-lep','--local_epochs_post',type=int,default=5,help='local epochs of post training')
parser.add_argument('-lr','--learning_rate',type=int,default=0.001,help='learning rate of predictor')
parser.add_argument('-lrs','--learning_rate_strategy',type=str,default='adaptive',help='learning rate strategy of post training',choices=['fixed','adaptive'])
parser.add_argument('-mu','--mu',type=float,default=0.1,help='parameter mu to control the learning rate')
parser.add_argument('-nq','--not_query',action="store_false",help='with means False not query, without means True query')

args = parser.parse_args()
print(args)
# Custom model parameters and mu for each cloud
paras = {
    'hidden_size':{'Alibaba':args.hidden_size,'Azure':args.hidden_size,'Google':args.hidden_size,'Alibaba-ML':args.hidden_size,'Sugon-HF':args.hidden_size,'Sugon-KS':args.hidden_size,'Sugon-WZ':args.hidden_size},
    'layer_size':{'Alibaba':args.layer_num   ,'Azure':args.layer_num,'Google':args.layer_num,'Alibaba-ML':args.layer_num,'Sugon-HF':args.layer_num,'Sugon-KS':args.layer_num,'Sugon-WZ':args.layer_num},
    'mu':{'Alibaba':args.mu,'Azure':args.mu,'Google':args.mu,'Alibaba-ML':args.mu,'Sugon-HF':args.mu,'Sugon-KS':args.mu,'Sugon-WZ':args.mu}
}
mode_type = args.model
columes = args.columes.split(',')
train_type = 'ours'
note = args.note
if args.learning_rate_strategy == 'fixed':
    lrs = args.learning_rate
else:
    lrs = str(args.learning_rate)
if args.seed is not None:
    SEED = args.seed
    print('Customize SEED:',SEED)
cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if cuda.is_available():
    cuda.manual_seed(SEED)
if cuda.is_available() and args.gpu>=0:
    DEVICE = nndevice(f"cuda:{args.gpu}")
else:
    DEVICE = "cpu"
ID_ = ID
if args.id is not None:
    ID_ = args.id
predict_paras = ModelParas(seq_length=args.seq_len,
                    batch_size=args.batch_size,
                    hidden_size=args.hidden_size,
                    learning_rate=lrs,
                    num_epochs=args.local_epochs_post, 
                    model_type=mode_type,
                    layer_size=args.layer_num,
                    input_size=len(columes),
                    output_size=len(columes),
                    device=DEVICE
                    )
generate_paras = ModelParas(seq_length=args.seq_len+1,
                    batch_size=args.batch_size,
                    hidden_size=args.gan_hidden_size,
                    learning_rate=[args.gan_learning_rate,args.gan_learning_rate,args.gan_learning_rate],
                    num_epochs=args.gan_local_epochs,
                    model_type=mode_type,
                    layer_size=args.gan_layer_num,
                    input_size=len(columes),
                    device=DEVICE)
# refresh the history data, delete the deleted data in logs.json
# refresh_history()
if args.clouds is not None:
    Clouds_ = [Clouds[int(i)] for i in args.clouds.split(',')]
print('clouds: ',Clouds_)
if args.probability >1 or args.probability <0:
    raise ValueError('the probability of cloud selected must be in [0,1]')
clouds = []
for cloud in Clouds_:
    clouds.append(CloudClient(cloud,ID,ID_, mu=paras['mu'][cloud],device=DEVICE))

global_epoch=args.rounds
if note == '':
    note = f'TimeGAN train epochs = {generate_paras.num_epochs};weight is {args.weight}'
# record logs
logs = {
    str(ID):{
        'seed': SEED,
        'columes':columes,
        'train_type':train_type,
        'global_epoch':global_epoch,
        'note':note,
        'pred_paras':predict_paras.to_dict(),
        'gen_paras':generate_paras.to_dict(),
        'args':str(args),
        'paras':paras
    }
}

FL(generate_paras,train_type,clouds,columes,global_epoch,K=round(len(Clouds_)*args.probability), id=ID,id_=ID_,args=args)

save_json(logs)

post_training(predict_paras,generate_paras,clouds,columes,paras=paras,train_type = train_type,K=round(len(Clouds_)*args.probability),args=args)
   
txt = print_time_lag(st,datetime.datetime.now(),'main')
print(txt)
