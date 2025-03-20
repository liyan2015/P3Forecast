'''
Author: yooki(yooki.k613@gmail.com)
LastEditTime: 2025-03-20 21:23:49
Description: utils

(1) read_dataset: Read data from csv file
(2) time -> parse_time: Convert time to datetime
         -> get_time: Get the current time
         -> print_time_lag: Print the time difference
         -> parse_string: Convert string to datetime
(3) variant -> save_variants: Save variants to file
            -> load_variants: Load variants from file
(4) add_to_csv: Add data to csv file
(5) refresh_history: Automatically refresh historical data according to log.json
(6) metric: Evaluation metrics of the model
'''

from parameters import os, data2018_path, dataAzure_path, data2020_path, dataSugon_path, DataSets, Intervals, Clouds, Paths
import pandas as pd
import pickle
import re
import datetime
import numpy as np
import json
import copy

def save_json(data:dict,filename=Paths['results']+'/log.json',mode='update'):
    '''
    Save data to json file

    Args:
    ------------
        data: dict, the data
        filename: str, the file 
        mode: str, 'update' or 'rewrite'
    '''
    dt = read_json(filename)
    dt = {} if dt is None else dt
    if mode == 'update':
        dt.update(data)
    elif mode == 'rewrite':
        dt = data
    with open(filename, 'w') as file:
        json.dump(dt, file)


def read_json(filename=Paths['results']+'/log.json'):
    '''
    Read data from json file

    Args:
    ------------
        filename: str, the file name
    
    Returns:
    ------------
        data: dict, the data
    '''
    if not os.path.exists(filename):
        print(f"{filename} does not exist")
        return None
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


def refresh_history():
    """Refresh the history data (eg. ./output/*) by ./output/log.json """
    logs = read_json()
    logs_ = copy.deepcopy(logs)
    import os,re
    hz=['pt','csv','pkl','pt']
    def dg(folder_path,index):
        for file in os.listdir(folder_path):
            if os.path.isdir(folder_path+'/'+file):
                dg(folder_path+'/'+file, index)
            else:
                res = re.search(f'_(\d+)\.{hz[index]}',file)
                if res:
                    id = res.group(1)
                    if id not in logs and len(id) > 4:
                        os.remove(folder_path+'/'+file)
    for i,path_ in enumerate([Paths['prediction_model'], Paths['results'], Paths['variance'],Paths['timegan_model']]):
        dg(path_,i)


def save_variants(variants, var_name:str, is_update=False):
    '''
    Save variants to file

    Args:
    ------------
        variants: dict, the variants
        var_name: str, the name of the variant
        is_update: bool, if True, update the file, else create a new file
    '''
    if is_update and load_variants(var_name) is not None:
        former = load_variants(var_name)
        former.update(variants)
        variants = former
    with open(f"{Paths['variance']}/{var_name}.pkl", 'wb') as file:
        pickle.dump(variants, file)


def load_variants(var_name):
    '''
    Load variants from file

    Args:
    ------------
        var_name: str, the name of the variant

    Returns:
    ------------
        variants: dict, the variants
        if the file does not exist, return None
    '''
    if os.path.exists(f"{Paths['variance']}/{var_name}.pkl"):
        with open(f"{Paths['variance']}/{var_name}.pkl", 'rb') as file:
            variants = pickle.load(file)
        return variants
    else:
        return None


def custom_modelparas(model_paras, cloud:str, paras:dict, train_type:str):
    """Customize the model parameters
    
    Args:
    ------------
        model_paras: ModelParas, the model parameters
        cloud: str, the cloud provider
        paras: dict, the parameters

    Returns:
    ------------
        model_paras_: ModelParas, the customized model parameters
    """
    if train_type not in ['ours','normal'] or paras is None:
        return model_paras
    else:
        model_paras_ = copy.deepcopy(model_paras)
        model_paras_.hidden_size = paras['hidden_size'][cloud]
        model_paras_.layer_size = paras['layer_size'][cloud]
        if model_paras_.layer_size == 1:
            model_paras_.dropout = 0
        # model_paras_.batch_size = paras['batch_size'][cloud]
        return model_paras_


def read_dataSet(cloud_type, isRaw=True):
    """
    Read data from csv file

    Args:
    -----------
        cloud_type: str, the cloud provider
        isRaw: bool, if True, read raw data from csv file, else read processed data from csv file

    Returns:
    ------------
        if isRaw:
            df: pd.DataFrame
        else:
            df: pd.DataFrame
            freq: str, the time interval of the data
    """
    start = datetime.datetime.now()
    if isRaw:
        if cloud_type == 'Alibaba':
            df = pd.read_csv(data2018_path + 'machine_usage.csv',
                                names=[
                                    'machine_id', 'time', 'cpu_util', 'mem_util',
                                    'mem_gps', 'mpki', 'net_in', 'net_out',
                                    'disk_io'
                                ],
                                usecols=['machine_id', 'time', 'cpu_util','mem_util'])
        elif cloud_type == 'Azure':
            df1 = pd.read_csv(dataAzure_path + 'cluster_cpu_util.csv')
            df2 = pd.read_csv(dataAzure_path + 'cluster_mem_util.csv')
            reader = pd.read_csv(dataAzure_path + 'cluster_gpu_util.csv', chunksize=1)
            times = []
            gpus = []
            machines = []
            i=0
            j=0
            for chunk in reader:
                chunk.reset_index(inplace=True)
                gpus.append(chunk.iloc[0,2:].mean())
                times.append(chunk.iloc[0,0])
                machines.append(chunk.iloc[0,1])
                i+=1
                if i > 100*10000:
                    j+=1
                    print('%d :read %d of the data'%(j,i))
                    i=0
            df3 = pd.DataFrame({'time':times,'machine_id':machines,'gpu_util':gpus})
            df2['mem_util'] = (1-df2['mem_free'].astype('float')/df2['mem_total'].astype('float'))*100
            df2.drop(['mem_free', 'mem_total'], axis=1, inplace=True)
            df = pd.merge(df1, df2, on=['time','machine_id'])
            df = pd.merge(df, df3, on=['time','machine_id'])
            df=df.dropna()
        elif cloud_type == 'Alibaba-ML':
            df1 = pd.read_csv(
                data2020_path + 'pai_machine_metric.csv',
                usecols=[0, 1, 2, 3, 7, 11],
                names=['worker_name','machine_id', 'start_time', 'end_time', 'gpu_util','cpu_util'])
            df2 = pd.read_csv(
                data2020_path + 'pai_sensor_table.csv',
                usecols=[2, 4, 8],
                names=['worker_name','machine_id', 'avg_mem'])
            df3 = pd.read_csv(
                data2020_path + 'pai_machine_spec.csv',
                usecols=[0, 3],
                names=['machine_id', 'cap_mem'])
            df2 = pd.merge(df2, df3, on='machine_id')
            df2.dropna(inplace=True)
            df2['mem_util'] = df2['avg_mem'].astype('float')/df2['cap_mem'].astype('float')*100
            df2.drop(['avg_mem', 'cap_mem'], axis=1, inplace=True)
            df = pd.merge(df1, df2, on=['worker_name','machine_id'])           
        elif 'Sugon' in  cloud_type:
            if cloud_type == 'Sugon-HF':
                path_ = dataSugon_path + 'hefei-(22.1-22.6).csv'
                u=128
            elif cloud_type == 'Sugon-KS':
                path_ = dataSugon_path + 'ks-(22.1-22.2).csv'
                u=32
            elif cloud_type == 'Sugon-WZ':
                path_ = dataSugon_path + 'wuzhen-(2.1-22.6).csv'
                u=1
            if cloud_type == 'Sugon-HF':
                df = pd.read_csv(
                    path_,
                    usecols=[1, 9, 13, 14, 18, 20, 21, 24],
                    names=['acct_time', 'job_queue_time', 'need_nodes', 'nodect', 'job_response_time', 'job_cpu_time', 'job_mem_used','job_gpu_num'])               
            else:
                df = pd.read_csv(
                    path_,
                    usecols=[1, 9, 13, 14, 18, 20, 21, 24])
            df=df.dropna()
            df['cpu_util'] = df['job_cpu_time'].astype(
                'float')/df['job_response_time'].astype('float')/df['nodect'].astype('float')*100/u
            df.replace(np.inf,np.nan,inplace=True)
            df['mem_util'] = df['job_mem_used'].astype('float')/df['nodect'].astype('float')
            df['gpu_util'] = np.where(df['job_gpu_num'] > 0, 100, 0)
            # # 归一化df['job_mem_used']
            # df['mem_util'] = (df['job_mem_used'] - df['job_mem_used'].min())/(df['job_mem_used'].max()-df['job_mem_used'].min())*100
            df=df.dropna()
            df.drop(['nodect', 'job_cpu_time', 'job_response_time','job_gpu_num'], axis=1, inplace=True)
        delta = datetime.datetime.now() - start
        print("read {}.csv, spend {} hours {} minutes {} seconds".format(
            cloud_type, delta.seconds // 3600, (delta.seconds % 3600) // 60,
            delta.seconds % 60))
        return df
    else:
        df = pd.read_csv(DataSets[Clouds.index(cloud_type)])
        df['cpu_util'].replace(0, np.nan, inplace=True)
        if df.isna().any().any():
            df.interpolate(inplace=True)
            if df.isna().any().any():
                df.fillna(method='ffill', inplace=True)
                if df.isna().any().any():
                    df.fillna(method='bfill', inplace=True)
        freq = re.findall("(.*)_(.*?)\.",DataSets[Clouds.index(cloud_type)])[0][1]
        return df, freq


def get_time(format_ = '%Y-%m-%d %H:%M:%S')->str:
    '''
    Get the current time

    Args:
    ------------
        format_: str, the format of the time
    
    Returns:
    ------------
        str_time: str, the current time
    '''
    return datetime.datetime.now().strftime(format_)


def print_time_lag(t1:datetime.datetime, t2:datetime.datetime, function_name:str='This function'):
    '''
    Print the time function spent

    Args:
    ------------
        t1: datetime.datetime, the start time
        t2: datetime.datetime, the end time
        function_name: str, the function name
    
    Returns:
    ------------
        txt: str, the time lag
    '''

    delta = t2 - t1
    txt = "{}, spend {} hours {} minutes {} seconds".format(
        function_name, delta.seconds // 3600, (delta.seconds % 3600) // 60,
        delta.seconds % 60)
    print(txt)
    return txt


def parse_time(time: datetime.datetime, format_ = '%Y-%m-%d %H:%M:%S'):
    """
    Parse datetime.datetime to string

    Args:
    ------------
        time: datetime.datetime, the time
        format_: str, the format of the time
    
    Returns:
    ------------
        str_time: str, the time
    """
    return time.strftime(format_)


def parse_string(str_time: str):
    """Parse string to datetime.datetime
    
    Args:
    ------------
        str_time: str, the string
    
    Returns:
    ------------
        datetime.datetime
    """
    if len(str_time) == 16:
        return datetime.datetime.strptime(str_time, '%Y-%m-%d %H:%M')
    else:
        return datetime.datetime.strptime(str_time, '%Y-%m-%d %H:%M:%S')


def add_to_csv(filename_: str, df_: pd.DataFrame):
    """
    Add data to csv file and save, if the file does not exist, create it
    
    Args:
    ------------
        filename_: str, the file path
        df_: pd.DataFrame, the data
    """
    path_ = os.path.dirname(filename_)
    if not os.path.exists(path_):
        os.makedirs(path_)
    with open(filename_, mode='a', newline='') as file:
        df_.to_csv(file, header=file.tell() == 0, index=False)

def normalize(x):
    """Min-max normalize the data

    Args:
    -------------
    x: np.ndarray, the data

    Returns:
    -------------
    np.ndarray, the normalized data
    max_x
    min_x"""
    max_x = np.max(x)
    min_x = np.min(x)
    return (x - min_x) / (max_x - min_x), max_x,min_x

def z_score(x):
    """Z-score normalization

    Args:
    -------------
    x: np.ndarray, the data

    Returns:
    -------------
    np.ndarray, the normalized data
    mean
    std"""
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / std, mean, std

def metric(y_pred, y_true, t: str, axis=0):
    '''Evaluation metrics of the model

    Args:
    -------------
    y_pred: np.ndarray, the predicted value
    y_true: np.ndarray, the true value
    t: str, the type of evaluation metrics
        - ``'mape'``: mean absolute percentage error
        - ``'mae'``: mean absolute error
        - ``'rmse'``: root mean square error
        -``'smape'``: symmetric mean absolute percentage error

    Returns:
    -------------
        np.ndarray, the evaluation result
    '''
    if t.lower() == 'mape':
        return np.mean(np.abs((y_true - y_pred) / (y_true+1e-7)),axis=axis) * 100.0
    if t.lower() == 'mae':
        return np.mean(np.abs((y_true - y_pred)),axis=axis)
    if t.lower() == 'rmse':
        return np.sqrt(np.mean((y_true - y_pred)**2,axis=axis))
    if t.lower() == 'smape':
        return np.mean(np.abs((y_true - y_pred) / ((y_true + y_pred)/2)),axis=axis) * 100.0
