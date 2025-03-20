'''
Author: yooki(yooki.k613@gmail.com)
LastEditTime: 2025-03-20 21:24:04
Description: Shared parameters
'''
import matplotlib.pyplot as plt
# import seaborn as sns
import os,time
from random import randint
# --------------update--------------
original_data_path = '/usr/data1/'
output_path = './output'
data_path = './data/'

# --------------dataset--------------
# Raw dataset file path
data2018_path = original_data_path + 'Alibaba/cluster-trace-v2018/'
data2020_path = original_data_path + 'Alibaba/cluster-trace-gpu-v2020/'
dataAzure_path = original_data_path + 'Azure/philly-traces/'
dataSugon_path = original_data_path + 'Sugon/'
dataGoogle_path = original_data_path + 'Google/clusterdata-2011-2/task_usage/'
# Processed dataset file path
DataSets = [
    data_path+'Alibaba_1T.csv', 
    data_path+'Azure_10T.csv', 
    data_path+'Google_5T.csv', 
    data_path+'Alibaba-ML_10T.csv', 
    data_path+'Sugon-Ks_10T.csv', 
    data_path+'Sugon-Hefei_5T.csv', 
    data_path+'Sugon-Wuzhen_30T.csv' 
]

# --------------clients--------------
Clouds = ['Alibaba', 'Azure', 'Google', 'Alibaba-ML',
          'Sugon-KS','Sugon-HF', 'Sugon-WZ'] # All cloud providers
Clouds_ = Clouds.copy() #   Selected cloud providers

# --------------plot--------------
font2 = {
    'family': 'DejaVu Sans',
    'weight': 'normal',
    'size': 28,
}
font1 = {
    'family': 'DejaVu Sans',
    'weight': 'normal',
    'size': 20,
}
# sns.set_style(
#     "whitegrid", {
#         'font.sans-serif': ['DejaVu Sans'],
#         'font.size': '28',
#         'font.weight': 'normal',
#         "mathtext.fontset": 'stix'
#     })
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['font.size'] = 28
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["mathtext.fontset"] = 'stix'
plt.rcParams['figure.dpi'] = 100
Colors = ['cyan', 'lime', 'k', 'm', 'orange', 'b', 'r', 'g']
Markers = ['.', '^', '+', 'v']
LineStyles = ['solid', (0,(5,1)), 'dashed', 'dashdot', (0, (5, 1)),(0, (3, 1, 1, 1))]
SubLabels = 'abcdefghijklmnopqrstuvwxyz'


ID = int(time.time())
print('ID: ',ID)
SEED = randint(1, 100000)
DEVICE = "cpu"

Paths = {
    'timegan_model': "./models/timegan",
    'timegan_visualization': "./lib/timegan/images",
    'prediction_model': "./models/prediction",
    'variance': output_path+"/vars",
    'images': output_path+"/images",
    'results': output_path,
}
for key in Paths:
    if not os.path.exists(Paths[key]):
        os.makedirs(Paths[key])
Intervals = {'10S':10,'1T': 60, '5T': 300, '10T':600, '15T': 900,'30T':1800}

