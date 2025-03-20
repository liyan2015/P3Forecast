'''
Author: yooki(yooki.k613@gmail.com)
LastEditTime: 2025-03-20 21:23:41
Description: Preprocess the cloud raw data

(1) cal_stats: Calculate the average and standard deviation of the resource utilization of the cloud cluster and save it.
'''
from P3Forecast.lib.utils import read_dataSet,get_time,parse_string,parse_time,print_time_lag,np,pd,plt,datetime
from tqdm import tqdm
from parameters import dataGoogle_path,Clouds,Intervals,Colors
def cal_stats(cloud_type, columns=['cpu_util'],
                freqs={
                'Alibaba': '10S',
                'Azure': '1T',
                'Google': '5T',
                'Alibaba-ML': '10T',
                'Sugon-KS': '10T',
                'Sugon-HF': '5T',
                'Sugon-WZ': '10T'    
            },
            isShow=True):
    '''
    Calculate the average and standard deviation of the resource utilization of the cloud cluster and save it.

    Args:
    ------------
        cloud_type: str, the name of the cloud cluster
        freqs: dict, the frequency of the cloud cluster
        isShow: bool, whether to show the plot
    '''
    start_time = datetime.datetime.now()
    try:
        freq = freqs.get(cloud_type)
    except ValueError:
        print('当前不支持云提供商：' + cloud_type)
        return
    if cloud_type == 'Google':
        df = cal_Google_stats(freq,columns)
    elif cloud_type == 'Alibaba-ML':
        df = cal_AlibabaML_stats(freq,columns)
    elif 'Sugon' in cloud_type:
        i = Clouds.index(cloud_type)
        df = cal_Sugon_stats(i,freq,columns)
    else:
        dfs = []
        df = read_dataSet(cloud_type, isRaw=True)
        df.dropna(inplace=True)
        groups = df.groupby('time')
        for column in columns:
            res = groups.agg({column: [np.mean, np.std],})
            def transTime(x, cloud_type):
                if cloud_type == 'Alibaba':
                    return datetime.datetime.fromtimestamp(x + 1514736000)
                elif cloud_type == 'Azure':
                    return parse_string(x[:-7])

            res.index = res.index.map(lambda x: transTime(x, cloud_type))

            T = pd.date_range(start=parse_time(res.index[0]),
                            end=parse_time(res.index[-1]),
                            freq=freq)
            res = res.reindex(index=T)
            
            df_ = res[column]
            df_.reset_index(inplace=True)
            df_ = df_.rename(columns={
                'index': 'time',
                'mean': column,
                'std': '%s_std' % column.split('_')[0]
            })
            dfs.append(df_)
        # 根据time合并dfs
        df = dfs[0]
        if len(dfs) > 1:
            for i in range(1,len(dfs)):
                df = pd.merge(df, dfs[i], on='time')

    df.to_csv(f'data/{cloud_type}_{freq}_new.csv', index=False)
    print_time_lag(start_time, datetime.datetime.now(), cal_stats.__name__)
    if isShow:
        plt.figure(figsize=(15, 5))
        for i,column in enumerate(columns):
            plt.plot(df['time'], df[column], Colors[i], label=column)
        plt.title('Workload of %s' % (cloud_type))
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Workload(%)')
        plt.show()



def cal_Google_stats(freq, columns):
    """
    Calculate the average and standard deviation of the resource utilization of the ``Google`` cluster

    Args:
    ------------
        freq: int, the frequency of the Google cluster
        columns: list, the columns of the Google cluster
    
    Returns:
    ------------
        df_: pd.DataFrame, the average and standard deviation of the resource utilization of the Google cluster
    """
    from pandarallel import pandarallel
    pandarallel.initialize()  #nb_workers=4,progress_bar=True
    alls = [np.zeros((2506200, ), dtype=np.float32) for x in columns]
    nums = np.zeros((2506200, ), dtype=np.int32)

    def run(df):
        l = int(df['start_time'].min())
        r = int(df['end_time'].max())
        sums = [np.zeros((r - l, ), dtype=np.float32) for x in columns]

        def run_(x, l):
            for i, col in enumerate(columns):
                sums[i][int(x.start_time) - l:int(x.end_time) - l] += x[col]

        df.apply(run_, axis=1, args=(l, ))
        return [sums, l, r]

    for index_ in tqdm(range(0, 500, 25)):
        df = pd.concat([
            pd.read_csv(
                dataGoogle_path +
                "part-{}-of-00500.csv".format(str(x).zfill(5)),
                names=['start_time', 'end_time', 'machine_id', 'cpu_util', 'mem_util'],
                usecols=[0, 1, 4, 5, 6]) for x in range(index_, index_ + 25)
        ])
        df['start_time'] = (df['start_time'] / 1000000).astype(np.int32)
        df['end_time'] = (df['end_time'] / 1000000).astype(np.int32)
        print('%s: read csv is done'%get_time())
        res = df.groupby(by='machine_id').parallel_apply(run)
        print('%s: run is done'%get_time())
        for mid, x in res.items():
            l, r = x[1], x[2]
            for i in range(len(columns)):
                alls[i][l:r] += x[0][i]
            nums[l:r] += 1
    avgs = [np.divide(x, nums) for x in alls]
    alls = [np.zeros((2506200, ), dtype=np.float32) for x in columns]

    def run1(df):
        l = int(df['start_time'].min())
        r = int(df['end_time'].max())
        sums = [np.zeros((r - l, ), dtype=np.float32) for x in columns]

        def run1_(x, l):
            for i, col in enumerate(columns):
                sums[i][int(x.start_time) - l:int(x.end_time) - l] += (
                    x[col] - avgs[i][int(x.start_time):int(x.end_time)])**2

        df.apply(run1_, axis=1, args=(l, ))
        return [sums, l, r]

    for index_ in tqdm(range(0, 500, 25)):
        df = pd.concat([
            pd.read_csv(
                dataGoogle_path +
                "part-{}-of-00500.csv".format(str(x).zfill(5)),
                names=['start_time', 'end_time', 'machine_id', 'cpu_util', 'mem_util'],
                usecols=[0, 1, 4, 5, 6]) for x in range(index_, index_ + 25)
        ])
        df['start_time'] = (df['start_time'] / 1000000).astype(np.int32)
        df['end_time'] = (df['end_time'] / 1000000).astype(np.int32)
        print('%s: read csv is done'%get_time())
        res = df.groupby(by='machine_id').parallel_apply(run1)
        for mid, x in res.items():
            l, r = x[1], x[2]
            for i in range(len(columns)):
                alls[i][l:r] += x[0][i]
        print('%s: run1 is done'%get_time())
    stds = [np.divide(np.sqrt(x), nums) for x in alls]
    d = {'time': np.arange(2506200)}
    for i, col in enumerate(columns):
        d[col] = avgs[i]
        d[col.split('_')[0] + '_std'] = stds[i]
    df = pd.DataFrame(d)
    df.dropna(inplace=True)
    bins = np.arange(599, 2506200, Intervals[freq])
    labels = (((bins + 1)[:-1] - 600) / Intervals[freq]).astype(np.int32)
    # (599,899]
    df['period'] = pd.cut(df['time'], bins=bins, labels=labels)
    res = df.groupby(by=['period']).mean().reset_index()
    times = pd.date_range(start='2011-01-01', periods=len(res), freq=freq)
    dd={'time': times}
    for col in columns:
        dd[col] = res[col].values * 100
        dd[col.split('_')[0] + '_std'] = res[col.split('_')[0] + '_std'].values * 100
    return pd.DataFrame(dd)

Month = 1
Temp = 0
def cal_Sugon_stats(i:int,freq:int, columns):
    """ 
    Calculate the average and standard deviation of the resource utilization of the Sugon cluster(i=4 ``Sugon-KS``; i=5 ``Sugon-HF``;i=6 ``Sugon-WZ``)

    Args:
    -----------
        i: int, the index of Sugon cluster
        freq: int, the frequency of the Sugon cluster

    Returns:
    -------------
        DF: pd.DataFrame, the average and standard deviation of the resource utilization of the Sugon clusterq
    """
    unit_index = [7,7,30]
    tts = [
        pd.date_range(start="20220101", end="20220225 15:40", freq=freq), # KS
        pd.date_range(start="20220901 00:00", end="20220927 21:55", freq=freq), # HF
        pd.date_range(start="20220101 00:00", end="20220927 20:30", freq=freq) # WZ
    ]
    cloud_type=Clouds[i]
    df = read_dataSet(cloud_type,True)
    if cloud_type == 'Sugon-WZ':
        def run_(x):
            global Month,Temp
            t1 = str(x.job_queue_time)[:-3]
            t2 = str(x.acct_time)[:-3]
            if int(t2.split(' ')[0])==28:
                Temp = 1
            elif int(t2.split(' ')[0])==1:
                if Temp == 1:   
                    Month+=1
                Temp = 0
            if int(t1.split(' ')[0]) > int(t2.split(' ')[0]):
                if Month == 1:
                    x.job_queue_time = "2022-01-01 00:00"
                else:
                    x.job_queue_time = "2022-{:>02}-{:>02} {}".format(Month-1, t1.split(' ')[0] ,t1.split(' ')[2])
            else:
                x.job_queue_time = "2022-{:>02}-{:>02} {}".format(Month, t1.split(' ')[0], t1.split(' ')[2])
            x.acct_time = "2022-{:>02}-{:>02} {}".format(Month, t2.split(' ')[0], t2.split(' ')[2])
            return x
        df = df.apply(run_, axis=1)
    if cloud_type == 'Sugon-HF':
        def run_(x):
            t1 = datetime.datetime.strptime(str(x.job_queue_time), '%Y-%m-%d %H:%M:%S')
            if t1.month < 9:
                x.job_queue_time = "2022-09-01 00:00"
            return x
        df = df.apply(run_, axis=1)
    times = tts[i-4]
    df['acct_time'] = pd.to_datetime(df['acct_time'])
    df['job_queue_time'] = pd.to_datetime(df['job_queue_time'])
    print('handle time is done')
    # avg
    nums = pd.Series(0,index=times)
    nodes = pd.Series('',index=times)
    sums = [pd.Series(0,index=times) for x in columns]
    def run_avg(x):
        ls = int((x.job_queue_time - times[0]).total_seconds())
        lf = ls//Intervals[freq]
        rs = int((x.acct_time - times[0]).total_seconds())
        rf = rs//Intervals[freq]
            
        if cloud_type=='Sugon-WZ' and str(x.need_nodes) != '':
            if str(x.need_nodes) > 'g01':
                x.cpu_util /= 64
            else:
                x.cpu_util /= 32
        for j in range(lf,rf+1):
            if str(x.need_nodes) != '':
                for node in str(x.need_nodes).split('+'):
                    if node not in nodes[j]:
                        nodes[j] += node + ','
                        nums[j] += 1
        # 1   2   3   4   5
        # 10  
        # 800
        for i,col in enumerate(columns):
            if lf==rf: # 在一个时间段
                sums[i][lf] += x[col]*(rs-ls)/Intervals[freq]
                return x
            if ls%Intervals[freq] != 0: # 左边超出
                sums[i][lf] += x[col]*(1-(ls%Intervals[freq])/Intervals[freq])
                lf += 1
            if rs%Intervals[freq] != 0: # 右边超出
                sums[i][rf] += x[col]*(rs%Intervals[freq])/Intervals[freq]
            if lf<rf: # 中间
                sums[i][lf:rf] += x[col]
        return x
    df = df.apply(run_avg, axis=1)
    print('run_avg is done')
    for i in range(len(columns)):
        sums[i].loc[nums>0] = sums[i].loc[nums>0]/nums.loc[nums>0]
    # std
    sums_ = [pd.Series(0,index=times) for x in columns]
    def run_std(x):
        ls = int((x.job_queue_time - times[0]).total_seconds())
        lf = ls//Intervals[freq]
        rs = int((x.acct_time - times[0]).total_seconds())
        rf = rs//Intervals[freq]
        for i, col in enumerate(columns):
            if lf==rf:
                sums_[i][lf] += (x[col]*(rs-ls)/Intervals[freq]-sums[i][lf])**2
                return
            if ls % Intervals[freq] != 0:
                sums_[i][lf] += (x[col]*(1-(ls%Intervals[freq])/Intervals[freq])-sums[i][lf])**2
                lf += 1
            if rs % Intervals[freq] != 0:
                sums_[i][rf] += (x[col]*(rs%Intervals[freq])/Intervals[freq]-sums[i][rf])**2
            if lf<rf:
                sums_[i][lf:rf] += (x[col] - sums[i][lf:rf])**2   
    df.apply(run_std, axis=1)
    print('run_std is done')
    d={'time':times}
    for i in range(len(columns)):
        sums_[i].loc[nums>0] = sums_[i].loc[nums>0]/nums.loc[nums>0]
        sums_[i] = sums_[i].apply(np.sqrt)
        sums[i].replace(0, np.nan, inplace=True)
        sums_[i].replace(0, np.nan, inplace=True)
        if columns[i] == 'mem_util':# 需要归一化
            d[columns[i]] = (sums[i]-sums[i].min())/(sums[i].max()-sums[i].min())*100
            d[columns[i].split('_')[0] + '_std'] = (sums_[i]-sums_[i].min())/(sums_[i].max()-sums_[i].min())
        else:
            d[columns[i]] = sums[i]
            d[columns[i].split('_')[0] + '_std'] = sums_[i]
    # plt.figure(figsize=(15,3))
    # data = sums.dropna()
    # step = int(datetime.timedelta(days=unit_index[i-4], hours=0,minutes=0).total_seconds() / Intervals[freq])
    # xhandel = range(0, len(data), step)
    # x = []
    # unit = {1: 'Day',
    #         7: 'Week',
    #         30: 'Month'}
    # for j in range(0, len(data), step):
    #     x.append(unit[unit_index[i-4]]+str((j//step)+1))
    # plt.xticks(xhandel, x, rotation=40)
    # plt.plot(range(len(sums)),sums)
    # plt.show()
    return pd.DataFrame(d)

def cal_AlibabaML_stats(freq, columns):
    """
    Calculate the average and standard deviation of the resource utilization of the ``Alibaba-ML`` cluster

    Args:
    ------------
        freq: int, the frequency of the Alibaba-ML cluster
        columns: list, the columns of the Alibaba-ML cluster
    
    Returns:
    ------------
        df_: pd.DataFrame, the average and standard deviation of the resource utilization of the GooAlibaba-MLgle cluster
    """
    from pandarallel import pandarallel
    pandarallel.initialize(progress_bar=True)  #nb_workers=4,
    df = read_dataSet('Alibaba-ML', isRaw=True)
    df.dropna(inplace=True)
    ll = df['start_time'].min()
    rr = df['end_time'].max()
    times = pd.date_range(start='2020-07-01',periods=rr-ll+1,freq='1S')
    alls = [np.zeros((rr - ll + 1, ), dtype=np.float32) for x in columns]
    nums = np.zeros((rr - ll + 1, ), dtype=np.int32)

    def run(df):
        l = int(df['start_time'].min())
        r = int(df['end_time'].max())
        sums = [np.zeros((r - l + 1, ), dtype=np.float32) for x in columns]

        def run_(x, l):
            for i, col in enumerate(columns):
                sums[i][int(x.start_time) - l:int(x.end_time) - l + 1] += x[col]

        df.apply(run_, axis=1, args=(l, ))
        return [sums, l, r]

    df['start_time'] = df['start_time'].astype(np.int32)
    df['end_time'] = df['end_time'].astype(np.int32)

    groups = df.groupby(by='machine_id')
    res = groups.parallel_apply(run)
    print('%s: run is done'%get_time())
    for mid, x in res.items():
        l, r = x[1], x[2]
        l -= ll
        r -= ll
        for i in range(len(columns)):
            alls[i][l:r + 1] += x[0][i]
        nums[l:r + 1] += 1
    avgs = [np.divide(x, nums) for x in alls]
    for i in range(len(columns)):
        alls[i][nums > 0] = alls[i][nums > 0] / nums[nums > 0]
    alls_ = [np.zeros((rr - ll + 1, ), dtype=np.float32) for x in columns]

    def run1(df):
        l = int(df['start_time'].min())
        r = int(df['end_time'].max())
        sums = [np.zeros((r - l + 1, ), dtype=np.float32) for x in columns]

        def run1_(x, l):
            for i, col in enumerate(columns):
                sums[i][int(x.start_time) - l:int(x.end_time) - l + 1] += (
                    x[col] - avgs[i][int(x.start_time)-ll:int(x.end_time)+1-ll])**2
                
        df.apply(run1_, axis=1, args=(l, ))
        return [sums, l, r]

    res = groups.parallel_apply(run1)
    for mid, x in res.items():
        l, r = x[1], x[2]
        l -= ll
        r -= ll
        for i in range(len(columns)):
            alls_[i][l:r + 1] += x[0][i]
    print('%s: run1 is done'%get_time())
    d = {'time': times[::Intervals[freq]]}
    n = pd.Series(nums, index=times).resample(freq).apply(lambda x: (x > 0).sum())
    for i in range(len(columns)):
        alls_[i][nums > 0] = alls_[i][nums > 0] / nums[nums > 0]
        alls_[i] = np.sqrt(alls_[i])
        s = pd.Series(alls[i], index=times).resample(freq).sum()
        s_ = pd.Series(alls_[i], index=times).resample(freq).sum()
        avg = s / n
        std = s_ / n
        avg.replace(0, np.nan, inplace=True)
        d[columns[i]] = avg
        d[columns[i].split('_')[0] + '_std'] = std
    return pd.DataFrame(d)

def resample(cloud_type,freq2,is_save=True):
    '''
    Reduce the sampling frequency of the data and save it. For Alibaba and Azure
    
    Args:
    ------------
        cloud_type: str, the name of the cluster
        freq2: str, the following freq of the cluster
        is_save: bool, whether to save the data
    '''
    index_ = Clouds.index(cloud_type)
    df,freq1 = read_dataSet(Clouds[index_],False)
    freq = Intervals[freq2]//Intervals[freq1]
    df_ = df.rolling(freq).mean()[freq-1::freq]
    time=df['time'].iloc[::freq]
    t = time.to_frame('time')
    if len(t)!=len(df_): #有余数
        l = t.tail(1).index.start
        temp=df.loc[l:].mean()
        df_.loc[l+freq-1]=[temp[x] for x in df.columns if x != 'time']
    d = {'time':t['time'].values}
    for col in df.columns:
        if col == 'time': continue
        d[col] = df_[col].values
    DF=pd.DataFrame(d)
    name = Clouds[index_]
    if name == 'Sugon-WZ':
        name = 'Sugon-Wuzhen'
    elif name == 'Sugon-HF':
        name = 'Sugon-Hefei'
    elif name == 'Sugon-KS':
        name = 'Sugon-Ks'
    if is_save:
        DF.to_csv('./data/{}_{}.csv'.format(Clouds[index_],freq2),index=False)
    return DF

