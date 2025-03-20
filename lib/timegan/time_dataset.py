'''
Author: yooki(yooki.k613@gmail.com)
LastEditTime: 2024-07-02 17:50:14
Description: TimeGAN dataset
'''
import numpy as np
import torch

class TimeDataset(torch.utils.data.Dataset):
    def __init__(self, data_, seq_len):
        """
        Args:
        -----------
            data_: str or np.array, data file path or data
            seq_len: int, sequence length
        """
        if type(data_) is str:
            data = np.loadtxt(data_, delimiter=",", skiprows=1)
            data = data[::-1]
        else:
            data = data_
        norm_data = self.normalize(data)
        seq_data = []
        for i in range(len(norm_data) - seq_len + 1):
            x = norm_data[i : i + seq_len]
            seq_data.append(x)
        self.samples = []
        # Random shuffle
        self.idx = np.random.permutation(len(seq_data))
        for i in range(len(seq_data)):
            self.samples.append(seq_data[self.idx[i]])

    def normalize(self,data):
        """Normalize the data to [0,1]

        Args:
        -----------
            data: np.array, original data

        Returns:
        -----------
            norm_data: np.array
        """
        self.mins = np.min(data, 0)
        self.maxs = np.max(data, 0)
        numerator = data - self.mins
        denominator = self.maxs - self.mins
        norm_data = numerator / (denominator + 1e-7)
        return norm_data

    def renormalize(self, norm_data):
        """Renormalize the data to the original scale

        Args:
        -----------
            norm_data: np.array, normalized data
        
        Returns:
        -----------
            data: np.array
        """
        numerator = norm_data * (self.maxs - self.mins + 1e-7)
        data = numerator + self.mins
        return data
        
    def concat_data(self, data, dataset_size):
        """Recover the order of the shuffled data

        Args:
        -----------
            data: np.array, generated data
            dataset_size: int, original data size

        Returns:
        -----------
            recover_data: np.array
        """
        res = np.zeros((dataset_size,data.shape[2]))
        i,ii=0,0
        while i < dataset_size:
            if i + data.shape[1] <= len(res):
                res[i:i+data.shape[1],:]=data[ii, :, :]
            else:
                res[i:,:]=data[-1, data.shape[1]-len(res)%data.shape[1]:, :]
            i = i + data.shape[1]
            ii = ii + 1
        return res
    

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)
