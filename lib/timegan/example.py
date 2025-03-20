'''
Author: yooki(yooki.k613@gmail.com)
LastEditTime: 2025-03-20 20:27:06
Description: This file contains an example implementation of TimeGAN. It demonstrates how to use TimeGAN for workload prediction.
Note: 
    >>> python lib/timegan/example.py 
'''

import argparse
from timegan import TimeGAN,torch,TimeDataset,np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./lib/timegan/data/stock_data.csv")
    parser.add_argument("-g","--gpu", type=int, default=0)
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--iteration", type=int, default=10000)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--save_model", action="store_true",default=True)
    parser.add_argument("--is_train", type=int, default=1)
    parser.add_argument("--is_pre", type=int, default=0)
    args = parser.parse_args() 
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    dataset = TimeDataset(args.data_path, args.seq_len)
    input_size = dataset[0].shape[1] # dim
    hidden_size = 24
    timegan = TimeGAN(input_size, hidden_size, device)
    if args.is_train > 0:
        timegan.train(args.iteration,dataset)
    cols = ['Open','High','Low','Close','Adj Close','Volume']
    gz_ = timegan.generate(dataset,cols = cols,isEvaluate=True)
    data = dataset.concat_data(np.asarray(dataset.samples))
    gz = dataset.concat_data(gz_)
    fig, axs = plt.subplots(len(cols), 1)
    for j, col in enumerate(cols):
        x = np.arange(len(data))
        axs[j].plot(x, data[:, j], label='Real')
        axs[j].plot(x, gz[:, j], label='Synthetic')
        axs[j].set_title('Synthetic vs Real on ' + col)
        axs[j].set_xlabel('Time')
        axs[j].set_ylabel('Value')
        axs[j].legend()
    plt.show()