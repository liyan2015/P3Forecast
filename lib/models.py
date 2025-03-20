import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.autograd import Variable
from parameters import DEVICE

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        Residual block

        :param n_inputs: int
        :param n_outputs: int
        :param kernel_size: int, convolution kernel size
        :param stride: int
        :param dilation: int
        :param padding: int
        :param dropout: float, dropout
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding) 
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.1):
        """
        TCN model

        Args:
        ---------------
            num_inputs: int， 
            num_channels: list，hidden_channel for each layer，such as:[25,25,25,25]
            kernel_size: int, convolution kernel size
            dropout: float, drop_out
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  
            out_channels = num_channels[i] 
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)  

    def forward(self, x):
        """

        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        inputs = inputs.permute(0, 2, 1)
        y1 = self.tcn(inputs)
        y1 = y1.permute(0, 2, 1)
        o = self.linear(y1)
        return o[:,-1,:] 

class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers,
                 batch_fist=True, dropout=0.2):
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True,dropout=dropout)
        self.num_directions = 1  
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.batch_fist = batch_fist

    def forward(self, x):
        if self.batch_fist:
            batch_size = x.size()[0]
        else:
            batch_size = x.size()[1]
        h_0 = Variable(
            torch.randn(self.num_directions * self.n_layers, batch_size,
                        self.hidden_dim)).to(DEVICE)
        c_0 = Variable(
            torch.randn(self.num_directions * self.n_layers, batch_size,
                        self.hidden_dim)).to(DEVICE)
        output, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(self.relu(output[:, -1, :]))
        return out

class GRU(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 batch_first=True,
                 dropout=0.2):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.gru = nn.GRU(input_dim,
                          hidden_dim,
                          n_layers,
                          batch_first=batch_first,
                          dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.batch_first:
            batch_size = x.size()[0]
        else:
            batch_size = x.size()[1]
        h = self.init_hidden(batch_size)
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:, -1, :]))
        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        return hidden




