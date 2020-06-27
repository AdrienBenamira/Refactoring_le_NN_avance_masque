import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

import math

#DoReFaNet
class Linear_bin(nn.Module):
    def __init__(self, args, input):
        super(Linear_bin, self).__init__()
        self.infl_ratio=1
        self.args = args
        self.fc1 = BinarizeLinear(input, 2048*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(2048*self.infl_ratio)
        self.fc2 = BinarizeLinear(2048*self.infl_ratio, 1024)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = BinarizeLinear(1024, 512)
        self.htanh3 = nn.Hardtanh()
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 1)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc3(x)
        x = self.drop(x)
        x = self.bn3(x)
        x = self.htanh3(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)
        return x






import torch.nn as nn

class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)
        return out


def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)
