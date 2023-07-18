import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class Networks(nn.Module):
    """
    this class is used to create the network for maml
    we need special network to deal with the second order derivative
    """
    def __init__(self, net, config):
        super(Networks, self).__init__()
        self.params = nn.ParameterList()
        self.params_batch_norm = nn.ParameterList()
        self.net = net
        self.config = config
        for name, param in net:
            if name == 'conv2d':
                w = nn.Parameter(torch.ones(param[:4]))
                b = nn.Parameter(torch.zeros(param[0]))
                torch.nn.init.kaiming_normal_(w)
                self.params.append(w)
                self.params.append(b)
            elif name == 'linear':
                w = nn.Parameter(torch.ones(*param))
                b = nn.Parameter(torch.zeros(param[0]))
                torch.nn.init.kaiming_normal_(w)
                self.params.append(w)
                self.params.append(b)
            elif name == 'batch_norm':
                w = nn.Parameter(torch.ones(param[0]))
                b = nn.Parameter(torch.zeros(param[0]))
                self.params.append(w)
                self.params.append(b)
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.params_batch_norm.extend([running_mean, running_var])
            elif name in ['flatten', 'relu', 'tanh', 'sigmoid', 'max_pool2d']:
                continue
            else:
                raise NotImplementedError
    
    def forward(self, x, params=None, bn_training=True):
        if params is None:
            params = self.params
        idx, bn_idx = 0, 0
        for name, param in self.net:
            if name == 'conv2d':
                x = F.conv2d(x, params[idx], params[idx+1], stride=param[4], padding=param[5])
                idx += 2
            elif name == 'linear':
                x = F.linear(x, params[idx], params[idx+1])
                idx += 2
            elif name == 'batch_norm':
                x = F.batch_norm(x, self.params_batch_norm[bn_idx], self.params_batch_norm[bn_idx+1], weight=params[idx], bias=params[idx+1], training=bn_training)
                idx += 2
                bn_idx += 2
            elif name == 'flatten':
                x = x.view(x.size(0), -1) # (batch, -1)
            elif name == 'relu':
                x = F.relu(x, inplace=param[0])
            elif name == 'tanh':
                x = F.tanh(x)
            elif name == 'sigmoid':
                x = F.sigmoid(x)
            elif name == 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            else:
                raise NotImplementedError
        return x
    
    def zero_grad(self, params=None):
        with torch.no_grad():
            if params is None:
                params = self.params
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()

    def parameters(self):
        return self.params