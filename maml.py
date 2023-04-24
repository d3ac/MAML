import torch
from torch import nn
import numpy as np

class Meta(nn.Module):
    def __init__(self, args, net):
        super(Meta, self).__init__()
        # set the parameters of the config
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_qry = args.k_qry
        self.k_spt = args.k_spt
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        # set the network
        self.net = net
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.meta_lr)
    
    def clip_grad(self, grad, max_norm):
        