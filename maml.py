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
    
    def clip_grad(self, grad, max_norm): #TODO : Clip the gradient of the network
        """
        Clip the gradient of the network
        grad : the gradient of the network
        max_norm : the maximum norm of the gradient
        """
        total_norm = 0 # total norm is the sum of the norm of each gradient
        cnt = 0 # count the number of gradients
        for g in grad:
            param_norm = g.data.norm(2) # calculate the 2-norm of the gradient
            total_norm += param_norm.item() ** 2 # add the 2-norm of the gradient to the total norm
            # item is used to get the value of a tensor (torch.tensor -> float)
            cnt += 1
        total_norm = total_norm ** 0.5 # calculate the total norm
        clip_coef = max_norm / (total_norm + 1e-6) # calculate the clip coefficient
        # use max_norm / total_norm because all the gradients will be multiplied by the clip coefficient
        # # we should confirm that the total norm of the gradient is less than max_norm
        if clip_coef < 1:
            g.data.apply_(lambda x: x * clip_coef) # clip the gradient
        return total_norm / cnt # return the average norm of the gradient
    
    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """
        x_spt : support set [batch_size, n_way * k_spt, channel, width, height]
        y_spt : support set [batch_size, n_way * k_spt]
        x_qry : query set   [batch_size, n_way * k_qry, channel, width, height] 
        y_qry : query set   [batch_size, n_way * k_qry]
        """
        