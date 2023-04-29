import torch
from torch import nn
from torch.nn import functional as F
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
    
    def forward(self, x_spt, y_spt, x_qry, y_qry, device=torch.device('gpu')):
        """
        x_spt : support set [batch_size, n_way * k_spt, channel, width, height]
        y_spt : support set [batch_size, n_way * k_spt]
        x_qry : query set   [batch_size, n_way * k_qry, channel, width, height] 
        y_qry : query set   [batch_size, n_way * k_qry]
        """
        task_num, set_size, channel, width, height = x_spt.size() # get the size of the support set
        y_spt = y_spt.long()
        y_qry = y_qry.long()
        loss_query = np.zeros(self.update_step + 1) # loss of the query set
        corr_query = np.zeros(self.update_step + 1) # correct number of the query set
        
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.update_lr) # set the optimizer
        x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device) #TODO speed test, whether it is necessary to move the data to GPU
        
        for i in range(task_num): # meta_bach_size default is 64
            X, y = x_spt[i], y_spt[i] # support set
            # fast weight begin with the same value of the slow weight, but it update once
            loss = F.cross_entropy(self.net(X), y)
            grad = torch.autograd.grad(loss, self.net.parameters()) # calculate the gradient of the loss
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))
            
            # The accuracy of zero shot is calculated here
            with torch.no_grad():
                y_hat = self.net(x_qry[i])
                loss_query[0] += F.cross_entropy(y_hat, y_qry[i]) # calculate the loss of the query set
                y_hat = F.softmax(y_hat, dim=1).argmax(dim=1)
                corr_query[0] += torch.eq(y_hat, y_qry[i]).sum().item() # calculate the correct number of the query set
                # eq function will compare two tensor and return Ture or False
            
            # The accuracy of one shot is calculated here
            with torch.no_grad():
                y_hat = self.net(x_qry[i], fast_weights)
                loss_query[1] += F.cross_entropy(y_hat, y_qry[i])
                y_hat = F.softmax(y_hat, dim=1).argmax(dim=1)
                corr_query[1] += torch.eq(y_hat, y_qry[i]).sum().item()
            
            for k in range(1, self.update_step): # update_step is task-level inner update steps, default is 5
                # train
                y_hat = self.net(x_spt[i], fast_weights)
                loss = F.cross_entropy(y_hat, y_spt[i])
                grad = torch.autograd.grad(loss, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                # test
                y_hat = self.net(x_qry[i], fast_weights)
                loss_query[k + 1] += F.cross_entropy(y_hat, y_qry[i]) # calculate the loss of the query set after k update
                # calculate the correct number of the query set after k update
                with torch.no_grad():
                    y_hat = F.softmax(y_hat, dim=1).argmax(dim=1)
                    corr_query[k + 1] += torch.eq(y_hat, y_qry[i]).sum().item()
        # calculate the second order derivative
        self.optimizer.zero_grad()
        loss = loss_query[-1] / task_num
        loss.backward()
        self.optimizer.step()
        return np.array(corr_query) / (task_num * x_qry.size(1))