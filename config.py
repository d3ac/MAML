import argparse
from torch import nn
import copy
import mamlnet

def load_args():
    arg = argparse.ArgumentParser()
    arg.add_argument('--epoch', type=int, help='epoch number', default=1000)
    arg.add_argument('--n_way', type=int, help='n way', default=5)
    # note that k_spt + k_qry < 20, 20 is the number of samples for each class
    arg.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    arg.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    arg.add_argument('--image_size', type=int, help='image size', default=28) # reduce the size to accelerate
    arg.add_argument('--imgc', type=int, help='image channel', default=1)
    arg.add_argument('--task_num', type=int, help='meta batch size', default=64)
    arg.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    arg.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
    arg.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    arg.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    arg.add_argument('--device', type=str, help='device', default='cuda:0')
    arg.add_argument('--data_path', type=str, help='data path', default='data')
    return arg.parse_args()

def create_network():
    return [
        ('conv2d', [64, 1, 3, 3, 2, 0]),
        ('relu', [True]),
        ('batch_norm', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('batch_norm', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('batch_norm', [64]),
        ('conv2d', [64, 64, 2, 2, 1, 0]),
        ('relu', [True]),
        ('batch_norm', [64]),
        ('flatten', []),
        ('linear', [load_args().n_way, 64])
    ]

def load_net():
    net = mamlnet.Networks(create_network(), load_args())
    return net