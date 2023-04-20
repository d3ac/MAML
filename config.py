import argparse
from torch import nn

def load_args():
    arg = argparse.ArgumentParser()
    arg.add_argument('--epoch', type=int, help='epoch number', default=40000)
    arg.add_argument('--n_way', type=int, help='n way', default=5)
    # note that k_spt + k_qry < 20, 20 is the number of samples for each class
    arg.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    arg.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    arg.add_argument('--image_size', type=int, help='image size', default=28) # reduce the size to accelerate
    arg.add_argument('--imgc', type=int, help='image channel', default=1)
    arg.add_argument('--task_number', type=int, help='meta batch size', default=256)
    arg.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    arg.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
    arg.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    arg.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    arg.add_argument('--device', type=str, help='device', default='cuda:0')
    arg.add_argument('--data_path', type=str, help='data path', default='data')
    return arg.parse_args()

def load_net():
    net = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        nn.BatchNorm2d(64), 
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        nn.BatchNorm2d(64), 
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Flatten(),
        nn.Linear(256*(load_args().image_size**2), load_args().n_way)
    )
    return net