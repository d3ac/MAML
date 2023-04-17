import torch
import numpy as np
import config
from maml import meta
from DataMaker import omniglot

def main(args, net):
    # here we can set random seed (ignore)
    device = torch.device(args.device)
    model = meta(args, net).to(device)
    # caculate the number of parameters in the model
    print(f'Total parameters: {sum(map(lambda x: np.prod(x.shape), model.parameters()))}')
    # load the dataset
    data_iter = omniglot(args)
    
    # define a list to store the accuracy 
    acc_list = []
    # train the model
    for step in range(args.epoch):
        support_task, support_label, query_task, query_label = data_iter.next('train') #TODO remember to move the data to GPU
        acc = model(support_task, support_label, query_task, query_label) # train
        acc_list.append(acc)
        mesage = f'Epoch: {step}, Accuracy: {acc}'
        if step % 100 == 0: # test the model
            support_task, support_label, query_task, query_label = data_iter.next('test')
            accs = []
            for support_task_one, support_label_one, query_task_one, query_label_one in zip(support_task, support_label, query_task, query_label):
                accs.append(model.finetunning(support_task_one, support_label_one, query_task_one, query_label_one))
            mesage += f', Test Accuracy: {np.mean(accs)}'
        print(mesage)

if __name__ == '__main__':
    main(config.load_args(), config.load_net())