import torch
import numpy as np
import config
from maml import Meta
from DataMaker import omniglot
import tqdm

def main(args, net):
    # here we can set random seed (ignore)
    device = torch.device(args.device)
    model = Meta(args, net).to(device)
    # caculate the number of parameters in the model
    print(f'Total parameters: {sum(map(lambda x: np.prod(x.shape), model.parameters()))}')
    # load the dataset
    data_iter = omniglot(args)
    
    # define a list to store the accuracy 
    acc_list = []
    # train the model
    Trange = tqdm.trange(args.epoch)
    for step in Trange:
        support_task, support_label, query_task, query_label = data_iter.next('train')
        support_task, support_label, query_task, query_label = torch.from_numpy(support_task), torch.from_numpy(support_label), torch.from_numpy(query_task), torch.from_numpy(query_label)
        acc = model(support_task, support_label, query_task, query_label) # train
        acc_list.append(acc)
        mesage = f'Epoch: {step}, Accuracy: {acc}'
        if step % 50 == 0: # test the model
            support_task, support_label, query_task, query_label = data_iter.next('test')
            support_task, support_label, query_task, query_label = torch.from_numpy(support_task), torch.from_numpy(support_label), torch.from_numpy(query_task), torch.from_numpy(query_label)
            accs = []
            for support_task_one, support_label_one, query_task_one, query_label_one in zip(support_task, support_label, query_task, query_label):
                accs.append(model.finetunning(support_task_one, support_label_one, query_task_one, query_label_one))
            mesage += f'\nTest Accuracy: {np.array(accs).mean(axis=0).astype(np.float16)}'
        # print(mesage)
        Trange.set_description(mesage)
    # save the model
    torch.save(model.state_dict(), args.save_path)

if __name__ == '__main__':
    main(config.load_args(), config.load_net()) 