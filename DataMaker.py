import os
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class omniglot:
    """
    This class is used to load the omniglot dataset.
    the data set is divided into two parts: train and test.
    train: images_background
    test: images_evaluation
    the structure of the folders is : language/character/png
    the main content of the data is the alphabet of each language
    """
    def init_data(args):
        def read_data(name):
            data = []
            resize_method = transforms.Resize((args.image_size, args.image_size))
            for language_index, language_label in enumerate(os.listdir(os.path.join('data', name))):
                for label_index, label in enumerate(os.listdir(os.path.join('data', name, language_label))):
                    tempdata = []
                    for img_index, img in enumerate(os.listdir(os.path.join('data', name, language_label, label))):
                        filename = os.path.join('data', name, language_label, label, img)
                        img = resize_method(Image.open(filename).convert('L'))
                        tempdata.append(np.array(img).astype(float).reshape(1, args.image_size, args.image_size) / 255.0)
                    data.append(np.array(tempdata))
            return data
        
        train_data = np.array(read_data('images_background')) # (19280, 1, x, x)
        test_data = np.array(read_data('images_evaluation')) # (13180, 1, x, x)
        return train_data, test_data
    
    def __init__(self, args):
        if os.path.exists(os.path.join(args.data_path, 'data.npy')):
            self.train_data, self.test_data = np.load(os.path.join(args.data_path, 'data.npy'))
        else:
            self.train_data, self.test_data = self.init_data(args)
            np.save(os.path.join(args.data_path, 'data.npy'), (self.train_data, self.test_data)) # save the data
        self.batch_size = args.task_number
        self.n_class = self.train_data.shape[0] + self.test_data.shape[0]
        self.n_way = args.n_way # n-way means the taks have n-way classes
        self.k_shot = args.k_spt # k-shot means the task have k-shot samples for each class
        self.k_query = args.k_qry # k-query means the task have k-query samples for each class
        #TODO: can dynamically save the training model without having to train from sctrach

    def load_data_cache(self, data):
        """
        This function is used to prepare the N-shot learning data
        """
        data_cache = []
        for sample in range(10):
            support_set_feature, support_set_label, query_set_feature, query_set_label = [], [], [], []
            for i in range(self.batch_size):
                support_x, support_y, query_x, query_y = [], [], [], []
                selected_class = np.random.choice(data.shape[0], self.n_way, replace=False)
                for j, current_class in enumerate(selected_class): # for each selected class
                    selected_image = np.random.choice(20, self.k_shot+self.k_query, replace=False)
                    support_x.append(data[current_class][selected_class[:self.k_shot]])
                    support_y.append([j for _ in range(self.k_shot)])
                    query_x.append(data[current_class][selected_class[self.k_shot:]])
                    query_y.append([j for _ in range(self.k_query)])
                permutation = np.random.permutation(self.n_way * self.k_shot) # (self.n_way * self.k_shot) total of data
                support_x = np.array(support_x).reshape(self.n_way * self.k_shot, 1, self.image_size, self.image_size)[permutation] # shuffle the data
                


                
    def next(self, mode):
        # get the next batch of data
        pass