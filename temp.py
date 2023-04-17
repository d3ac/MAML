import os
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import config
import numpy as np

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
    
    train_data = np.array(read_data('images_background')) # (, 1, x, x)
    test_data = np.array(read_data('images_evaluation')) # (, 1, x, x)
    print(train_data.shape, test_data.shape)
    return train_data, test_data

init_data(config.load_args())