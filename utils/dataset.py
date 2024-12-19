import os
import imageio
import numpy as np
from PIL import Image
from jittor.dataset import Dataset


class CUB():
    def __init__(self, root, data_len=None, train_transform=None, test_transform=None):
        self.root = root
        self.train_transform = train_transform
        self.test_transform = test_transform
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))
        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
        
        self.train_img = [imageio.imread(os.path.join(self.root, 'images', train_file)) for train_file in train_file_list[:data_len]]
        self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
        self.train_imgname = [x for x in train_file_list[:data_len]]
        
        self.test_img = [imageio.imread(os.path.join(self.root, 'images', test_file)) for test_file in test_file_list[:data_len]]
        self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]
        self.test_imgname = [x for x in test_file_list[:data_len]]
    
    def get_train_dataset(self):
        return self.CUBDataset(self.train_img, self.train_label, self.train_imgname, self.train_transform)
    
    def get_test_dataset(self):
        return self.CUBDataset(self.test_img, self.test_label, self.test_imgname, self.test_transform)
    
    class CUBDataset(Dataset):
        def __init__(self, img, label, imgname, transform=None):
            super().__init__()
            self.img = img
            self.label = label
            self.imgname = imgname
            self.transform = transform
        
        def __len__(self):
            return len(self.label)
        
        def __getitem__(self, idx):
            img, target = self.img[idx], self.label[idx]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            if self.transform is not None:
                img = self.transform(img)
            return img, target
        