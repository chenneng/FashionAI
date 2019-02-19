
import os
import csv
import math
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

import random

class fashionData(Dataset):
    def __init__(self, root, split, data_type, transform = None):
        self.root_path = root
        self.transform = transform
        self.data_type = data_type
        self.label_folder = 'Annotations'
        self.train_label = 'label.csv'
        self.train_list = []
        self.test_list = []

        label_file = os.path.join(self.root_path, self.label_folder, self.train_label)

        csvdata = []
        with open(label_file) as f:
            reader = csv.reader(f)
            for row in reader:
                if row[1] == 'coat_length_labels' or row[1] == 'pant_length_labels' or row[1] == 'skirt_length_labels' or row[1] == 'sleeve_length_labels':
                    csvdata.append(row)
        
        random.shuffle(csvdata)

        for row in range(int(math.floor(split * len(csvdata)))):
            csvdata[row][0] = os.path.join(self.root_path, csvdata[row][0])
            self.train_list.append(csvdata[row])
        for row in range(int(math.ceil(split * len(csvdata))), len(csvdata)):
            csvdata[row][0] = os.path.join(self.root_path, csvdata[row][0])
            self.test_list.append(csvdata[row])

    def __getitem__(self, index):
        class_label = [0] * 29
        attribute = None
        if self.data_type == 'train':
            img_path = self.train_list[index][0]
            y_index = self.train_list[index][2].find('y')
            attribute = self.train_list[index][1]
        elif self.data_type == 'test':
            img_path = self.test_list[index][0]
            y_index = self.test_list[index][2].find('y')
            attribute = self.test_list[index][1]

        if attribute == 'coat_length_labels':
            class_label[y_index] = 1
        elif attribute == 'pant_length_labels':
            class_label[y_index + 8] = 1
        elif attribute == 'skirt_length_labels':
            class_label[y_index + 14] = 1
        elif attribute == 'sleeve_length_labels':
            class_label[y_index + 20] = 1

        with Image.open(img_path) as img:
            image = img.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(class_label, dtype = torch.float)

    def __len__(self):
        if self.data_type == 'train':
            return len(self.train_list)
        elif self.data_type == 'test':
            return len(self.test_list)
