
import os
import csv
import math
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class fashionData(Dataset):
    def __init__(self, root, attribute, split, data_type, transform = None):
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
                if row[1] == attribute:
                    csvdata.append(row)

        for row in range(int(math.floor(split * len(csvdata)))):
            csvdata[row][0] = os.path.join(self.root_path, csvdata[row][0])
            self.train_list.append(csvdata[row])
        for row in range(int(math.ceil(split * len(csvdata))), len(csvdata)):
            csvdata[row][0] = os.path.join(self.root_path, csvdata[row][0])
            self.test_list.append(csvdata[row])

    def __getitem__(self, index):
        class_label = []
        if self.data_type == 'train':
            img_path = self.train_list[index][0]
            y_index = self.train_list[index][2].find('y')
        elif self.data_type == 'test':
            img_path = self.test_list[index][0]
            y_index = self.test_list[index][2].find('y')

        class_label.append(y_index)

        with Image.open(img_path) as img:
            image = img.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(class_label, dtype = torch.long)

    def __len__(self):
        if self.data_type == 'train':
            return len(self.train_list)
        elif self.data_type == 'test':
            return len(self.test_list)