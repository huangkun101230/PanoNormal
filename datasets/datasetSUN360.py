import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import torch
import numpy as np
import cv2
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt

import socket
import random

localname = str(socket.getfqdn()).split('.')[0]


def flip_y(ang):
    return np.array([-np.cos(ang), 0, np.sin(ang), 0, 1, 0, -np.sin(ang), 0, np.cos(ang)]).reshape(3,3)

def rot_x(ang):
    return np.array([1, 0, 0, 0, np.cos(ang), -np.sin(ang), 0, np.sin(ang), np.cos(ang)]).reshape(3,3)

class DatasetSUN360(Dataset):
    def __init__(self, filenames_filepath, height, width, color_augmentation=True,
                 rotate=True,
                 flip=True,
                 is_training=False):
        
        self.rotate = rotate
        self.flip = flip
        self.is_training = is_training
        self.length = 0
        self.height = height
        self.width = width
        self.filenames_filepath = filenames_filepath
        self.delim = " "
        self.color_augmentation = color_augmentation
        self.data_paths = {}
        self.init_data_dict()
        self.gather_filepaths()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                

    def init_data_dict(self):
        self.data_paths = {
            "rgb": [],
        }
    
    def gather_filepaths(self):
        fd = open(self.filenames_filepath, 'r')
        lines = fd.readlines()
        for line in lines:
            rgb_path = line.replace('\n','')
            self.data_paths["rgb"].append(str(rgb_path))

        fd.close()
        assert len(self.data_paths["rgb"])
        self.length = len(self.data_paths["rgb"])

    def load_rgb(self, filepath):
        if not os.path.exists(filepath):
            print("\tGiven filepath <{}> does not exist".format(filepath))
            return np.zeros((self.height, self.width, 3), dtype = np.float32)
        rgb_np = cv2.imread(filepath, cv2.IMREAD_ANYCOLOR)
        rgb_np = cv2.resize(rgb_np, dsize=(self.width, self.height), interpolation=cv2.INTER_CUBIC)
        rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2RGB)
        return rgb_np


    def load_item(self, idx):
        item = { }
        if (idx >= self.length):
            print("Index out of range.")
        else:
            rgb_np = self.load_rgb(self.data_paths["rgb"][idx])
            # plt.imshow(rgb_np)
            # plt.show()
            rgb = self.to_tensor(rgb_np.copy())
            item['rgb'] = self.normalize(rgb)
            item['filename'] = os.path.splitext(os.path.basename(self.data_paths["rgb"][idx]))[0]
            return item
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.load_item(idx)