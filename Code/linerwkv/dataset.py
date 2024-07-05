import torch
import numpy as np
import h5py

import os 
import glob
import rasterio
import random

from PIL import Image




class HySpecNet11k(torch.utils.data.Dataset):
   
    def __init__(self, config, mode="train"):
       
        self.root_dir = config.dataset_dir
        self.config = config
        self.mode = mode
        
        if config.dataset_difficulty == "easy":
            in_directory = os.path.join(self.root_dir, "splits","easy")
        else:
            in_directory = os.path.join(self.root_dir, "splits","hard")

        if mode == "train":
            with open(os.path.join(in_directory, "train.csv"), "r") as f:
                self.files = f.readlines()
        if mode == "val":
            with open(os.path.join(in_directory, "val.csv"), "r") as f:
                self.files = f.readlines()[-4:]
        if mode == "test":
            with open(os.path.join(in_directory, "test.csv"), "r") as f:
                self.files = f.readlines()

        self.files = [os.path.join(self.root_dir, "patches", f.strip()) for f in self.files]

        invalid_channels = [126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 160, 161, 162, 163, 164, 165, 166]
        self.valid_channels_ids = [c+1 for c in range(224) if c not in invalid_channels]

        self.minimum_value = 0
        #self.maximum_value = 10000
        self.maximum_value = 1000

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        data = rasterio.open(self.files[idx])
        # remove nodata channels and randomly subsample 8 channels 
        if self.mode=="train":
            start_ch = random.randint(0,len(self.valid_channels_ids)-16)      
            cur_channels = self.valid_channels_ids[start_ch:start_ch+16]
            #start_ch = random.randint(0,len(self.valid_channels_ids)-100)      
            #cur_channels = self.valid_channels_ids[start_ch:start_ch+100]
            #cur_channels = self.valid_channels_ids
        else:
            cur_channels = self.valid_channels_ids
            #cur_channels = self.valid_channels_ids[85:135]
        image = data.read(cur_channels)
        image = image.astype(np.float32)
        # clip data to remove uncertainties
        #image = np.clip(image, a_min=self.minimum_value, a_max=self.maximum_value)
        # [0,1] normalization
        image = (image - self.minimum_value) / (self.maximum_value - self.minimum_value)
        # standardization
        #image = (image - np.mean(image)) / np.std(image)

        image = np.transpose(image, (1,2,0)) # (H,W,C)        

        label = image[1:,:,:] # first line need to be dealt separately (T,C,I-1)
        image = image[:-1,:,:] # cut last line as it will predict beyond what we have (T,C,I)

        return image, label
    
