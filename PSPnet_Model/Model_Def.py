#------------- import custom libs ---------------------
# from Dataset_Def import cityScapeDataset
# from Dataset_Def import train_transform
# from Dataset_Def import test_transformt
# from Dataset_Def import UnNormalize
# from Dataset_Def import unorm
# from Dataset_Def import train_dataset
# from Dataset_Def import test_dataset
from Dataset_Def import train_size

#------------- import built-in libs ---------------------
# import numpy as np
# import cv2
import matplotlib as plt
import matplotlib.pyplot as plt
import torch
# from  torch.utils.data import Dataset, DataLoader
# import torchmetrics
# from torchmetrics import Dice, JaccardIndex #IOU
# import segmentation_models_pytorch as smp
# import albumentations as A
# to convert np.array to torch.tensor
# from albumentations.pytorch import ToTensorV2
# others
# import os
# processing effecting
# from tqdm import tqdm
# read all images in a folder
# from glob import glob

import torch.nn as nn
# import torch.nn.functional as F

#with (conv2d + pooling)x2
def Unet_Block(in_channels, out_channels):
    return nn.Sequential(
        #in_channels, out_channels, kernel_size, stride, padding
        nn.Conv2d(in_channels, out_channels, 3, 1, 0),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3,1, 0),
        nn.ReLU()
    )

class Unet_Model(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        # max pooling, kernel_size = 2
        self.max_pool  = nn.MaxPool2d(2)
        # interpolation
        self.block_down1 = Unet_Block(3, 64)
        self.block_down2 = Unet_Block(64, 128)
        self.block_down3 = Unet_Block(128, 256)
        self.block_down4 = Unet_Block(256, 512)

        self.block_neck  = Unet_Block(512, 1024)

        self.up_conv_1  = nn.Upsample(size=(64,64), mode="bilinear")
        self.block_up1   = Unet_Block(1024 + 512, 512)
        self.up_conv_2  = nn.Upsample(size=(136,136), mode="bilinear")
        self.block_up2   = Unet_Block(512+256, 256)
        self.up_conv_3  = nn.Upsample(size=(280, 280), mode="bilinear")
        self.block_up3   = Unet_Block(256+128, 128)
        self.up_conv_4  = nn.Upsample(size=(568,568), mode="bilinear")
        self.block_up4   = Unet_Block(128+64, 64)
        # B, n_classes, H, W
        self.up_conv_5  = nn.Upsample(size=(train_size[0],train_size[1]), mode="bilinear")
        self.conv_classify = nn.Conv2d(64, self.n_classes, 1, 1)

    def forward(self, x):
        # encoder
        # print("INPUT: ", x.shape)
        x1 = self.block_down1(x)
        # print("DOWN1: ", x1.shape)
        x = self.max_pool(x1)
        # print("MAXPOOL: ", x.shape, "\n")
        x2 = self.block_down2(x)
        # print("DOWN2: ", x2.shape)
        x = self.max_pool(x2)
        # print("MAXPOOL: ", x.shape, "\n")
        x3 = self.block_down3(x)
        # print("DOWN3: ", x3.shape)
        x = self.max_pool(x3)
        # print("MAXPOOL: ", x.shape, "\n")
        x4 = self.block_down4(x)
        # print("DOWN4: ", x4.shape)
        x = self.max_pool(x4)
        # print("MAXPOOL: ", x.shape, "\n")

        #neck
        x = self.block_neck(x)
        # print("NECK: ", x.shape, "\n")

        #decoder
        #[b c=1024 h w] cat [b c=512 h w] -> [b c=1536 h w]  (dim=1 - channel)
        x = torch.cat([x4, self.up_conv_1(x)], dim=1)
        # print("CAT1: ", x.shape)
        x = self.block_up1(x)
        # print("UP1: ", x.shape, "\n")
        x = torch.cat([x3, self.up_conv_2(x)], dim=1)
        # print("CAT2: ", x.shape)
        x = self.block_up2(x)
        # print("UP2: ", x.shape, "\n")
        x = torch.cat([x2, self.up_conv_3(x)], dim=1)
        # print("CAT3: ", x.shape)
        x = self.block_up3(x)
        # print("UP3: ", x.shape, "\n")
        x = torch.cat([x1, self.up_conv_4(x)], dim=1)
        # print("CAT4: ", x.shape)
        x = self.block_up4(x)
        # print("UP4: ", x.shape, "\n")

        x = self.up_conv_5(x)
        return self.conv_classify(x)

#------------- test model ---------------------






