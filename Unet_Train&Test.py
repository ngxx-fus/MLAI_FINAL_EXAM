#------------- import custom libs ---------------------
from Custom_Dataset_Def import *
from Unet_Def import *

#------------- import built-in libs ---------------------
import numpy as np
import cv2
import matplotlib as plt
import matplotlib.pyplot as plt
import torch
from  torch.utils.data import Dataset, DataLoader
import torchmetrics
from torchmetrics import Dice, JaccardIndex #IOU
import segmentation_models_pytorch as smp
import albumentations as A
# to convert np.array to torch.tensor
from albumentations.pytorch import ToTensorV2
# others
import os
# processing effecting
from tqdm import tqdm
# read all images in a folder
from glob import glob


#------------------ TEST DATASET --------------------
train_dataset = cityScapeDataset(root_path, train_txt, train_transform)
test_dataset = cityScapeDataset(root_path, test_txt, test_transformt)

# test dataset
img, mask = train_dataset.__getitem__(2)
print(img.shape)

plt.subplot(1,2,1)
plt.imshow(unorm(img).permute(1,2,0))
plt.subplot(1,2,2)
plt.imshow(mask)
plt.show()