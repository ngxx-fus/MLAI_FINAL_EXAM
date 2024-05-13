#----------------------  custom libs   ----------------------


#---------------------- built-in libs ----------------------
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from  torch.utils.data import Dataset, DataLoader
import torchmetrics
from torchmetrics      import Dice, JaccardIndex #IOU
import segmentation_models_pytorch as smp
import albumentations              as A
#to convert np.array to torch.tensor
from albumentations.pytorch        import ToTensorV2
#others
import os
#processing effecting
from tqdm import tqdm
#read all images in a folder
from glob import glob
import os
import sys
import pandas as pd
from torchvision.io import read_image

#------------------ declare size, path --------------------

original_size = (1024, 2048)
train_size = (572, 572)
root_path = r"D:\DOC\23-24_HK02\MLAI\CityScape_Dataset"
img_path  = root_path + r"\IMG" 
mask_path  = root_path + r"\MASK"
train_txt = root_path + r"\trainval.txt"
test_txt  = root_path + r"\test.txt"


#---------------------- DATASET DEFINATION ----------------------

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

CLASSES = [
    "VOID",
    "DUONG_DI",
    "LAN_HIEN_TAI",
    "LAN_TRAI_0",
    "LAN_PHAI_0",
    "VOID",
    "VOID",
    "VOID",
    "VOID",
    "VOID",
    "VOID",
    "VOID",
    "VOID",
    "VOID",
    "VOID",
    "VOID",
    "VOID",
    "VOID",
    "VOID",
    "VOID",
    "VOID",    
]

COLORMAP = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]

class cityScapeDataset(Dataset):
    def __init__(self, root_dir, txt_file, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.txt_file = txt_file
        self.transform = transform
        self.img_path_list = []

        # get filename without extension
        with open(self.txt_file) as file_:
                for line in file_:
                    #TODO: modify with custom dataset
                    self.img_path_list.append(line.split('.')[0])


    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        #TODO: modify with custom dataset
        image_path = os.path.join(self.root_dir, "IMG", "{}.png".format(self.img_path_list[idx]))
        mask_path = os.path.join(self.root_dir, "MASK", "{}.png".format(self.img_path_list[idx]))
        if not os.path.exists(mask_path) or not os.path.exists(image_path):
            print("\nImage or mask not found!\n")
            sys.exit() 
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            transformed_image = transformed["image"]
            transformed_mask = transformed["mask"]
            return transformed_image, transformed_mask
        return image, mask
    
#---------------------- TRANSFORM ----------------------

train_transform = A.Compose([
    A.RandomCrop(height=original_size[0]-100, width=original_size[1]-100),
    A.Resize (height=train_size[0], width=train_size[1], interpolation=1, always_apply=True, p=1),
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(),
    A.Blur(),
    A.RGBShift(),
    #max_holes=None, max_height=None, max_width=None, min_holes=None, min_height=None, min_width=None, fill_value=0
    A.CoarseDropout(max_holes=5, max_height=25, max_width=25, fill_value=255),
    A.CoarseDropout(max_holes=5, max_height=25, max_width=25, fill_value=0),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
    ToTensorV2(),
])

test_transformt = A.Compose([
    A.RandomCrop(height=original_size[0]-100, width=original_size[1]-100),
    A.Resize (height=train_size[0], width=train_size[1], interpolation=1, always_apply=True, p=1),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
    ToTensorV2(),
])

#------------------ declaration UnNormalize --------------------

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
    
unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

#------------------ Declare dataset  --------------------
train_dataset = cityScapeDataset(root_path, train_txt, train_transform)
test_dataset = cityScapeDataset(root_path, test_txt, test_transformt)

"""
image, mask = train_dataset.__getitem__(10)
print(mask.shape)
plt.subplot(1, 2, 1)
plt.imshow(unorm(image).permute(1, 2, 0))
plt.subplot(1, 2, 2)
plt.imshow(mask)
plt.show()
"""
