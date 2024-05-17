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
import torch.nn.functional as F
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

#---------------------- DATA PATH DEFINATION ----------------------

original_size = (1024, 2048)
train_size = (401, 401)
root_path = r"/home/ngxxfus/Desktop/PSPNet"
dataset_path = r"/home/ngxxfus/Downloads/DATASET"
img_path  = dataset_path + r"/IMG"
mask_path  = dataset_path + r"/MASK"
train_txt = dataset_path + r"/trainval.txt"
test_txt  = dataset_path + r"/test.txt"
null_img_path = root_path + r"/Default_Mask/null_img.png"

#---------------------- DATASET DEFINATION ----------------------

CLASSES = [
    "VOID",    "DUONG_DI",    "LAN_HIEN_TAI",
    "LAN_TRAI_0",    "LAN_PHAI_0",    "VOID",
    "VOID",    "VOID",    "VOID",    "VOID",
    "VOID",    "VOID",    "VOID",    "VOID",
    "VOID",    "VOID",    "VOID",    "VOID",
    "VOID",    "VOID",    "VOID",    ]

COLORMAP = [
    [0, 0, 0],    [128, 0, 0],    [0, 128, 0],
    [128, 128, 0],    [0, 0, 128],    [128, 0, 128],
    [0, 128, 128],    [128, 128, 128],    [64, 0, 0],
    [192, 0, 0],    [64, 128, 0],    [192, 128, 0],
    [64, 0, 128],    [192, 0, 128],    [64, 128, 128],
    [192, 128, 128],    [0, 64, 0],    [128, 64, 0],
    [0, 192, 0],    [128, 192, 0],    [0, 64, 128],
]

class cityScapeDataset(Dataset):
    def __init__(self, root_dir, dataset_dir, txt_file, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.txt_file = txt_file
        self.transform = transform
        self.dataset_dir = dataset_dir
        self.img_path_list = []

        # get filename without extension
        with open(self.txt_file) as file_:
                for line in file_:
                    self.img_path_list.append(line.split('.')[0])


    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        # print("\nOpening",self.img_path_list[idx],"\n")
        
        image_path = os.path.join(self.dataset_dir, "IMG", "{}.png".format(self.img_path_list[idx]))
        mask_path = os.path.join(self.dataset_dir, "MASK", "{}.png".format(self.img_path_list[idx]))
            
        if not os.path.exists(image_path):
            print("\nImage {} not found!\n".format(self.img_path_list[idx]))
            sys.exit()
        else:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            
        if not os.path.exists(mask_path):
            print("\nMask {} not found!\n".format(self.img_path_list[idx]))
            mask = cv2.imread(null_img_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            transformed_image = transformed["image"]
            transformed_mask = transformed["mask"]
            return transformed_image, transformed_mask
        return image, mask


#---------------------- TRANSFORM ----------------------

train_transform = A.Compose([
    A.Resize(width=train_size[0], height=train_size[1]),
    # A.HorizontalFlip(),
    A.RandomBrightnessContrast(),
    A.Blur(),
    A.Sharpen(),
    A.RGBShift(),
    #max_holes=None, max_height=None, max_width=None, min_holes=None, min_height=None, min_width=None, fill_value=0
    A.CoarseDropout(max_holes=5, max_height=25, max_width=25, fill_value=255),
    A.CoarseDropout(max_holes=5, max_height=25, max_width=25, fill_value=0),
    # A.Cutout(num_holes=5, max_h_size=25, max_w_size=25, fill_value=0),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
    ToTensorV2(),
])

test_trainsform = A.Compose([
    A.Resize(width=train_size[0], height=train_size[1]),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
    ToTensorV2(), # numpy.array -> torch.tensor (B, 3, H, W)
])


#------------------ declaration UnNormalize --------------------

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

#------------------ Declare dataset  --------------------
train_dataset = cityScapeDataset(root_path, dataset_path, train_txt, train_transform)
test_dataset = cityScapeDataset(root_path, dataset_path, test_txt, test_trainsform)


# image, mask = train_dataset.__getitem__(13)
# print(mask.shape)
# plt.subplot(1, 2, 1)
# plt.imshow(unorm(image).permute(1, 2, 0))
# plt.subplot(1, 2, 2)
# plt.imshow(mask)
# plt.show()

