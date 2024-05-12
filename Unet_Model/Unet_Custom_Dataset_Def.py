#------------- import custom libs ---------------------

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
import os
import pandas as pd
from torchvision.io import read_image

#------------------ declare size, path --------------------

original_size = (1024, 2048)
train_size = (572, 572)
root_path = "."
img_path  = root_path + r"\IMG" # winforms
mask_path  = root_path + r"\MASK"
train_txt = root_path + r"\trainval.txt"
test_txt  = root_path + r"\test.txt"


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
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            transformed_image = transformed["image"]
            transformed_mask = transformed["mask"]
            return transformed_image, transformed_mask
        return image, mask

train_transform = A.Compose([
    A.RandomCrop(height=original_size[0]-50, width=original_size[1]-50),
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
    A.RandomCrop(height=original_size[0]-50, width=original_size[1]-50),
    A.Resize (height=train_size[0], width=train_size[1], interpolation=1, always_apply=True, p=1),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
    ToTensorV2(),
])

#------------------ declaration UnNormalize --------------------
# to show
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

# test dataset
img, mask = train_dataset.__getitem__(60)
plt.subplot(1,2,1)
plt.imshow(unorm(img).permute(1,2,0))
plt.subplot(1,2,2)
plt.imshow(mask)
plt.show()