#------------- import custom libs ---------------------
from Unet_Custom_Dataset_Def import *

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

#------------- AverageMeter ---------------------

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#------------- accuracy_function ---------------------
def accuracy_function(preds, targets):
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()
    acc = torch.sum(preds_flat == targets_flat)
    return acc/targets_flat.shape[0]

