#---------------------- local func and var ----------------------
from Dataset_Def  import *
from Backbone_Def import *
from PSPNet_def   import *

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

#---------------------- WEIGHTS PATH DEFINATION ----------------------
# weights_path = r"./Model_Weights"

#---------------------- TRAIN MODEL DEFINATION ----------------------

#loss
criterion = nn.CrossEntropyLoss()

#optimizer
optimizer = torch.optim.Adam(my_PSPnet_model.parameters(), lr=1e-4)

#metrics
dice_fn = torchmetrics.Dice(num_classes=21, average="macro").to(device)
iou_fn = torchmetrics.JaccardIndex(num_classes=21, task="multiclass", average="macro").to(device)

#meter
acc_meter = AverageMeter()
train_loss_meter = AverageMeter()
dice_meter = AverageMeter()
iou_meter = AverageMeter()

aux_weight = 0.4

Loss = []
Loss_ep = []
index_plt = 0

# trainning
n_eps = 0
if n_eps > 0 :
    for ep in range(1, 1+n_eps):
        acc_meter.reset()
        train_loss_meter.reset()
        dice_meter.reset()
        iou_meter.reset()
        my_PSPnet_model.train()
    
        for batch_id, (x, y) in enumerate(tqdm(trainloader), start=1):
            if batch_id > 41 :
              continue
            optimizer.zero_grad()
            n = x.shape[0]
    
    
            x = x.to(device).float()
            y = y.to(device).long()
            y_hat_mask, main_loss, aux_loss = my_PSPnet_model(x, y)
    
    
            loss = main_loss + aux_weight * aux_loss
            # y_hat = my_PSPnet_model(x) #(B, C, H, W)
            # loss = criterion(y_hat, y) #(B, C, H, W) >< (B, H, W)
            loss.backward()
            optimizer.step()
    
            with torch.no_grad():
                # y_hat_mask = y_hat.argmax(dim=1).squeeze() # (B, C, H, W) -> (B, 1, H, W) -> (B, H, W)
                dice_score = dice_fn(y_hat_mask, y.long())
                iou_score = iou_fn(y_hat_mask, y.long())
                accuracy = accuracy_function(y_hat_mask, y.long())
    
                train_loss_meter.update(loss.item(), n)
                iou_meter.update(iou_score.item(), n)
                dice_meter.update(dice_score.item(), n)
                acc_meter.update(accuracy.item(), n)
    
                Loss.append(train_loss_meter.avg)
        print("EP {}, train loss = {}, accuracy = {}, IoU = {}, dice = {}".format(
            ep, train_loss_meter.avg, acc_meter.avg, iou_meter.avg, dice_meter.avg
        ))
        Loss_ep.append(train_loss_meter.avg)
        if ep > 25:
            torch.save(my_PSPnet_model.state_dict(), weights_path + "\modelPSPNet_ep.pth")
    plt.plot([i for i, _ in enumerate(Loss_ep)],Loss_ep,'o')
    plt.show()
