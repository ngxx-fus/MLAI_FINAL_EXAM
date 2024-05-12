#------------- import custom libs ---------------------
from Unet_Custom_Dataset_Def import *
from Unet_Mesure_Def import *
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


#------------------ Declare my_ss_model --------------------
# my_ss_model
n_classes = 1
# my_ss_model = Unet_Model(1).to(device) # GPU trainning
my_ss_model = Unet_Model(n_classes)  # CPU trainning

#------------------ Declare device --------------------
#TODO: Must change manually
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print("device: ", device)

#------------------ Declare data loader (not dataset!) --------------------
batch_size = 8
n_workers = 0
print("num_workers: ", n_workers)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=n_workers)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False, num_workers=n_workers)


#------------- loss_function ---------------------
criterion = nn.BCEWithLogitsLoss()

#------------- optimizer_method ---------------------
optimizer = torch.optim.Adam(my_ss_model.parameters(), lr=1e-4)
#------------- metrics ---------------------
dice_function = torchmetrics.Dice(num_classes=n_classes+1, average="macro").to(device)
iou_function = torchmetrics.JaccardIndex(num_classes=n_classes+1, task="binary", average="macro").to(device)

#------------- meter ---------------------
accurancy_meter = AverageMeter()
train_loss_meter = AverageMeter()
dice_meter = AverageMeter()
iou_meter = AverageMeter()

#------------------ training the my_ss_model --------------------
n_eps = 30

for ep in range(1, 1+n_eps):
    print("ep: ", ep)
    accurancy_meter.reset()
    train_loss_meter.reset()
    dice_meter.reset()
    iou_meter.reset()
    my_ss_model.train()

    for batch_id, (x, y) in enumerate(tqdm(trainloader), start=1):
        print("batch_id: ", batch_id)
        n = x.shape[0]
        x = x.to(device).float()
        y = y.to(device).float()
        optimizer.zero_grad()
        print("Computing predict output!")
        y_hat = my_ss_model(x)
        y_hat = y_hat.squeeze() # -> logit (-vc, +vc)
        print("Computing lost!")
        loss = criterion(y_hat, y)
        print("Optimizing!")
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            y_hat_mask = y_hat.sigmoid().round().long() # -> mask (0, 1)
            dice_score = dice_function(y_hat_mask, y.long())
            iou_score = iou_function(y_hat_mask, y.long())
            accuracy = accuracy_function(y_hat_mask, y.long())

            train_loss_meter.update(loss.item(), n)
            iou_meter.update(iou_score.item(), n)
            dice_meter.update(dice_score.item(), n)
            accurancy_meter.update(accuracy.item(), n)

    # print("EP {}, train loss = {}, accuracy = {}, IoU = {}, dice = {}".format(
    #     ep, train_loss_meter.avg, accurancy_meter.avg, iou_meter.avg, dice_meter.avg
    # ))
    if ep >= 25:
        torch.save(my_ss_model.state_dict(), "/content/model_ep_{}.pth".format(ep))