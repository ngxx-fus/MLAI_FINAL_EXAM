import json, os
import numpy as np
import cv2 
import matplotlib as plt
import shutil
from PIL import Image

root_path        = r"/home/ngxxfus/Downloads/img/leftImg8bit/val/frankfurt"
# dest_path        = r"/mnt/sda1/DOC/23-24_HK02/MLAI/CitiScape_Dataset_2/IMG"
# save_txt_path    = r"/mnt/sda1/DOC/23-24_HK02/MLAI/CitiScape_Dataset_2/trainval.txt"
folders  = os.listdir(root_path)

for fname in os.listdir(root_path):
    os.rename(root_path + '/' + fname, root_path + '/' + fname.replace("_leftImg8bit", ""))

