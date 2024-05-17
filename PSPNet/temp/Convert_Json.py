import json, os
import numpy as np
import cv2 
import matplotlib as plt
from PIL import Image
import shutil

img_source       = "/mnt/sda1/WINDOWS_DOWNLOAD/Compressed/leftImg8bit_demoVideo_S"
root_path        = "/mnt/sda1/DOC/23-24_HK02/MLAI/CitiScape_Dataset_2"
masks_path       = "/mnt/sda1/DOC/23-24_HK02/MLAI/CitiScape_Dataset_2/MASK_F"
masks_review_path = "/mnt/sda1/DOC/23-24_HK02/MLAI/CitiScape_Dataset_2/MASK_REVIEW_F"
json_files_path  = "/mnt/sda1/DOC/23-24_HK02/MLAI/CitiScape_Dataset_2/JSON_LABELME"
json_files_list  = os.listdir(json_files_path)


# print list files
print("Exporting list_img!")
file = open(root_path+r"/IMG_LIST.txt", "w")

start_index = 0
end_index = len(json_files_list)

height = 1024
width = 2048

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



for ith_json_filename in range(start_index, end_index):
    json_filename = json_files_list[ith_json_filename]
    print("\nConverting {} to mask!".format(json_filename))
    with open(json_files_path + "/" + json_filename) as json_file:

        png_file_name = json_filename.replace(".json", ".png")
        png_file_path = img_source + "/" + png_file_name
        if os.path.exists(png_file_path):
            print("Converting ", png_file_name)
            file.write(png_file_name)
            file.write("\n")
            shutil.copy(png_file_path, root_path+"/IMG_F")
        else:
            continue
        json_data = json.load(json_file)
        labels_list = []
        points_list = [] 
        for classes_of_object in json_data["shapes"]:
            my_label = classes_of_object["label"]
            my_points = classes_of_object["points"]
            my_new_points = []
            for a_point in my_points:
                x = int(a_point[0])
                y = int(a_point[1])
                my_new_points.append((x,y))
            my_new_points = np.array(my_new_points)
            labels_list.append(my_label)
            points_list.append(my_new_points)
        mask_img = np.zeros((height, width), dtype=np.uint8)
        for ith_label in range(len(labels_list)):
            for i in range(width):
                for j in range(height):
                    if mask_img[j, i] != 0 and labels_list[ith_label] == "DUONG_DI":
                        continue
                    if cv2.pointPolygonTest(points_list[ith_label], (i, j), False) > 0:
                        ID_LABEL_CLASS = CLASSES.get(labels_list[ith_label])
                        if ID_LABEL_CLASS is None:
                            continue
                        mask_img[j, i] = CLASSES.get(labels_list[ith_label])
        cv2.imwrite( masks_path + "/" + json_filename.replace("json", "png"), mask_img)
        print("Mask image saved at: ", masks_path + "/" + json_filename.replace("json", "png"))
print("Converted to images!")