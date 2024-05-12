import json, os
import numpy as np
import cv2 
import matplotlib as plt
from PIL import Image

root_path        = r"."
masks_path       = r".\MASK"
masks_review_path = r".\MASK_REVIEW"
json_files_path  = r".\JSON_LABELME"
json_files_list  = os.listdir(json_files_path)

height = 1024
width = 2048

Label_Class = {
    "void" : 0,
    "DUONG_DI" : 1,
    "LAN_HIEN_TAI" : 2,
    "LAN_TRAI_0" : 3,
    "LAN_PHAI_0" : 4,
    "DUONG_CHUYEN_LAN" : 5
}

for json_filename in json_files_list:
    # for each json file in the folder, open it
    with open(json_files_path + "\\" + json_filename) as json_file:
        # load data in json file
        json_data = json.load(json_file)
        # the data contains multi label, just focus to "shapes"
        labels_list = []
        points_list = []  # aka contour : đường viền :)
        for classes_of_object in json_data["shapes"]:
            # in the dict we have got - "shapes" also contain many labels
            # we just need "label" and "points" 
            # to make a mask
            my_label = classes_of_object["label"]
            my_points = classes_of_object["points"]
            # convert my_points from [[float, float], [float, float], ...] 
            # to [ (int, int), (int, int), ...] 
            # then to
            # [ [int int]
            #   [int int]
            #   ...     
            #   [int int] ]
            my_new_points = []
            # print(type(my_new_points))
            # print(my_new_points)
            for a_point in my_points:
                x = int(a_point[0])
                y = int(a_point[1])
                my_new_points.append((x,y))
            my_new_points = np.array(my_new_points)
            # print(type(my_new_points))
            # print(my_new_points)

            
            labels_list.append(my_label)
            points_list.append(my_new_points)
            # print(my_label)
            # print(my_new_points)
            # print(my_points)
        # make a mask image with zeros
        mask_img = np.zeros((height, width), dtype=np.uint8)
        mask_review_img = np.zeros([height, width, 3], dtype=np.uint8)
        # mask the image
        # print(labels_list)
        for ith_label in range(len(labels_list)):
            # print(ith_label, labels_list[ith_label], Label_Class.get(labels_list[ith_label]))
            for i in range(width):
                for j in range(height):
                    if cv2.pointPolygonTest(points_list[ith_label], (i, j), False) > 0:
                        if Label_Class.get(labels_list[ith_label]) == 1:
                            mask_review_img[j,i, 0] += 50
                            mask_review_img[j,i, 1] += 50
                            mask_review_img[j,i, 2] += 50
                        if Label_Class.get(labels_list[ith_label]) == 2:
                            mask_review_img[j,i,0] += 100
                        if Label_Class.get(labels_list[ith_label]) == 3:
                            mask_review_img[j,i,1] += 100
                        if Label_Class.get(labels_list[ith_label]) == 4:
                            mask_review_img[j,i,2] += 100
                        mask_img[j, i] = Label_Class.get(labels_list[ith_label])

        # save mask image
        cv2.imwrite( masks_path + "\\" + json_filename.replace("json", "png"), mask_img)
        print("Mask image saved at: ", masks_path + "\\" + json_filename.replace("json", "png"))
        cv2.imwrite( masks_review_path + "\\" + json_filename.replace("json", "png"), mask_review_img)
        print("Mask review image saved at: ", masks_review_path + "\\" + json_filename.replace("json", "png"))
        