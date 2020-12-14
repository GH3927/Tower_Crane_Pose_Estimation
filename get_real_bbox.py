### 20200520 gyuha park

import os
import cv2
import numpy as np
import csv

bbox_list = []
csvfile=open("./bbox_real.csv", 'w', newline="")
csvwriter = csv.writer(csvfile)

image_dir = "C:/Users/IVCL/Desktop/crane/real_mask"
 
def ft(mask):
    
    x = len(mask[:,0])
    y = len(mask[0,:])
    
    xl = x-1
    yl = y-1
    xr = 0
    yr = 0
    for v in range(0, y):
        for u in range(0, x):
            if mask[u,v] == 255:
                if xr <= u:
                    xr = u 
                if yr <= v:
                    yr = v
                if xl >= u:
                    xl = u
                if yl >= v:
                    yl = v
                
    return [xl, yl, xr, yr]

directory = os.listdir(image_dir)
os.chdir(image_dir)

for file in directory:
    # load image file 
    mask = cv2.imread(file, cv2.IMREAD_GRAYSCALE) # load 2048x2048 mask
    bbox = ft(mask)
    bbox_list.append(bbox)

    print(file[:-4], bbox)

for row in bbox_list:
    csvwriter.writerow(row)
csvfile.close()

print('End')