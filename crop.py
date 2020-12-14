### 20200519 gyuha park

import os
import cv2
import numpy as np
from skimage.measure import label, regionprops
import math
import glob
from pathlib import Path
import random

margin = 3
img_size = 512

directory = os.listdir('C:/Users/IVCL/Desktop/crane/real2_image')
os.chdir('C:/Users/IVCL/Desktop/crane/real2_image')
for file in directory:
    # load image file 
    img = cv2.imread(file) 
    img_mask = cv2.imread('../real2_mask/%s'%file, cv2.IMREAD_GRAYSCALE)
                
    # get bounding box
    mask = label(img_mask)
    props = regionprops(mask)
    [xl, yl, xr, yr] = [int(b) for b in props[0].bbox]
    [c_x, c_y] = [math.ceil((xl+xr)/2), math.ceil((yl+yr)/2)]
    [d_x, d_y] = [xr - xl, yr - yl]
    
    if d_x > d_y:
        [n_xl, n_yl] = [math.ceil(c_x - d_x/2), math.ceil(c_y - d_x/2)]
        [n_xr, n_yr] = [math.ceil(c_x + d_x/2), math.ceil(c_y + d_x/2)]
    
    else:
        [n_xl, n_yl] = [math.ceil(c_x - d_y/2), math.ceil(c_y - d_y/2)]
        [n_xr, n_yr] = [math.ceil(c_x + d_y/2), math.ceil(c_y + d_y/2)]
    
    # crop image
    if margin > n_yl:
        n_img = img[n_xl-margin:n_xr+margin,0:n_yr+margin, :]
        n_img_mask = img_mask[n_xl-margin:n_xr+margin,0:n_yr+margin]
    
    elif margin > n_xl:
        n_img = img[0:n_xr+margin,n_yl-margin:n_yr+margin, :]
        n_img_mask = img_mask[0:n_xr+margin,n_yl-margin:n_yr+margin]

    elif margin > n_yl and margin > n_xl:
        n_img = img[0:n_xr+margin,0:n_yr+margin, :]
        n_img_mask = img_mask[0:n_xr+margin,0:n_yr+margin]

    else:
        n_img = img[n_xl-margin:n_xr+margin,n_yl-margin:n_yr+margin, :]
        n_img_mask = img_mask[n_xl-margin:n_xr+margin,n_yl-margin:n_yr+margin]
    
    # resize image
    n_img = cv2.resize(n_img, dsize=(img_size, img_size), interpolation=cv2.INTER_AREA)
    n_img_mask = cv2.resize(n_img_mask, dsize=(img_size, img_size), interpolation=cv2.INTER_AREA)
    
    n_img_mask = np.where(n_img_mask>0,255,0)

    # save image
    cv2.imwrite('../real2_image_crop/%s'%file, n_img)
    cv2.imwrite('../real2_mask_crop/%s'%file, n_img_mask)
    print(file[:-4])