### 20200519 gyuha park

import os
import cv2
import numpy as np
from skimage.measure import label, regionprops
import math
import glob
from pathlib import Path
import random

margin = 50
img_size = 512

COCO_dir = Path('C:/Users/IVCL/Desktop/val2017')
backgrounds = glob.glob(str(COCO_dir / '*'))
name_list = os.listdir('C:/Users/IVCL/Desktop/crane/image')

def ft(img, img_mask, img_bg):
    img_mask = cv2.cvtColor(img_mask,cv2.COLOR_GRAY2BGR)
    fimg = np.where(img_mask==[0,0,0],img_bg,img)

    return fimg

for i in range(0, 20000):
    name = name_list[i%len(name_list)]
    img = cv2.imread('./image/%s'%name)
    img_mask = cv2.imread('./mask/%s'%name, cv2.IMREAD_GRAYSCALE)
    img_mask2 = cv2.imread('./mask2/%s'%name, cv2.IMREAD_GRAYSCALE)
    img_mask_u = cv2.imread('./U_mask/%s'%name, cv2.IMREAD_GRAYSCALE)
    img_mask_v = cv2.imread('./V_mask/%s'%name, cv2.IMREAD_GRAYSCALE)
    
    # change background
    img_bg = cv2.imread(np.random.choice(backgrounds))
    img_bg = cv2.resize(img_bg, dsize=(len(img), len(img)), interpolation=cv2.INTER_CUBIC)
    img = ft(img, img_mask2, img_bg)
       
    # get bounding box
    mask = label(img_mask2)
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
        n_img_mask_u = img_mask_u[n_xl-margin:n_xr+margin,0:n_yr+margin]
        n_img_mask_v = img_mask_v[n_xl-margin:n_xr+margin,0:n_yr+margin]
    
    elif margin > n_xl:
        n_img = img[0:n_xr+margin,n_yl-margin:n_yr+margin, :]
        n_img_mask = img_mask[0:n_xr+margin,n_yl-margin:n_yr+margin]
        n_img_mask_u = img_mask_u[0:n_xr+margin,n_yl-margin:n_yr+margin]
        n_img_mask_v = img_mask_v[0:n_xr+margin,n_yl-margin:n_yr+margin]

    elif margin > n_yl and margin > n_xl:
        n_img = img[0:n_xr+margin,0:n_yr+margin, :]
        n_img_mask = img_mask[0:n_xr+margin,0:n_yr+margin]
        n_img_mask_u = img_mask_u[0:n_xr+margin,0:n_yr+margin]
        n_img_mask_v = img_mask_v[0:n_xr+margin,0:n_yr+margin]

    else:
        n_img = img[n_xl-margin:n_xr+margin,n_yl-margin:n_yr+margin, :]
        n_img_mask = img_mask[n_xl-margin:n_xr+margin,n_yl-margin:n_yr+margin]
        n_img_mask_u = img_mask_u[n_xl-margin:n_xr+margin,n_yl-margin:n_yr+margin]
        n_img_mask_v = img_mask_v[n_xl-margin:n_xr+margin,n_yl-margin:n_yr+margin]
    
    # resize image
    n_img = cv2.resize(n_img, dsize=(img_size, img_size), interpolation=cv2.INTER_AREA)
    n_img_mask = cv2.resize(n_img_mask, dsize=(img_size, img_size), interpolation=cv2.INTER_AREA)
    n_img_mask_u = cv2.resize(n_img_mask_u, dsize=(img_size, img_size), interpolation=cv2.INTER_AREA)
    n_img_mask_v = cv2.resize(n_img_mask_v, dsize=(img_size, img_size), interpolation=cv2.INTER_AREA)
    
    n_img_mask = np.where(n_img_mask>0,255,0)

    number = ''
    if i+1 < 10:
        number = '0000' + str(i+1)
    elif i+1 < 100:
        number = '000' + str(i+1)
    elif i+1 < 1000:
        number = '00' + str(i+1)
    elif i+1 < 10000:
        number = '0' + str(i+1)             
    else: number = str(i+1)

    # save image
    cv2.imwrite('./image_coco_512/crane_{}.png'.format(number),n_img)
    cv2.imwrite('./mask_coco_512/crane_{}.png'.format(number), n_img_mask)
    cv2.imwrite('./U_mask_coco_512/crane_{}.png'.format(number), n_img_mask_u)
    cv2.imwrite('./V_mask_coco_512/crane_{}.png'.format(number), n_img_mask_v)
    print('crane_{}.png'.format(number))

print('End')