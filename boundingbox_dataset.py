### 20200520 gyuha park

import os
import cv2
import numpy as np
from skimage.measure import label, regionprops
import math
import glob
from pathlib import Path
import random
import copy
import csv

name_list = os.listdir('C:/Users/IVCL/Desktop/crane/image')
bbox_list = []
csvfile=open("./bbox.csv", 'w', newline="")
csvwriter = csv.writer(csvfile)
COCO_dir = Path('C:/Users/IVCL/Desktop/val2017')
backgrounds = glob.glob(str(COCO_dir / '*'))

def ft(img, img_mask, img_mask2, img_mask_u, img_mask_v, img_bg, x_size, y_size):
    #set random seed
    seed = random.randint(1, 1000)
    random.seed(seed)
    
    x = len(img_bg[:,0])
    y = len(img_bg[0,:])
    
    # set bounding box points
    xr = random.randint(x_size-1,x-1)
    yr = random.randint(y_size-1,y-1)
    xl = xr - (x_size-1)
    yl = yr - (y_size-1)
    
    fimg = copy.deepcopy(img_bg)
    fmask = np.zeros([x,y])
    fmask_u = np.zeros([x,y])
    fmask_v = np.zeros([x,y])
    
    for v in range(yl, yr+1):
        for u in range(xl, xr+1):
            if img_mask2[u-xr+(x_size-1), v-yr+(y_size-1)] != 0:
                fimg[u,v] = img[u-xr+(x_size-1), v-yr+(y_size-1)]
            else: fimg[u,v] = img_bg[u,v]
            
            if img_mask[u-xr+(x_size-1), v-yr+(y_size-1)] != 0:
                fmask[u,v] = 255
                fmask_u[u,v] = img_mask_u[u-xr+(x_size-1),v-yr+(y_size-1)]
                fmask_v[u,v] = img_mask_v[u-xr+(x_size-1),v-yr+(y_size-1)]
            else: fmask[u,v] = 0
            
    return fimg, fmask, fmask_u, fmask_v, [xl, yl, xr, yr]

for i in range(1, 40001):
    img_size = 512
    name = name_list[i%len(name_list)]
    
    # set random seed
    seed = random.randint(1, 1000)
    random.seed(seed)    
    
    # load image file 
    img = cv2.imread('./image/%s'%name) 
    img_mask = cv2.imread('./mask/%s'%name, cv2.IMREAD_GRAYSCALE)
    img_mask2 = cv2.imread('./mask2/%s'%name, cv2.IMREAD_GRAYSCALE)
    img_mask_u = cv2.imread('./U_mask/%s'%name, cv2.IMREAD_GRAYSCALE)
    img_mask_v = cv2.imread('./V_mask/%s'%name, cv2.IMREAD_GRAYSCALE)
          
    # get bounding box
    mask = label(img_mask)
    props = regionprops(mask)
    [xl, yl, xr, yr] = [int(b) for b in props[0].bbox]
    [c_x, c_y] = [math.ceil((xl+xr)/2), math.ceil((yl+yr)/2)]
    [d_x, d_y] = [xr - xl, yr - yl]

    # crop image
    n_img = img[xl:xr,yl:yr, :]
    n_img_mask = img_mask[xl:xr,yl:yr]
    n_img_mask2 = img_mask2[xl:xr,yl:yr]
    n_img_mask_u = img_mask_u[xl:xr,yl:yr]
    n_img_mask_v = img_mask_v[xl:xr,yl:yr]
    
    # resize cropped image
    if d_x > d_y:
        x_size = random.randint(math.ceil(len(img)*0.3),math.ceil(len(img)*0.9))
        y_size = math.ceil(x_size * (d_y/d_x))
        
    if d_x < d_y:
        y_size = random.randint(math.ceil(len(img)*0.3),math.ceil(len(img)*0.9))
        x_size = math.ceil(y_size * (d_x/d_y))
    
    n_img = cv2.resize(n_img, dsize=(y_size, x_size), interpolation=cv2.INTER_AREA)
    n_img_mask = cv2.resize(n_img_mask, dsize=(y_size, x_size), interpolation=cv2.INTER_AREA)
    n_img_mask2 = cv2.resize(n_img_mask2, dsize=(y_size, x_size), interpolation=cv2.INTER_AREA)
    n_img_mask_u = cv2.resize(n_img_mask_u, dsize=(y_size, x_size), interpolation=cv2.INTER_AREA)
    n_img_mask_v = cv2.resize(n_img_mask_v, dsize=(y_size, x_size), interpolation=cv2.INTER_AREA)
    n_img_mask = np.where(n_img_mask>250,255,0)    
    n_img_mask2 = np.where(n_img_mask2>250,255,0)
    
    # change background and set crane
    img_bg = cv2.imread(np.random.choice(backgrounds))
    img_bg = cv2.resize(img_bg, dsize=(len(img), len(img)), interpolation=cv2.INTER_CUBIC)
    img_out, mask_out, mask_u_out, mask_v_out, bbox = ft(n_img, n_img_mask, n_img_mask2,
                                                         n_img_mask_u, n_img_mask_v, img_bg, x_size, y_size)
    
    # resize output image and bounding box
    img_out = cv2.resize(img_out, dsize=(img_size, img_size), interpolation=cv2.INTER_AREA)
    mask_out = cv2.resize(mask_out, dsize=(img_size, img_size), interpolation=cv2.INTER_AREA)
    mask_u_out = cv2.resize(mask_u_out, dsize=(img_size, img_size), interpolation=cv2.INTER_AREA)
    mask_v_out = cv2.resize(mask_v_out, dsize=(img_size, img_size), interpolation=cv2.INTER_AREA)
    mask_out = np.where(mask_out>0,255,0)
    n_bbox = [math.ceil(x/(len(img)/(img_size))) for x in bbox]
    bbox_list.append(n_bbox)
    
    # save image
    number = ''
    if i < 10:
        number = '0000' + str(i)
    elif i < 100:
        number = '000' + str(i)
    elif i < 1000:
        number = '00' + str(i)
    elif i < 10000:
        number = '0' + str(i)             
    else: number = str(i)

    # save image
    cv2.imwrite('./image_coco_512_bbox/crane_{}.png'.format(number),img_out)
    cv2.imwrite('./mask_coco_512_bbox/crane_{}.png'.format(number), mask_out)
    cv2.imwrite('./U_mask_coco_512_bbox/crane_{}.png'.format(number), mask_u_out)
    cv2.imwrite('./V_mask_coco_512_bbox/crane_{}.png'.format(number), mask_v_out)
    print('crane_{}.png'.format(number))

for row in bbox_list:
    csvwriter.writerow(row)
csvfile.close()

print('End')