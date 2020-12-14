import os
import numpy as np
from PIL import Image
import random
import cv2

resize_scale = 900
crop_size = 512

img_fn = "./real_image"
mask_fn = "./real_mask"
img_list = os.listdir("./real_image")

for image in img_list:
    img = cv2.imread(img_fn + "/" + image)
    mask = cv2.imread(mask_fn + "/" + image, cv2.IMREAD_GRAYSCALE)
    
    seed = random.randint(1, 1000)
    random.seed(seed)
                      
    # preprocessing
    img = cv2.resize(img, (resize_scale, resize_scale), cv2.INTER_CUBIC)
    mask = cv2.resize(mask, (resize_scale, resize_scale), cv2.INTER_CUBIC)
        
    x = random.randint(0, resize_scale - crop_size + 1)
    y = random.randint(0, resize_scale - crop_size + 1)
    img = img[x:x + crop_size, y:y + crop_size]
    mask = mask[x:x + crop_size, y:y + crop_size]
    
    mask = np.array(mask)
    mask = np.where(mask>0,255,0)
    
    cv2.imwrite("./real_image_crop/%s"%image, img)
    cv2.imwrite("./real_mask_crop/%s"%image, mask)