import numpy as np
import os
import cv2

path = "./mask2/"
files = os.listdir(path)
for file in files: 
    img = cv2.imread("./mask2/%s"%file, cv2.IMREAD_GRAYSCALE)
    img = np.where(img==57,0,255)
    '''
    x = len(img[:,0])
    y = len(img[0,:])
    fimg = np.zeros([x,y], np.uint8) 
    for v in range(0,y):
        for u in range(0,x):
            if img[u,v] != 57:
                fimg[u,v] = 255
            else: fimg[u,v] = 0
    '''
    cv2.imwrite("./mask2/%s"%file, img)
    print(file)