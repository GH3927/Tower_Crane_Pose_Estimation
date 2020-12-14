import os
import re
import cv2
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from helper import save_obj, load_obj
import open3d as o3d


pt_cld = o3d.io.read_point_cloud("crane.ply")
pt_cld_data = np.asarray(pt_cld.points) * 0.063333

centre = np.mean(pt_cld_data, axis=0)
length = np.sqrt((centre[0]-pt_cld_data[:, 0])**2 + (centre[1] -
                                                     pt_cld_data[:, 1])**2 + (centre[2]-pt_cld_data[:, 2])**2)
unit_vector = [(pt_cld_data[:, 0]-centre[0])/length, (pt_cld_data[:,
                                                                  1]-centre[1])/length, (pt_cld_data[:, 2]-centre[2])/length]
u_coord = 0.5 + (np.arctan2(unit_vector[2], unit_vector[0])/(2*np.pi))
v_coord = 0.5 - (np.arcsin(unit_vector[1])/np.pi)
u_coord = (u_coord * 255).astype(int)
v_coord = (v_coord * 255).astype(int)
# save the mapping as a pickle file
dct = {}
for u, v, xyz in zip(u_coord, v_coord, pt_cld_data):
    key = (u, v)
    if key not in dct:
        dct[key] = xyz
save_obj(dct, "./UV-XYZ_mapping")