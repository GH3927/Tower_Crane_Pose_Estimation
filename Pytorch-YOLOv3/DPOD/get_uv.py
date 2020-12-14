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
from scipy.spatial.transform import Rotation as R
from glob import glob
import csv

image_num = 4000

# 2048x2048
fx = 595.259627*4
px = 255.312872*4
fy = 551.317594*4
py = 223.823052*4

# Camera parameters
intrinsic_matrix = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])

camera_rigid_transformation = np.array([[ 0.9999970197677612, -8.742251367266363e-08, -0.00243386160582304, 0.006821606773883104],
        [-8.742275525719378e-08, -0.9999997615814209, -1.3877781190369556e-17, -0.018590735271573067],
        [-0.0024338611401617527, 2.1277490880322603e-10, -0.9999967813491821, 0.09084007143974304]])

# load dataset
ry = (R.from_euler('y', 90, degrees=True)).as_dcm()
rz = (R.from_euler('z', 90, degrees=True)).as_dcm()

pose_list = []
f = open("C:/Users/IVCL/Desktop/crane/quat_blender.csv", "r")
lines = csv.reader(f)
for line in lines:
    pose_list.append(line)
f.close()

tvec_list = []
f = open("C:/Users/IVCL/Desktop/crane/tvec_blender.csv", "r")
lines = csv.reader(f)
for line in lines:
    tvec_list.append(line)
f.close()

# Load point cloud
pt_cld = o3d.io.read_point_cloud("crane.ply")
pt_cld_data = np.asarray(pt_cld.points) * 0.063333
ones = np.ones((pt_cld_data.shape[0], 1))
homogenous_coordinate = np.append(pt_cld_data[:, :3], ones, axis=1)
print("num_points:",len(pt_cld_data))

# Utils
def get_rot_tra(rot_adr, tra_adr):
    """
    Helper function to the read the rotation and translation file
        Args:
                rot_adr (str): path to the file containing rotation of an object
        tra_adr (str): path to the file containing translation of an object
        Returns:
                rigid transformation (np array): rotation and translation matrix combined
    """

    rot_matrix = np.loadtxt(rot_adr, skiprows=1)
    trans_matrix = np.loadtxt(tra_adr, skiprows=1)
    trans_matrix = np.reshape(trans_matrix, (3, 1))
    rigid_transformation = np.append(rot_matrix, trans_matrix, axis=1)

    return rigid_transformation


def fill_holes(idmask, umask, vmask):
    """
    Helper function to fill the holes in id , u and vmasks
        Args:
                idmask (np.array): id mask whose holes you want to fill
        umask (np.array): u mask whose holes you want to fill
        vmask (np.array): v mask whose holes you want to fill
        Returns:
                filled_id_mask (np array): id mask with holes filled
        filled_u_mask (np array): u mask with holes filled
        filled_id_mask (np array): v mask with holes filled
    """
    idmask = np.array(idmask, dtype='float32')
    umask = np.array(umask, dtype='float32')
    vmask = np.array(vmask, dtype='float32')
    thr, im_th = cv2.threshold(idmask, 0, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    res = cv2.morphologyEx(im_th, cv2.MORPH_OPEN, kernel)
    im_th = cv2.bitwise_not(im_th)
    des = cv2.bitwise_not(res)
    mask = np.array(des-im_th, dtype='uint8')
    filled_id_mask = cv2.inpaint(idmask, mask, 5, cv2.INPAINT_TELEA)
    filled_u_mask = cv2.inpaint(umask, mask, 5, cv2.INPAINT_TELEA)
    filled_v_mask = cv2.inpaint(vmask, mask, 5, cv2.INPAINT_TELEA)
    
    return filled_id_mask, filled_u_mask, filled_v_mask

# Get Ground Truth
for i in range(0,2000):
    num = i%2000
    
    # Get transform matrix
    
    ptc_rot_matrix = R.from_quat([pose_list[num][1], pose_list[num][2], pose_list[num][3], pose_list[num][0]]).as_dcm()
    ptc_rot_matrix = np.append(ptc_rot_matrix, [[0, 0, 0]], axis=0)
    
    tvec = np.array([tvec_list[num]]).astype('float64')
    ptc_trans_matrix = [[tvec_list[num][0]], [tvec_list[num][1]], [tvec_list[num][2]], [1.0]]
    ptc_rigid_transformation = np.append(ptc_rot_matrix, ptc_trans_matrix, axis=1).astype('float64')
    
    #x_size = int(image.shape[1])
    #y_size = int(image.shape[0])
    x_size = 2048
    y_size = 2048
    ID_mask = np.zeros((y_size, x_size))
    U_mask = np.zeros((y_size, x_size))
    V_mask = np.zeros((y_size, x_size))
    
    # Project 2D
    homogenous_2D = intrinsic_matrix @ (
            camera_rigid_transformation @ (ptc_rigid_transformation @ homogenous_coordinate.T))
    coord_2D = homogenous_2D[:2, :] / homogenous_2D[2, :]
    coord_2D = ((np.floor(coord_2D)).T).astype(int)
    x_2d = np.clip(coord_2D[:, 0], 0, x_size-1)
    y_2d = np.clip(coord_2D[:, 1], 0, y_size-1)
    ID_mask[y_2d, x_2d] = 255
    
    centre = np.mean(pt_cld_data, axis=0)
    length = np.sqrt((centre[0]-pt_cld_data[:, 0])**2 + (centre[1] -
                                                         pt_cld_data[:, 1])**2 + (centre[2]-pt_cld_data[:, 2])**2)
    unit_vector = [(pt_cld_data[:, 0]-centre[0])/length, (pt_cld_data[:,
                                                                      1]-centre[1])/length, (pt_cld_data[:, 2]-centre[2])/length]
    
    xm = unit_vector[0].sum()/(len(pt_cld_data))
    ym = unit_vector[1].sum()/(len(pt_cld_data))
    zm = unit_vector[2].sum()/(len(pt_cld_data))
    
    cm = (xm+ym+zm)/3
    
    xmc = cm/xm
    ymc = cm/ym
    zmc = cm/zm
    
    U = 0.5 + (np.arctan2(unit_vector[2]*zmc, unit_vector[0]*xmc)/(2*np.pi))
    V = 0.5 - (np.arcsin(unit_vector[1]*ymc)/np.pi)
    U_mask[y_2d, x_2d] = U
    V_mask[y_2d, x_2d] = V
    
    ID_mask, U_mask, V_mask = fill_holes(ID_mask, U_mask, V_mask)
    
    ID_mask, U_mask, V_mask = np.flip(ID_mask, axis=1), np.flip(U_mask, axis=1), np.flip(V_mask, axis=1)
    
    number = ''
    if i+1 < 10:
        number = '000' + str(i+1)
    elif i+1 < 100:
        number = '00' + str(i+1)
    elif i+1 < 1000:
        number = '0' + str(i+1)          
    else: number = str(i+1)
    
    # Save images
    cv2.imwrite("C:/Users/IVCL/Desktop/crane/mask/crane_{}.png".format(number), ID_mask)
    cv2.imwrite("C:/Users/IVCL/Desktop/crane/U_mask/crane_{}.png".format(number), U_mask*255)
    cv2.imwrite("C:/Users/IVCL/Desktop/crane/V_mask/crane_{}.png".format(number), V_mask*255)
    
    print("crane_{}".format(number))