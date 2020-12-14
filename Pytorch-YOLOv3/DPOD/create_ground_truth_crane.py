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

root_dir = "LineMOD_Dataset/"
classes = {'ape': 1, 'benchviseblue': 2, 'cam': 3, 'can': 4, 'cat': 5, 'driller': 6,
           'duck': 7, 'eggbox': 8, 'glue': 9, 'holepuncher': 10, 'iron': 11, 'lamp': 12, 'phone': 13}
fx = 560.0
px = 256.0
fy = 995.5555
py = 256.0
intrinsic_matrix = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])

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

image = cv2.imread("C:/Users/IVCL/Desktop/crane/image_coco_512_aug/crane_00001.png")
x_size = int(image.shape[1])
y_size = int(image.shape[0])
ID_mask = np.zeros((y_size, x_size))
U_mask = np.zeros((y_size, x_size))
V_mask = np.zeros((y_size, x_size))

trans_matrix = [[0],[0],[0]]
rot_matrix = [[1,0,0],[0,1,0],[0,0,1]]
rigid_transformation = np.append(rot_matrix, trans_matrix, axis=1)

# Read point Point Cloud Data
pt_cld = o3d.io.read_point_cloud("crane.ply")
pt_cld_data = np.asarray(pt_cld.points)

ones = np.ones((pt_cld_data.shape[0], 1))
homogenous_coordinate = np.append(pt_cld_data[:, :3], ones, axis=1)

# Perspective Projection to obtain 2D coordinates for masks
homogenous_2D = intrinsic_matrix @ (
    rigid_transformation @ homogenous_coordinate.T)
coord_2D = homogenous_2D[:2, :] / homogenous_2D[2, :]
coord_2D = ((np.floor(coord_2D)).T).astype(int)
x_2d = np.clip(coord_2D[:, 0], 0, x_size-1)
y_2d = np.clip(coord_2D[:, 1], 0, y_size-1)
ID_mask[y_2d, x_2d] = 255

# Generate Ground Truth UV Maps
centre = np.mean(pt_cld_data, axis=0)
length = np.sqrt((centre[0]-pt_cld_data[:, 0])**2 + (centre[1] -
                                                     pt_cld_data[:, 1])**2 + (centre[2]-pt_cld_data[:, 2])**2)
unit_vector = [(pt_cld_data[:, 0]-centre[0])/length, (pt_cld_data[:,
                                                                  1]-centre[1])/length, (pt_cld_data[:, 2]-centre[2])/length]
U = 0.5 + (np.arctan2(unit_vector[2], unit_vector[0])/(2*np.pi))
V = 0.5 - (np.arcsin(unit_vector[1])/np.pi)
U_mask[y_2d, x_2d] = U
V_mask[y_2d, x_2d] = V

# Saving ID, U and V masks after using the fill holes function
ID_mask, U_mask, V_mask = fill_holes(ID_mask, U_mask, V_mask)