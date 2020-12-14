import os
import cv2
import numpy as np
import imgaug as ia
import imageio
import imgaug.augmenters as iaa
from glob import glob
from pathlib import Path
from itertools import zip_longest

image_path = glob('C:/Users/IVCL/Desktop/crane/image_coco_512/*.png')
name_list = os.listdir('C:/Users/IVCL/Desktop/crane/image_coco_512')
image_list = [] 

for file in image_path:
    image = cv2.imread(file) #image file 불러옴
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_list.append(image)

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.

seq = iaa.SomeOf((1,5),
            [
                iaa.OneOf(
                [
                    iaa.GaussianBlur((0, 0.5)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(1, 3)), # blur image using local means with kernel sizes between 2 and 3
                    iaa.BilateralBlur(
                        d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250)),
                    iaa.AveragePooling([1, 3]),
                    iaa.MaxPooling([1, 3]),
                    iaa.MinPooling([1, 3]),
                    iaa.JpegCompression(compression=(5, 20)),
                ]),
                iaa.OneOf([
                    iaa.Sharpen(alpha=(0, 0.3), lightness=(0, 0.3)), # sharpen images
                    iaa.Emboss(alpha=(0, 0.3), strength=(0, 0.3)), # emboss images
                ]),
                iaa.OneOf(
                [
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                    iaa.AdditiveLaplaceNoise(scale=0.05*255, per_channel=True),
                    iaa.AdditivePoissonNoise((0, 20)),
                ]),
                iaa.OneOf(
                [
                    iaa.Cutout(nb_iterations=(1, 3), fill_mode="constant", cval=(0, 255), fill_per_channel=0.5),
                    iaa.Cutout(nb_iterations=(1, 3), fill_mode="gaussian", cval=(0, 255), fill_per_channel=0.5),                    
                ]),
                iaa.OneOf(
                [
                    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout((0.01, 0.1), size_percent=(0.02, 0.25)),
                    iaa.CoarseSaltAndPepper(
                        0.05, size_percent=(0.01, 0.03), per_channel=True),
                ]),
                iaa.OneOf(
                [
                    iaa.Invert(0.05, per_channel=0.5), # invert color channels
                    iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    iaa.AddToHueAndSaturation((-10, 10)), # change hue and saturation
                    iaa.AddToHue((-50, 50)),
                    iaa.Multiply((0.5, 1.5), per_channel=True),
                ]),
                iaa.OneOf([
                    iaa.SigmoidContrast(
                        gain=(3, 10), cutoff=(0.4, 0.6), per_channel=True),
                    iaa.LogContrast(gain=(0.6, 1.4), per_channel=True),
                    iaa.LinearContrast((0.5, 2.0)),
                    iaa.LogContrast(gain=(0.6, 1.4), per_channel=True),
                    iaa.AllChannelsCLAHE(clip_limit=(1, 10), per_channel=True),
                ]),
                iaa.OneOf(
                [
                    iaa.imgcorruptlike.Frost(severity=1),
                    iaa.imgcorruptlike.Snow(severity=1),
                    iaa.imgcorruptlike.Spatter(severity=1),
                    iaa.imgcorruptlike.Brightness(severity=1),
                    iaa.imgcorruptlike.Saturate(severity=1),                   
                ]),
                iaa.OneOf(
                [
                    iaa.pillike.Autocontrast((10, 20), per_channel=True),
                    iaa.pillike.Equalize(),
                    iaa.pillike.EnhanceColor(),
                    iaa.pillike.EnhanceContrast(),
                    iaa.pillike.FilterSmooth(),
                    iaa.pillike.FilterSharpen(),
                    iaa.pillike.FilterEdgeEnhance(),
                ]),
            ])
           
images_aug = seq(images=image_list)
#grid_image = ia.draw_grid(images_aug, cols=20)
#imageio.imwrite("C:/Users/IVCL/Desktop/example_segmaps.jpg", grid_image)

for image, name in zip(images_aug, name_list):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite('C:/Users/IVCL/Desktop/crane/image_coco_512_aug/{}'.format(name), image)