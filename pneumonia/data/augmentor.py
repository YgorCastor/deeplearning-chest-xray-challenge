import os
from typing import List

import matplotlib

import pathlib
import imgaug as ia
import numpy as np
import cv2 as cv2

from glob import glob
from imgaug import augmenters as iaa

def augment_image(image): 
    sometimes = lambda chance,aug: iaa.Sometimes(chance, aug)
    seq = iaa.Sequential(
    [
        # sometimes(0.2, iaa.CoarseSaltAndPepper((0.03, 0.15), size_percent=(0.02, 0.05))),
        iaa.SomeOf((1, 3),
            [  
                #iaa.OneOf([
                #    iaa.PiecewiseAffine(scale=(0.01, 0.05)),
                #    iaa.Affine(
                #        shear=(-16, 16), # shear by -16 to +16 degrees
                #        order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                #        cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                #        mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                #    ),
                #]),
                iaa.Multiply((0.5, 1.5)),
                iaa.Add((-5, 5)),
                iaa.ContrastNormalization((0.5, 1.5))
            ],
            random_order=True
        )
    ], random_order=True
    )
    images_aug = seq.augment_image(image)
    return images_aug

def augment_folder(folder, verbose, times, seed):
    ia.seed(seed)
    print("Applying prior augmentation to folder '%s'" % folder)
    images: List[str] = glob("%s/*.jpeg" % folder)

    for i in range(times):
        for idx, file in enumerate(images):
            if verbose:
                print("Applying augmentation to file %s" % file)
            im = cv2.imread(file)
            augimg = augment_image(im)
            cv2.imwrite("%s/aug-%s-%d.jpeg" % (folder, idx, i), augimg)
        
    print("Prior Augmentation Finished...")
