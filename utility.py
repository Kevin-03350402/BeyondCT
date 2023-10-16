import numpy as np
import nibabel as nib
from data_augmentation import *

def nii2np(img_path):
    filepath =  img_path
    img = nib.load(filepath).get_fdata()
    img = np.array(img,dtype = np.float32)
    img  = augmentor(img)

    img = np.expand_dims(img, axis=0)
    return img

def nii2npval(img_path):
    filepath = img_path
    img = nib.load(filepath).get_fdata()
    img = np.array(img,dtype = np.float32)
    img = np.expand_dims(img, axis=0)
    return img