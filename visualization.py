#!/home/spl/ml/tf2.0/bin/python

import SimpleITK as sitk 
import numpy as np 
from matplotlib import pyplot as plt 
from ipywidgets import interact, fixed
import h5py

file = r"/media/spl/D/Kaggle competition/fMRI_test/10003.mat"

f = h5py.File(file, "r")
mri = f["SM_feature"].value

def interact_display(MRI):

    img_shape = MRI.shape
    kwds = {"img": fixed(MRI), "img_k": (0, img_shape[0] - 1), "img_z": (0, img_shape[1] - 1)}
    
    def display(**kwds):
    
        fig, ax = plt.subplots(1,1)
        ax.imshow(kwds["img"][kwds["img_k"], kwds["img_z"], :, :], cmap = plt.cm.Greys_r)
       
        ax.axis("off")

    interact(display, **kwds)