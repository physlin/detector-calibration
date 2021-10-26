#! /usr/bin/env python

import sys, os, glob
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage
import h5py
import warnings
from skimage import io
from PIL import Image

cutoff = 300.
smooth_radius = 50
coeff1 = np.zeros(shape=(1032,1032))
dirname = "/projects/yd88/Linda/Delta_X-ray_Lab/delta_2021-02-11_Croton_Rings/beam_sweep_320ms_30kV_23Wmax_sod50_sid150_1/"
volname = "beam_sweep_320ms_30kV_23Wmax_sod50_sid150_1_MMStack_Default.ome.tif"
darkname = "../darks_320ms_avg.tif"

# Read in beam sweep volume and create smoothed volume

print("Reading in volume and creating smoothed volume...")
print(dirname + volname)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)

    volume = io.imread(dirname + volname)
    dark_avg = io.imread(dirname + darkname)
    smvolume = np.zeros(shape=volume.shape)

volume = volume - dark_avg

print("Smoothing volume...")

for i in range(volume.shape[0]):
    smvolume[i,:,:] = scipy.ndimage.filters.gaussian_filter(volume[i,:,:], smooth_radius, order=0, output=None, mode="nearest", cval=0.0, truncate=4.0)
    
    outim = Image.fromarray(smvolume[i,:,:])
    outim.save("TIFs/sm_volume_sm5_fixed_"+str(i)+".tif")

# Initialise fit array
fit = np.zeros(shape=(1,volume.shape[1],volume.shape[2]))

print("Fitting volume...")
# Fit volume
for j in range(volume.shape[1]):
    print("Fitting row",j)
    for i in range(volume.shape[2]):
        points = np.where(smvolume[:,j,i]>cutoff)[0]
        fit[:,j,i] = np.linalg.lstsq(volume[points,j,i].reshape(-1,1), 
                                    smvolume[points,j,i], rcond=None)[0][0]

# Save coefficients of fits to images

print("Saving fit coefficient...")
outim = Image.fromarray(fit[0,:,:].astype(np.float32))
outim.save(dirname+"coeff1_deg1_sm"+str(smooth_radius)+"_fixed.tif")

print("Done!")

# end
