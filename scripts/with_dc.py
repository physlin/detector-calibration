import sys, os, glob
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage
import warnings
from skimage import io
from PIL import Image


cutoff = 300.
smooth_radius = 50
coeff1 = np.zeros(shape=(1032,1032))
dirname = '/Users/amcg0011/GitRepos/ct-ring-correction/data'
volname = 'beam_sweep_320ms_30kV_23Wmax_sod50_sid150_1_MMStack_Default.ome.tif'
volume = io.imread(os.path.join(dirname, volname))
smvolume = np.zeros(shape=volume.shape)

print("Smoothing volume...")

for i in range(volume.shape[0]):
    smvolume[i,:,:] = scipy.ndimage.filters.gaussian_filter(volume[i,:,:], smooth_radius, order=0, output=None, mode="nearest", cval=0.0, truncate=4.0)


fit = np.zeros(shape=(1,volume.shape[1],volume.shape[2]))

print("Fitting volume...")
# Fit volume
for j in range(volume.shape[1]):
    print("Fitting row",j)
    for i in range(volume.shape[2]):
        points = np.where(smvolume[:,j,i]>cutoff)[0]
        fit[:,j,i] = np.linalg.lstsq(volume[points,j,i].reshape(-1,1), 
                                    smvolume[points,j,i], rcond=None)[0][0]


print("Saving fit coefficient...")
outim = Image.fromarray(fit[0,:,:].astype(np.float32))
outim.save(os.path.join(dirname, "coeff1_deg1_sm" + str(smooth_radius) + "_fixed_wdc.tif"))

print("Done!")