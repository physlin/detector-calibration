#! /usr/bin/env python

import sys, os, glob
import numpy as np
import numpy.polynomial.polynomial as poly
import scipy
import scipy.ndimage
import h5py
import warnings
from skimage import io
from PIL import Image

path = "/projects/yd88/Linda/Delta_X-ray_Lab/delta_2021-02-11_Croton_Rings/phantom_320ms_30kV_23W_sod50_sid150_1/"
volume = "phantom_320ms_30kV_23W_sod50_sid150_1_MMStack_Default.ome.tif"
dark = "../darks_320ms_avg.tif"
flat = "../flats_start_320ms_30kV_23W_avg.tif"
coeff_im = "coeff1_deg1_sm50_fixed.tif"
outname = "corr_volume_sm50_fixed"
smooth_radius = 50
sigma = None       # None for no residuals

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)

    ctvolume = io.imread(path+volume)
    dark_avg = io.imread(path+dark)

    ctvolume = ctvolume - dark_avg

    print("Loading coefficient...")
    coeff1 = io.imread(path+coeff_im).astype(np.float32)

    corr = np.zeros(shape=coeff1.shape)
    flat_corr = np.zeros(shape=coeff1.shape)

    print("Creating smoothed flat...")
    mean_flat = io.imread(path+flat).astype(np.float32)
    mean_flat = mean_flat

mean_flat_smoothed = scipy.ndimage.filters.gaussian_filter(mean_flat, smooth_radius, order=0, output=None, mode="nearest", cval=0.0, truncate=4.0)

if sigma == None:

    # Intensity correction

    print("Creating corrected images...")
    corr = ctvolume * coeff1
    del ctvolume
    corr = corr / mean_flat_smoothed

    with h5py.File("corr_volume_sm"+str(smooth_radius)+"_fixed_noresid.h5", "w") as hf:
        hf.create_dataset("corr_volume_sm"+str(smooth_radius)+"_fixed_noresid", data=corr)

else:

    print("Creating residual flat correction...")
    flat_corr = mean_flat * coeff1
    residuals = mean_flat_smoothed - flat_corr

    outim = Image.fromarray(residuals)
    outim.save("residuals.tif")

    residuals[np.abs(residuals)<(np.std(residuals)*sigma)] = 0.
    flat = mean_flat_smoothed - residuals

    outim = Image.fromarray(residuals)
    outim.save("residuals_"+str(sigma)+"sig.tif")

    # Intensity correction

    print("Creating corrected images...")
    corr = ctvolume * coeff1
    del ctvolume
    corr = corr / flat

    with h5py.File(outname+"_"+str(sigma)+"sig.h5", "w") as hf:
        hf.create_dataset(outname+"_"+str(sigma)+"sig", data=corr)

# End
