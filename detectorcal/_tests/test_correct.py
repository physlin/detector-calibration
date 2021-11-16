from detectorcal.correct import correct_image
from pathlib import Path
import pytest
import skimage.io as io
import numpy as np
import os
from shutil import rmtree


CURRENT_PATH = Path(__file__).parent.resolve()
SRC_PATH = CURRENT_PATH.parents[1]
CT_DATA = str(SRC_PATH / 'data/detectorcaleg_stp4_200-500_500-800_CT.tif')
FLAT_DATA = str(SRC_PATH / 'data/detectorcaleg_stp4_200-500_500-800_flat.tif')
DARK_DATA = str(SRC_PATH / 'data/detectorcaleg_stp4_200-500_500-800_dark.tif')
CORR_DATA_NR = str(SRC_PATH / 'data/detectorcaleg_stp4_200-500_500-800_corrected_NR.tif')
CORR_DATA_R = str(SRC_PATH / 'data/detectorcaleg_stp4_200-500_500-800_corrected_R.tif')
COEF_DATA = str(SRC_PATH / 'data/detectorcaleg_stp4_200-500_500-800_coeff.tif')
COEFD_DATA = str(SRC_PATH / 'data/detectorcaleg_stp4_200-500_500-800_coeff_dkrm.tif')

ct = io.imread(CT_DATA)
flat = io.imread(FLAT_DATA)
dark = io.imread(DARK_DATA)
corr_nr = io.imread(CORR_DATA_NR)
corr_r = io.imread(CORR_DATA_R)
coef = io.imread(COEF_DATA)

SAVE_DIR = SRC_PATH / 'data'

def test_correct_no_resid_corr_no_dask():
    result = correct_image(ct, coef, dark=dark, flat=flat, verbose=True)
    b = result == corr_nr
    assert b.min() == True


def test_correct_resid_corr_no_dask():
    result = correct_image(ct, coef, dark=dark, flat=flat, sigma=3,  verbose=True)
    b = result == corr_r
    assert b.min() == True


def test_correct_no_resid_corr_dask():
    result = correct_image(ct, coef, dark=dark, flat=flat, use_dask=True,  verbose=True)
    b = result == corr_nr
    assert b.min() == True


def test_correct_resid_corr_dask():
    result = correct_image(ct, coef, dark=dark, flat=flat, use_dask=True, sigma=3,  verbose=True)
    b = result == corr_r
    assert b.min() == True


def test_save_methods(save_dir=SAVE_DIR):
    ct = np.random.random((10, 10, 10))
    coef = np.random.random((10, 10))
    # tiffile
    save_tiff = str(save_dir / 'temp.tif')
    _ = correct_image(ct, coef, save_path=save_tiff)
    os.remove(save_tiff)
    # dask hdf5
    save_h5 = str(save_dir / 'temp.h5')
    _ = correct_image(ct, coef, save_path=save_h5)
    os.remove(save_h5)
    # dask zarr
    save_zarr = str(save_dir / 'temp.zarr')
    _ = correct_image(ct, coef, save_path=save_zarr)
    rmtree(save_zarr)


def test_wrong_sigma():
    ct = np.random.random((10, 10, 10))
    coef = np.random.random((10, 10))
    _ = correct_image(ct, coef, 
    sigma='muahahaha bet you didnt expect a string', 
    verbose=True)