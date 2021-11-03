from detectorcal.fit import fit_response, sequential_fit, sequential_gauss
import numpy as np
from pathlib import Path
import pytest
import skimage.io as io
import os
from shutil import rmtree

CURRENT_PATH = Path(__file__).parent.resolve()
SRC_PATH = CURRENT_PATH.parents[1]
BS_DATA = str(SRC_PATH / 'data/detectorcaleg_stp4_200-500_500-800_beamsweep.tif')
DARK_DATA = str(SRC_PATH / 'data/detectorcaleg_stp4_200-500_500-800_dark.tif')
COEFD_DATA = str(SRC_PATH / 'data/detectorcaleg_stp4_200-500_500-800_coeff_dkrm.tif')
SAVE_DIR = SRC_PATH / 'data'

bs = io.imread(BS_DATA)
dk = io.imread(DARK_DATA)
coeffd = io.imread(COEFD_DATA)


def test_fit_rmdk(bs=bs, dk=dk, coeff=coeffd):
    out = fit_response(bs, dark=dk, verbose=True)
    b = out == coeffd
    assert b.min() == True


def test_save_methods(save_dir=SAVE_DIR):
    bs = np.random.random((10, 10, 10))
    # tiffile
    save_tiff = str(save_dir / 'temp.tif')
    _ = fit_response(bs, save_path=save_tiff)
    os.remove(save_tiff)
    # dask hdf5
    save_h5 = str(save_dir / 'temp.h5')
    _ = fit_response(bs, save_path=save_h5)
    os.remove(save_h5)
    # dask zarr
    save_zarr = str(save_dir / 'temp.zarr')
    _ = fit_response(bs, save_path=save_zarr)
    rmtree(save_zarr)


def test_older_methods():
    bs = np.random.random((10, 10, 10))
    smoothed = sequential_gauss(bs)
    assert bs.shape == smoothed.shape
    fit = sequential_fit(bs, smoothed, 0.5)