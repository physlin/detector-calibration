import pytest
from skimage import io
import numpy as np
from detectorcal import fit_response, correct_image
import os
from shutil import rmtree
from pathlib import Path


cp = pytest.importorskip('cupy')

CURRENT_PATH = Path(__file__).parent.resolve()
SRC_PATH = CURRENT_PATH.parents[1]
SAVE_DIR = SRC_PATH / 'data'


def test_fit_with_gpu():
    bs = np.random.random((10, 100, 100))
    dk = np.random.random((100, 100))
    out = fit_response(bs, dark=dk, gpu=True)
    

def test_fit_save_methods_GPU(save_dir=SAVE_DIR):
    ct = np.random.random((10, 10, 10))
    coef = np.random.random((10, 10))
    # tiffile
    save_tiff = str(save_dir / 'temp.tif')
    _ = correct_image(ct, coef, save_path=save_tiff, gpu=True)
    os.remove(save_tiff)
    # dask hdf5
    save_h5 = str(save_dir / 'temp.h5')
    _ = correct_image(ct, coef, save_path=save_h5, gpu=True)
    os.remove(save_h5)
    # dask zarr
    save_zarr = str(save_dir / 'temp.zarr')
    _ = correct_image(ct, coef, save_path=save_zarr, gpu=True)
    rmtree(save_zarr)


def test_correction_with_gpu():
    ct = np.random.random((10, 100, 100))
    ff = np.random.random((100, 100))
    dk = np.random.random((100, 100))
    cf = np.random.random((100, 100))
    out = correct_image(ct, cf, dark=dk, flat=ff, gpu=True)


def test_correct_save_methods_gpu(save_dir=SAVE_DIR):
    ct = np.random.random((10, 10, 10))
    coef = np.random.random((10, 10))
    # tiffile
    save_tiff = str(save_dir / 'temp.tif')
    _ = correct_image(ct, coef, save_path=save_tiff, gpu=True)
    os.remove(save_tiff)
    # dask hdf5
    save_h5 = str(save_dir / 'temp.h5')
    _ = correct_image(ct, coef, save_path=save_h5, gpu=True)
    os.remove(save_h5)
    # dask zarr
    save_zarr = str(save_dir / 'temp.zarr')
    _ = correct_image(ct, coef, save_path=save_zarr, gpu=True)
    rmtree(save_zarr)
