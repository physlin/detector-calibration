from sys import version
import dask.array as da
from detectorcal.fit import gaussian_smooth
import numpy as np
from tifffile import TiffWriter
from pathlib import Path
import dask.array as da
from dask.distributed import Client
from time import time
# determine cupy will be imported and used
try:
    import cupy as cp
    USE_GPU = True
except ImportError:
    USE_GPU = False



def correct_image(
    ct_volume, 
    coefficients, 
    dark=None,
    flat=None,
    save_path=None,
    use_dask=False,
    sigma=None,
    smooth_radius=50,
    mode='nearest',
    cval=0.0, 
    truncate=4.0,
    gpu=False,
    verbose=False,
):
    '''
    Use coefficient array, dark correction, flat field correction, and
    (optionally) residual correction to correct rings in CT volume. 

    Parameters
    ----------
    ct_volume: ndarray like
        CT image volume to be corrected. (n, y, x)
    coefficients: ndarray like
        Linear coefficients for the detector. (y, x)
    dark: ndarray like
        Dark current image to be subtracted. (y, x)
    flat: ndarray like
        Mean of the flat-field images aquired with the CT sequence. (y, x)
    sigma: None or scalar
        Number of standard deviations
    save_path: None or str
        Default None. Optional path to which to save output.
    use_dask: bool
        Defaut False. Should dask be used to support computation? Dask manages
        memory to allow for computation on larger images. Only set this to
        True if the default option raises MemoryError.
    smooth_radius: int
        Default 50. Determines degree of smoothing  of flat-field (i.e., 
        sigma for Gaussian kernel). Please use the same value as that used 
        to fit the
        coefficients. 
    mode: str
        Default 'nearest'. Mode used for padding the edges of the flat-field 
        image duing Gaussian smoothing. 
    cval: scalar
        Default 0.0. Value with which to pad edges when above mode is equal 
        to 'constant'.
    truncate: scalar
        Default 4.0. Gaussian filter output will be truncated at at this many 
        standard deviations.
    gpu: bool
        Default False unless gpu option installed. Can NVIDIA GPUs be used 
        to accelerate this?
    verbose: bool
        Default False. Should descriptive print outs be written to console. 
        
    References
    ----------
    Croton, L.C., Ruben, G., Morgan, K.S., Paganin, D.M. and Kitchen, M.J., 
        2019. Ring artifact suppression in X-ray computed tomography using a 
        simple, pixel-wise response correction. Optics express, 27(10), 
        pp.14231-14245.
    '''
    if use_dask:
        corrected = dask_assisted_correction(ct_volume, coefficients, dark=dark,flat=flat, 
                                             sigma=sigma, smooth_radius=smooth_radius, 
                                             mode=mode, cval=cval, truncate=truncate, 
                                             gpu=gpu, verbose=verbose)
    else:
        corrected = dask_free_correction(ct_volume, coefficients, dark=dark,flat=flat, 
                                         sigma=sigma, gpu=gpu, verbose=verbose)
    save = save_path is not None
    if save:
        file_type = Path(save_path).suffix
        lazy_corr = da.from_array(corrected)
        if file_type == '.h5' or file_type == '.hdf5':
            lazy_corr.to_hdf5(save_path, '/data')
        if file_type == '.zar' or file_type == '.zarr':
            lazy_corr.to_zarr(save_path)
        if file_type == '.tif' or file_type == '.tiff':
            with TiffWriter(save_path) as tiff:
                tiff.save(corrected)
        if verbose:
            print(f'Saved corrected volume at: {save_path}')
    return corrected



# -------------------
# Dask Based Approach
# -------------------

def dask_assisted_correction(
    ct_volume, 
    coefficients, 
    dark=None,
    flat=None,
    sigma=None,
    smooth_radius=50,
    mode='nearest',
    cval=0.0, 
    truncate=4.0,
    gpu=False,
    verbose=False,
    ):
    '''
    Correction using dask. This is executed in detectorcal.correct_image 
    if the use_dask flag is set to True.Please only use this option if the CT 
    volume is too big to correct using numpy only as dask will add overhead 
    and increase compute time.

    Parameters
    ----------
    ct_volume: ndarray like
        CT image volume to be corrected. (n, y, x)
    coefficients: ndarray like
        Linear coefficients for the detector. (y, x)
    dark: ndarray like
        Dark current image to be subtracted. (y, x)
    flat: ndarray like
        Mean of the flat-field images aquired with the CT sequence. (y, x)
    sigma: None or scalar
        Number of standard deviations
    smooth_radius: int
        Determines degree of smoothing  of flat-field (i.e., sigma for 
        Gaussian kernel). Please use the same value as that used to fit the
        coefficients. 
    mode: str
        Mode used for padding the edges of the flat-field image duing Gaussian
        smoothing. 
    cval: scalar
        Value with which to pad edges when above mode is equal to 'constant'.
    truncate: scalar
        Gaussian filter output will be truncated at at this many standard deviations.
    gpu: bool
        Can NVIDIA GPUs be used to accelerate this?
    verbose: bool
        Should descriptive print outs be written to console. 
    '''
    t = time()
    client = Client(processes=False)
    if verbose:
        print(client.dashboard_link)
    use_flat = flat is not None
    chunks = coefficients.shape
    # dask conversions
    ct_volume = da.from_array(ct_volume)
    coefficients = da.from_array(coefficients, chunks=chunks)
    if dark is not None:
        dark = da.from_array(dark, chunks=chunks)
    if use_flat:
        # get the smoothed flat before converting to dask
        flat_ed = np.expand_dims(flat, 0)
        flat_smoothed = gaussian_smooth(flat_ed, smooth_radius, mode, cval, 
                                        truncate, gpu, verbose=True)
        flat = da.from_array(flat, chunks=chunks)
        flat_smoothed = flat_smoothed[0, :, :]
        flat_smoothed = da.from_array(flat_smoothed, chunks=chunks)
    # gpu cupy if necessary
    if gpu:
        ct_volume = ct_volume.map_blocks(cp.array)
        coefficients = coefficients.map_blocks(cp.array)
        if dark is not None:
            dark = dark.map_blocks(cp.array)
        if use_flat:
            flat = flat.map_blocks(cp.array)
            flat_smoothed = flat_smoothed.map_blocks(cp.array)
    # Dark removal
    if dark is not None:
        ct_volume = ct_volume - dark
    # initial correction
    corr = ct_volume * coefficients
    # flat field correction no residual correction
    if sigma is None and use_flat:
        corr = corr / flat_smoothed
    # flat field correction with residual correction
    elif sigma is not None and use_flat:
        flat_corr = flat * coefficients
        residuals = flat_smoothed - flat_corr
        # filter out residuals to ensure that only those persistant 
        # throughout the CT are adressed (otherwise --> extra rings)
        stds = da.std(residuals) * sigma
        if gpu:
            abs_residuals = residuals.map_blocks(cp.abs)
        else:
            abs_residuals = residuals.map_blocks(np.abs)
        residuals[abs_residuals < stds] = 0.
        flat = flat_smoothed - residuals
        corr = corr / flat
    if gpu:
        corr = corr.map_blocks(cupy_to_numpy)
    # compute
    corr = corr.compute()
    client.close()
    if verbose:
        print(f'Corrected volume in {time() - t} seconds')
    return corr


def cupy_to_numpy(array):
    return cp.asnumpy(array)


# ------------------
# Dask Free Approach
# ------------------

def dask_free_correction(
    ct_vol, 
    coeffs, 
    dark, 
    flat, 
    sigma=3., 
    gpu=False,
    verbose=False
    ):
    '''
    Numpy or cupy based correction. This is executed when detectorcal.correct_image
    is called with the use_dask set to False (as is default). 

    Parameters
    ----------
    ct_volume: ndarray like
        CT image volume to be corrected. (n, y, x)
    coefficients: ndarray like
        Linear coefficients for the detector. (y, x)
    dark: ndarray like
        Dark current image to be subtracted. (y, x)
    flat: ndarray like
        Mean of the flat-field images aquired with the CT sequence. (y, x)
    sigma: None or scalar
        Number of standard deviations
    smooth_radius: int
        Determines degree of smoothing  of flat-field (i.e., sigma for 
        Gaussian kernel). Please use the same value as that used to fit the
        coefficients. 
    mode: str
        Mode used for padding the edges of the flat-field image duing Gaussian
        smoothing. 
    cval: scalar
        Value with which to pad edges when above mode is equal to 'constant'.
    truncate: scalar
        Gaussian filter output will be truncated at at this many standard deviations.
    gpu: bool
        Can NVIDIA GPUs be used to accelerate this?
    verbose: bool
        Should descriptive print outs be written to console. 
    '''
    t = time()
    if dark is not None:
        ct_vol = ct_vol - dark
    if gpu:
        ct_vol = cp.array(ct_vol)
    corr = np.zeros(shape=coeffs.shape)
    if flat is not None:
        flat_smoothed = gaussian_filter(flat, 50, order=0, output=None, mode="nearest", cval=0.0, truncate=4.0)
        if gpu:
            flat = cp.array(flat)
            flat_smoothed = cp.array(flat_smoothed)
    corr = ct_vol * coeffs
    del ct_vol
    if sigma is None and flat is not None:
        corr = corr / flat_smoothed
    elif flat is not None:
        flat_corr = flat * coeffs
        residuals = flat_smoothed - flat_corr
        residuals[np.abs(residuals)<(np.std(residuals)*sigma)] = 0.
        flat = flat_smoothed - residuals
        corr = corr / flat
    if gpu:
        corr = cp.asnumpy(corr)
    if verbose:
        print(f'Corrected volume in {time() - t} seconds')
    return corr


if __name__ == '__main__':
    # imports 
    from skimage.io import imread
    from scipy.ndimage.filters import gaussian_filter
    from time import time
    from napari_bioformats import read_bioformats


    # data
    ct_vol = read_bioformats('/home/abigail/GitRepos/detector-calibration/untracked/phantom_320ms_30kV_23W_sod50_sid150_1_MMStack_Default.ome.tif')[0][0]
    ct_vol = np.array(ct_vol)[::5]
    coeffs = imread('/home/abigail/GitRepos/detector-calibration/untracked/full_coefficients_gpu.tif')
    dark = imread('/home/abigail/GitRepos/detector-calibration/untracked/darks_320ms_avg.tif')
    flat = imread('/home/abigail/GitRepos/detector-calibration/untracked/flats_start_320ms_30kV_23W_avg.tif')
    
    # Resid, Original
    t = time()
    save_path = '/home/abigail/GitRepos/detector-calibration/untracked/corrected_resid_orig.zarr'
    _ = correct_image(ct_vol, coeffs, dark, flat, save_path, sigma=None, gpu=False, verbose=True)
    print(f'Original correction with residual in {time() - t} s')

    # No resid, No GPU
    t = time()
    save_path = '/home/abigail/GitRepos/detector-calibration/untracked/corrected_no_resid_no_gpu.zarr'
    _ = correct_image(ct_vol, coeffs, dark, flat, save_path, use_dask=True, sigma=None, gpu=False, verbose=True)
    print(f'Correction with no residual correction (no GPU) in {time() - t} s')
    
    # Resid, No GPU
    t = time()
    save_path = '/home/abigail/GitRepos/detector-calibration/untracked/corrected_resid_no_gpu.zarr'
    _ = correct_image(ct_vol, coeffs, dark, flat, save_path, use_dask=True, sigma=3., gpu=False, verbose=True)
    print(f'Correction with residual correction (no GPU) in {time() - t} s')