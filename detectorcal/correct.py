import dask.array as da
from .fit import gaussian_smooth
import numpy as np
import os
from pathlib import Path
from tifffile import TiffWriter


def correct_image(
    ct_volume, 
    coefficients, 
    dark=None,
    flat=None,
    save_dir=None,
    save_name=None,
    sigma=None,
    file_type='.tif',
    smooth_radius=50,
    mode='nearest',
    cval=0.0, 
    truncate=4.0,
    gpu=False,
    verbose=True,
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
    save: None or str
        Optional path to the output directory to which to save files.
        If None, data is not saved.
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
        
    References
    ----------
    Croton, L.C., Ruben, G., Morgan, K.S., Paganin, D.M. and Kitchen, M.J., 
        2019. Ring artifact suppression in X-ray computed tomography using a 
        simple, pixel-wise response correction. Optics express, 27(10), 
        pp.14231-14245.
    '''
    save = save_name is not None or save_dir is not None
    use_flat = flat is not None
    if save:
        if save_name is None:
            save_name = ''
        if save_dir is None:
            save_dir = os.getcwd()
    if dark is not None:
        ct_volume = ct_volume - dark
    if use_flat:
        flat_ed = np.expand_dims(flat, 0)
        flat_smoothed = gaussian_smooth(flat_ed, smooth_radius, mode, cval, 
                                        truncate, gpu, verbose)
        assert flat_smoothed.ndim == 3
        flat_smoothed = flat_smoothed[0, :, :]
    corr = ct_volume * coefficients
    del ct_volume
    if not isinstance(sigma, float) and not isinstance(sigma, int) and sigma is not None:
        print(f'Incorrect type for argument sigma: {type(sigma)}. Setting to None.')
        sigma = None
    if sigma is None and use_flat:
        if verbose:
            print("Creating corrected images...")
        corr = corr / flat_smoothed
        resid = "_fixed_noresid"
    elif sigma is not None and use_flat:
        if verbose:
            print("Creating residual flat correction...")
        flat_corr = flat * coefficients
        residuals = flat_smoothed - flat_corr
        # filter out residuals to ensure that only those persistant 
        # throughout the CT are adressed (otherwise --> extra rings)
        stds = np.std(residuals) * sigma
        abs_residuals = np.abs(residuals)
        residuals = np.where(abs_residuals < stds, 0., residuals)
        flat = flat_smoothed - residuals
        if verbose:
            print("Creating corrected images...")
        corr = corr / flat
        resid = "_" + str(sigma) + "-sig"
    if save:
        name = save_name + "corr_volume_sm" + str(smooth_radius) + resid + file_type
        path = os.path.join(save_dir, name)
        lazy_corr = da.from_array(corr)
        if file_type == '.h5' or file_type == '.hdf5':
            lazy_corr.to_hdf5(path, '/' + name)
        if file_type == '.zar' or file_type == '.zarr':
            lazy_corr.to_zarr(path)
        if file_type == '.tif' or file_type == '.tiff':
            with TiffWriter(path) as tiff:
                tiff.save(corr)
    return corr

if __name__ == '__main__':
    from skimage.io import imread
    coeffs = imread('/home/abigail/GitRepos/detector-calibration/data/detectorcaleg_stp4_200-500_500-800_coeff_dkrm.tif')
    dark = imread('/home/abigail/GitRepos/detector-calibration/data/detectorcaleg_stp4_200-500_500-800_dark.tif')
    flat = imread('/home/abigail/GitRepos/detector-calibration/data/detectorcaleg_stp4_200-500_500-800_flat.tif')
    ct_vol = imread('/home/abigail/GitRepos/detector-calibration/untracked/phantom_320ms_30kV_23W_sod50_sid150_1_MMStack_Default_200:500_500:800.tif')
    save_dir = '/home/abigail/GitRepos/detector-calibration/untracked'
    name = 'phantom_320ms_30kV_23W_sod50_sid150_1_MMStack_Default_200:500_500:800'
    corr = correct_image(ct_vol, coeffs, dark, flat, save_dir, name, sigma=3)