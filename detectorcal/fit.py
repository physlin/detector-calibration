
import numpy as np
import dask.array as da
from dask.distributed import Client, LocalCluster, as_completed
from numba import jit
from time import time
from toolz import curry
from tifffile import TiffWriter
from pathlib import Path
from shutil import rmtree
# determine cupy will be imported and used
try:
    import cupy as cp
    USE_GPU = True
except ImportError:
    USE_GPU = False



# -------------
# Fit Beamsweep
# -------------

def fit_response(
    volume,
    dark=None, 
    save_path=None,
    sigma=50, 
    mode='nearest',
    cval=0.0, 
    truncate=4.0,
    cutoff=300,
    gpu=False,
    verbose=False,
    ):
    '''
    For each pixel in the detector, fit a linear model predicting the 
    true artefact-free detector response to an X-ray beam. For this, 
    a stack of images representing the detector's response across beam 
    intensities is required. Additionally, a dark current image can be 
    used to correct the dark current for this volume. A Gaussian filter 
    is used to smooth the volume, yielding estimate the true intensity
    profile of the X-ray beam. The standard deviation of the Gaussian
    kernel should be chosen to eliminate pixel-wise variations. For each 
    pixel, least-squares regression is then used to determine a coefficient 
    that can be used to map the 

    Parameters
    ----------
    volume: np.ndarray
        Array containing the stack of detector responses.
    dark: None or np.ndarray
        Default None. Array containing the dark current image or None.
    save_path: None or str
        Default None. Path to which the fit coefficients should be saved.
        If None, the coefficients will not be saved to disk. 
    sigma: scalar
        Default 50. Standard deviation of the Gaussian kernel to be used for 
        smoothing
    mode: str
        Default 'nearest'. Determines how the input array is extended at the
        boarder. 
    cval: scalar
        Default 0.0. Value with which to pad edges when mode 'constant' is 
        used.
    truncate: float
        Default 4.0. truncate filter at this many standard deviations.
    gpu: bool
        Default determined by whether GPU version is installed (with pip 
        install detectorcal[gpu]). Will gpu acceleration be required 
        (or possible)? 
    verbose: bool
        Default False. Should messages be printed to the console? Will print a
        link to dask dashboard - this allows you to watch the computation across
        the workers. 
    cutoff: scalar
        Default 300. Minimum value in smoothed image at which to include 
        the value in the regression. This is chosen to elminate
        values that fall outside of the range of linear response. Value choice
        depends on data type and range (e.g., 300 for 12 bit image). 

    Returns
    -------
    fit: np.ndarray
        Coefficients for each pixel in the detector

    References
    ----------
    Croton, L.C., Ruben, G., Morgan, K.S., Paganin, D.M. and Kitchen, M.J., 
        2019. Ring artifact suppression in X-ray computed tomography using a 
        simple, pixel-wise response correction. Optics express, 27(10), 
        pp.14231-14245.
    '''
    # initialise a local dask cluster and client
    #cluster, client = intialise_cluster_client()
    client = Client(processes=False)
    if verbose:
        print(client.dashboard_link)
    volume = rm_dark(volume, dark, gpu)
    # obtain volume with Gaussian smoothing along x-y planes
    smoothed = gaussian_smooth(volume, sigma=sigma, mode=mode, cval=cval, 
                               truncate=truncate, gpu=gpu, verbose=verbose)
    #fit = sequential_fit(volume, smoothed, cutoff)
    fit = np.zeros((1, volume.shape[1], volume.shape[2]))
    client.close()
    #cluster.close()
    fit = parallel_fit(fit, volume, smoothed, cutoff)
    # save coefficients if a path is provided
    if save_path is not None:
        file_type = Path(save_path).suffix
        lazy_fit = da.from_array(fit)
        if file_type == '.h5' or file_type == '.hdf5':
            lazy_fit.to_hdf5(save_path, '/data')
        if file_type == '.zar' or file_type == '.zarr':
            lazy_fit.to_zarr(save_path)
        if file_type == '.tif' or file_type == '.tiff':
            with TiffWriter(save_path) as tiff:
                tiff.save(fit)
    print(fit.shape)
    return fit


def rm_dark(volume, dark, gpu):
    if dark is not None:
        volume = volume - dark
    return volume

# ------------------
# Gaussian Smoothing
# ------------------


def gaussian_smooth(
    volume: da.Array, 
    sigma=50, 
    mode='nearest',
    cval=0.0, 
    truncate=4.0,
    gpu=False,
    verbose=True,
):
    '''
    Apply a 2D Gaussian filter to slices of a 3D volume using dask. 
    Facilitates dask-mediated parallelisation and has optional
    GPU acceleration. 

    Parameters
    ----------
    volume: np.ndarray
        Array containing a series of x-y planes that require smoothing
        stacked along z. 
    sigma: scalar
        Standard deviation of the Gaussian kernel to be used for smoothing
    mode: str
        Determines how the input array is extended at the boarder. For 
        options see: 
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
    cval: scalar
        Value with which to pad edges when mode 'constant' is used.
    truncate: float
        truncate filter at this many standard deviations.
    gpu: bool
        Will gpu acceleration be required (or possible)?
    verbose: bool
        Should messages be printed to the console?

    Returns
    -------
    smoothed: np.ndarray
        Smoothed version of the input volume. 
    '''
    #as_array = not gpu
    # chunk along the axis along which function should be applied
    _, y, x = volume.shape
    if isinstance(volume, np.ndarray):
        lazy_volume = da.from_array(volume, chunks=(1, y, x)) 
    elif isinstance(volume, da.core.Array):
        lazy_volume = volume
    if gpu:
        from cupyx.scipy.ndimage import gaussian_filter
        #dtype = cp.ndarray
    else:
        from scipy.ndimage.filters import gaussian_filter
    dtype = np.ndarray
    order=0 # derivative order for gaussian kernel - i.e., normal gaussian
    t = time()
    if verbose:
        print('Smoothing volume...')
    gaus_kwargs = {
        'sigma' : sigma,
        'order' : order,
        'mode' : mode,
        'cval' : cval,
        'truncate' : truncate
    }
    if gpu:
        lazy_volume = lazy_volume.map_blocks(cp.array)
    #def inner_GF(array):
     #   if gpu:
      #      array = cp.array(array)
       # sm = gaussian_filter(array, sigma=sigma, order=order, mode=mode, cval=cval, truncate=truncate)
        #if gpu:
        #    sm = sm.get()
        #return sm
    lazy_smoothed = lazy_volume.map_blocks(gaussian_filter, dtype=dtype, **gaus_kwargs)
    if gpu:
        lazy_smoothed = lazy_smoothed.map_blocks(cupy_to_numpy)
    # when the compute is called, as long as a client is active
    #   the dask scheduler will parallelise the work across workers/threads
    smoothed = lazy_smoothed.compute()
    #import napari
    #v = napari.view_image(smoothed)
    #v.add_image(volume)
    #napari.run()
    if verbose:
        m = f'Gaussian smoothing of stack of shape {lazy_volume.shape}'
        m = m + f' completed in {time() - t} seconds'
        print(m)
    return smoothed


def cupy_to_numpy(array):
    return cp.asnumpy(array)


# -------
# Polyfit
# -------

@jit(forceobj=True)
def parallel_fit(fit, volume, smoothed, cutoff):
    '''
    Apply linear regression to each pixel. This is done using a
    numba accelerated nested for loop. 
    '''
    for j in range(volume.shape[1]):
        for i in range(volume.shape[2]):
            points = np.where(smoothed[:,j,i]>cutoff)[0]
            fit[:,j,i] = np.linalg.lstsq(volume[points,j,i].reshape(-1,1), 
                                        smoothed[points,j,i], rcond=None)[0][0]
    return fit[0, ...]


# --------------------
# Sequential Functions
# --------------------

def sequential_gauss(
    volume: np.ndarray, 
    sigma=50, 
    mode='nearest',
    cval=0.0, 
    truncate=4.0,
    ):
    '''
    Apply Gaussian smoothing to image planes in sequence.
    '''
    from scipy.ndimage.filters import gaussian_filter
    t = time()
    out = np.zeros_like(volume)
    for i in range(volume.shape[0]):
        v = gaussian_filter(volume[i, ...], sigma, 0, None, mode, cval, truncate)
        out[i, ...] = v
    print(f'Smoothed volume in {time() - t} seconds')
    return out


def sequential_fit(volume, smoothed, cutoff):
    '''
    Apply linear regression to pixels sequentially.
    '''
    t = time()
    fit = np.zeros(shape=(1,volume.shape[1],volume.shape[2]))

    #print("Fitting volume...")
    # Fit volume
    for j in range(volume.shape[1]):
        #print("Fitting row",j)
        for i in range(volume.shape[2]):
            points = np.where(smoothed[:,j,i]>cutoff)[0]
            fit[:,j,i] = np.linalg.lstsq(volume[points,j,i].reshape(-1,1), 
                                        smoothed[points,j,i], rcond=None)[0][0]
    #print(f'Fitted volume in {time() - t} seconds')
    return fit[0, ...]


# -------------------
# Per Pixel for Plots
# -------------------

def fit_pixel(
    volume, 
    smoothed,
    cutoff, 
    i
    ):
    '''
    Least squares regression for a single pixel. 

    Parameters
    ----------
    volume: np.ndarray
        The detector-response volume. 
    smoothed: np.ndarray
        The detector-response volume once Gaussian smoothed across xy. 
    cutoff: scalar
        Minimum value in smoothed image at which to include 
        the value in the regression. This is chosen to elminate
        values that fall outside of the range of linear response.  
    pair: tuple of int 
        Of the form (j, i), where j and i are the 
        y and x indices respectively.

    Returns
    -------
    pixel_fit: scalar
        Linear coefficient for the pixel.
    i: int
        index of the pixel in flattened corrdinates
        (i.e., raveled along x-y plane).

    '''
    points = np.where(smoothed[:,i]>cutoff)[0]
    measured = volume[points, i].reshape(-1,1) 
    expected = smoothed[points, i]
    pixel_fit = np.linalg.lstsq(measured, expected, rcond=None)[0][0]
    return pixel_fit, i



if __name__ == '__main__':
    import os
    CURRENT_PATH = Path(__file__).parent.resolve()
    SRC_PATH = CURRENT_PATH.parents[0]
    BS_DATA = str(SRC_PATH / 'data/detectorcaleg_stp4_200-500_500-800_beamsweep.tif')
    DARK_DATA = str(SRC_PATH / 'data/detectorcaleg_stp4_200-500_500-800_dark.tif')
    SAVE_DIR = str(SRC_PATH / 'untracked')

    BS_DATA = '/home/abigail/GitRepos/detector-calibration/untracked/beam_sweep_320ms_30kV_23Wmax_sod50_sid150_1_MMStack_Default.ome.tif'
    DARK_DATA = '/home/abigail/GitRepos/detector-calibration/untracked/darks_320ms_avg.tif'
    
    from skimage import io
    bs = io.imread(BS_DATA)
    dk = io.imread(DARK_DATA)

    t = time()
    save_path = os.path.join(SAVE_DIR, 'full_coefficients_no_gpu.tif')
    _ = fit_response(bs, dk, gpu=False, verbose=True, save_path=save_path)
    print(f'Non-gpu time: {time() - t}')

    t = time()
    save_path = os.path.join(SAVE_DIR, 'full_coefficients_gpu.tif')
    _ = fit_response(bs, dk, gpu=True, verbose=True, save_path=save_path)
    print(f'Gpu time: {time() - t}')

    t = time()
    bs = bs - dk
    sm = sequential_gauss(bs)
    _ = sequential_fit(bs, sm, cutoff=300)
    save_path = os.path.join(SAVE_DIR, 'full_coefficients_linear.tif')
    with TiffWriter(save_path) as tiff:
        tiff.save(_)
    print(f'Sequential time: {time() - t}')

