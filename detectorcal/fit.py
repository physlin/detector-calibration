
import numpy as np
import dask.array as da
from dask.distributed import Client, LocalCluster
from time import time
from toolz import curry
from tifffile import TiffWriter
from scipy.ndimage.filters import gaussian_filter
from multiprocessing import Pool, cpu_count
from pathlib import Path
from shutil import rmtree
# determine cupy will be imported and used
#try:
 #   import cupy as cp
  #  USE_GPU = True
#except ImportError:
 #   USE_GPU = False
#USE_GPU = True
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
    n_processes=None
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
        Array containing the dark current image or None.
    save_path: None or str
        Path to which the fit coefficients should be saved.
        If None, the coefficients will not be saved to disk. 
    sigma: scalar
        Standard deviation of the Gaussian kernel to be used for smoothing
    mode: str
        Determines how the input array is extended at the boarder. 
    cval: scalar
        Value with which to pad edges when mode 'constant' is used.
    truncate: float
        truncate filter at this many standard deviations.
    gpu: bool
        Will gpu acceleration be required (or possible)? If unspecified, this will
        be determined by whether the cupy package is installed (as it will be if
        detectorcal has been installed with the .[gpu] option)
    verbose: bool
        Should messages be printed to the console?
    hpc: bool
        Will the cluster be a HPC cluster?
    cutoff: scalar
        Minimum value in smoothed image at which to include 
        the value in the regression. This is chosen to elminate
        values that fall outside of the range of linear response.  

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
    if gpu:
        try:
            import cupy as cp
            volume = cp.array(volume)
            dark = cp.array(dark)
        except ImportError:
            raise ImportError('Please install cupy into your local environment to use GPU acceleration')
    # initialise a local dask cluster and client
    cluster, client = intialise_cluster_client()
    if dark is not None:
        # remove the dark current
        volume = volume - dark
    # obtain volume with Gaussian smoothing along x-y planes
    smoothed = gaussian_smooth(volume, sigma=sigma, mode=mode, cval=cval, 
                               truncate=truncate, gpu=gpu, verbose=verbose)
    client.close()
    cluster.close()
    # new client
    #_, client = intialise_cluster_client()
    # use smoothed volume to fit values
    if gpu:
        smoothed = smoothed.get()
        volume = volume.get()
    #fit = sequential_fit(volume, smoothed, cutoff)
    fit = polyfit_deg_1(client, volume, smoothed, cutoff, verbose, n_processes)
    # save coefficients if a path is provided
    if save_path is not None:
        if gpu:
            if isinstance(fit, cp._core.core.ndarray):
                fit = fit.get()
        file_type = Path(save_path).suffix
        lazy_fit = da.from_array(fit)
        if file_type == '.h5' or file_type == '.hdf5':
            lazy_fit.to_hdf5(save_path, '/data')
        if file_type == '.zar' or file_type == '.zarr':
            lazy_fit.to_zarr(save_path)
        if file_type == '.tif' or file_type == '.tiff':
            with TiffWriter(save_path) as tiff:
                tiff.save(fit)
    client.close()
    return fit


# -------------
# Dask Specific
# -------------

def intialise_cluster_client():
    '''
    Parameters
    ----------
    hpc: bool
        Will the cluster be a HPC cluster?

    TODO: add hpc compatibility if there is time
    '''
    cluster = LocalCluster(processes=False)
    client = Client(cluster)
    #print('Dask dashboard can be found at the following link')
    #print(client.dashboard_link)
    return cluster, client


# ------------------
# Gaussian Smoothing
# ------------------


def gaussian_smooth(
    volume: np.ndarray, 
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
    '''
    if gpu:
        import cupy as cp
        from cupyx.scipy.ndimage import gaussian_filter
        dtype = cp.ndarray
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
    y, x = volume.shape[1], volume.shape[2]
    # chunk along the axis along which function should be applied
    as_array = not gpu
    lazy_volume = da.from_array(volume, chunks=(1, y, x), asarray=as_array) 
    #if gpu:
        #lazy_volume = lazy_volume.map_blocks(cp.asarray)
    # when the function is called it will be applied separeatley 
    #    to each slice in z. I checked that this is true on dummy data
    lazy_smoothed = lazy_volume.map_blocks(gaussian_filter, dtype=dtype, **gaus_kwargs)
    # when the compute is called, as long as a client is active
    #   the dask scheduler will parallelise the work across workers/threads
    smoothed = lazy_smoothed.compute()
    if verbose:
        m = f'Gaussian smoothing of stack of shape {volume.shape}'
        m = m + f' completed in {time() - t} seconds'
        print(m)
    return smoothed


# -------
# Polyfit
# -------

def polyfit_deg_1(
    client, 
    volume, 
    smoothed, 
    cutoff, 
    verbose, 
    n_processes
    ):
    '''
    Linear fit for every pixel in the detector. 
    Code is parallelised via dask client.  

    Parameters
    ----------
    client: dask.distributed.Client
    volume: np.ndarray
    smoothed: np.ndarray
    cutoff: scalar
    '''
    t = time()
    z, y, x = tuple(volume.shape)
    if verbose:
        print('Fitting volume... ')
    # Initialise fit array
    fit = np.zeros(shape=(y,x)).ravel()
    volume = volume.reshape(z, y * x)
    smoothed = smoothed.reshape(z, y * x)
    # get the coordinates of each pixel to be fitted
    #y, x = range(volume.shape[1]), range(volume.shape[2])
    #pairs = [p for p in product(y, x)]
    idxs = list(range(len(fit)))
    c_fit_pixel = curry(fit_pixel)
    c_fit_pixel = c_fit_pixel(volume, smoothed, cutoff)
    pf, i = c_fit_pixel(idxs[0])
    if n_processes is None:
        n_processes = cpu_count()
    with Pool(n_processes) as p:
        result = p.map(c_fit_pixel, idxs)
    for pf, i in result:
        fit[i] = pf
    #f = client.map(c_fit_pixel, idxs)
    #counter = 0
    #for future in as_completed(f):
     #   pixel_fit, i = future.result()
      #  fit[i] = pixel_fit
       # if counter % 500 == 499 or counter == 1:
        #    print(i, pixel_fit)
       # counter += 1
    if verbose:
        m = f'Linear fit completed in {time() - t} seconds'
        print(m)
    fit = fit.reshape(y, x)
    return fit


def fit_pixel(
    volume, 
    smoothed,
    cutoff, 
    i
    ):
    '''
    Least squares regression for a single pixel

    Parameters
    ----------
    volume: np.ndarray
    smoothed: np.ndarray
    cutoff: scalar
    pair: tuple of int 
        Of the form (j, i), where j and i are the 
        y and x indices respectively.
    '''
    points = np.where(smoothed[:,i]>cutoff)[0]
    measured = volume[points, i].reshape(-1,1) 
    expected = smoothed[points, i]
    pixel_fit = np.linalg.lstsq(measured, expected, rcond=None)[0][0]
    return pixel_fit, i



def sequential_gauss(
    volume: np.ndarray, 
    sigma=50, 
    mode='nearest',
    cval=0.0, 
    truncate=4.0,
):
    out = np.zeros_like(volume)
    for i in range(volume.shape[0]):
        v = gaussian_filter(volume[i, ...], sigma, 0, None, mode, cval, truncate)
        out[i, ...] = v
    return out


def sequential_fit(volume, smoothed, cutoff):
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



if __name__ == '__main__':
    bs_path = '/home/abigail/GitRepos/detector-calibration/data/detectorcaleg_stp4_200-500_500-800_beamsweep.tif'
    d_path = '/home/abigail/GitRepos/detector-calibration/data/detectorcaleg_stp4_200-500_500-800_dark.tif'
    out_path = '/home/abigail/GitRepos/detector-calibration/untracked/detectorcaleg_stp4_200-500_500-800_coeff_dkrm_gpu.tif'
    from tifffile import imread
    bs = imread(bs_path)
    dark = imread(d_path)
    fit = fit_response(bs, dark, save=out_path, gpu=True)
    import napari
    v = napari.view_image(fit)
    napari.run()


