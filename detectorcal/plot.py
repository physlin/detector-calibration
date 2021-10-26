from dask.utils import factors
import numpy as np
import matplotlib.pyplot as plt
from .fit import fit_pixel, gaussian_smooth
from functools import reduce


def plot_pixel_calibrations(
    volume, 
    dark=None,
    n_pixels=9, 
    save_path=None,
    show=True,
    sigma=50, 
    cutoff=300, 
    coords=None, 
    gpu=False, 
    inch_per_plot=3
    ):
    '''
    Plot the calibration a selected number of pixels. If not otherwise
    specified, n random pixels will be selected. Otherwise, the coords
    argument can be used to provide (x, y) coordinates for the pixel to 
    be plotted. 

    Parameters
    ----------
    volume: np.ndarray
        Detector volume to be used for calibration 
    n_pixels: int
        Number of random pixels for which to plot the calibration
    save_path: None or str
        Optional path to which to save the output (probably .png)
    show: bool
        Should matplotlib display the plot? Perhaps this isnt necessary
        if you are saving the output.
    sigma: scalar
        Standard deviation for the Gaussian Kernel used for smoothing
    cutoff: scalar
        ##
    coords: list of list
        List of the form [[y_coord, x_coord], ...]
    inch_per_plot: scalar
        How many inches to add to figure size in each dim for each plot
    '''
    if dark is not None:
        volume = volume - dark
    z, y, x = volume.shape
    smoothed = gaussian_smooth(volume, sigma=sigma, gpu=gpu, verbose=False)
    smoothed = smoothed.reshape(z, y * x) # effectively raveling the xy plane
    volume = volume.reshape(z, y * x)
    if coords is None:
        # random raveled indicies for pixels
        idxs = np.random.choice(np.arange(y * x), size=n_pixels)
    else:
        # get the raveled indices for the xy plane
        coords = np.array(coords).T
        idxs = np.ravel_multi_index(coords)
    fits = []
    for i in idxs:
        # linear fits only for the selected pixels
        pf, i = fit_pixel(volume, smoothed, cutoff, i)
        fits.append(pf)
    # find the layout
    n_plots_y, n_plots_x = min_diff_factors(n_pixels)
    # plot for each pixel
    fig, axs = plt.subplots(n_plots_y, n_plots_x)
    fig.set_size_inches(inch_per_plot * n_plots_y, inch_per_plot * n_plots_x)
    axs = axs.ravel()
    for i, idx in enumerate(idxs):
        xs = volume[:, idx]
        ys = smoothed[:, idx]
        axs[i].scatter(xs, ys)
        axs[i].set_xlabel(f'Measured intensity ({volume.dtype})')
        axs[i].set_ylabel(f'Estimated true intensity ({smoothed.dtype})')
        # plot the calibration line (interpolation only)
        x_max = np.ceil(xs.max()).astype(int)
        x_min = np.floor(xs.min()).astype(int)
        x_line = np.arange(x_min, x_max)
        axs[i].plot(x_line, x_line * fits[i])
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
    if show:
        plt.show()


def min_diff_factors(n):
    factors = get_factors(n)
    if len(factors) == 1:
        # primes will never be next to one another
        factors = get_factors(n + 1)
    min_diff = n
    idx = 0
    for i, fs in enumerate(factors):
        n0, n1 = fs
        if abs(n1-n0) < min_diff:
            min_diff = abs(n1-n0)
            idx = i
    return factors[idx]


def get_factors(n):
    step = 2 if n%2 else 1
    return reduce(list.__add__,
                ([(i, n//i)] for i in range(1, int(np.sqrt(n))+1, step) if n % i == 0))

if __name__ == '__main__':
    bs_path = '/home/abigail/GitRepos/detector-calibration/data/detectorcaleg_stp4_200-500_500-800_beamsweep.tif'
    d_path = '/home/abigail/GitRepos/detector-calibration/data/detectorcaleg_stp4_200-500_500-800_dark.tif' 
    from tifffile import imread
    bs = imread(bs_path)
    dark = imread(d_path)
    plot_pixel_calibrations(bs, dark)