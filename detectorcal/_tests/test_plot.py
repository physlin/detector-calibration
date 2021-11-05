from detectorcal.plot import plot_pixel_calibrations
import numpy as np

def test_plotting_random():
    volume = np.random.random((10, 100, 100))
    plot_pixel_calibrations(volume, show=False)

def test_plotting_chosen():
    pixels = [[0, 0], [0, 1], [1, 0], 
              [1, 1], [0, 2], [2, 0], 
              [1, 2], [2, 1], [2, 2]]
    volume = np.random.random((10, 100, 100))
    plot_pixel_calibrations(volume, coords=pixels, show=False)