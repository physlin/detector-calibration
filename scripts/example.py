from skimage.io import imread
beam_sweep = imread('beamsweep.tif')

from detectorcal import fit_response
coefficients = fit_response(beam_sweep)