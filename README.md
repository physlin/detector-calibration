# detectorcal

[![PyPI](https://img.shields.io/pypi/v/detectorcal.svg?color=green)](https://pypi.org/project/detectorcal)

Detectorcal is a pixel-by-pixel detector calibration for quantitative direct imaging and suppression of CT ring artefacts. Detectorcal's documentation can be found [here](physlin.github.io). 


## Installation 
The `detectorcal` package can be found on the Python Package Index (PyPI) and can be installed using pip. There are two extra install options including GPU support and testing capabilities. 

```bash
# CPU only installation
pip install detectorcal

# with GPU support
pip install detectorcal[gpu]

# with testing capabilities
pip install detectorcal[testing]
```

## Usage

Using detectorcal is a two step process: (1) calibrating the detector response, which is done by `detectorcal.fit_response` ; and (2) applying the calibration to the image or images that you want to correct, which requires  `detectorcal.correct_image`. Input to these funtions is expected to be `np.ndarray`. Both functions can be used to save output directly when supplied with the `save_path` argument. We support saving output as `.tif`, `.hdf5`, and `.zarr` files. 


## About `detectorcal`

The algorithm implemented in this package calibrates the response of a 2D detector using the measured pixel values, as well as an estimate of the true beam intensity, over a range of image intensities. This correction is then applied to an image or images aquired by the same detector. For this to work, you first need to aquire an image sequence across a range of beam intensities. For a synchrotron source, this would be done by sweeping the detector through the X-ray beam, exploiting the roll-off of the beam to aquire images across the range of possible intensities at every pixel. This intensity sequence should be a 3D array of shape (z, y, x), where the z-axis represents the range of measurements for each pixel, and the x- and y-axes represent the position of each pixel on the detector. 

### Step 1: Fit

During the 'fit' stage of the calibration process, a linear regression is performed between the calibration sequence and the estimated true beam intensity for every pixel on the detector. We estimate the true intensity by smoothing each image of the calibration sequence with a Gaussian filter, which removes local pixel-to-pixel intensity variations. 

<a href="http://www.youtube.com/watch?feature=player_embedded&v=rqQI8tXwAgc" target="_blank">
 <img src="https://github.com/physlin/detector-calibration/blob/main/data/detectorcal-fit-thumbnail.png" alt="Watch the video" width="100%" height="100%" border="10" />
</a>


*Image redirects to a video showing an example of* `fit_response`. Animations were created using `napari-animation`.

Fitting is performed using the `fit_response` function, for which example usage is shown below. By default, the standard deviation for the Gaussian kernel used to smooth the image is 50 pixels. This can be adjusted using the optional `sigma` argument.

```Python
from detectorcal import fit_response
from skimage.io import imread # or any array reading modality of your choice

# read in your detector response data as a numpy array. shape = (z, y, x)
volume = imread('path/to/image/volume')

# read in the dark current image to be subtracted from each yx plane. shape = (y, x) 
dark = imread('path/to/dark/current/image')

# Get the linear coefficients for the detector. shape = (y, x)
coefficients = fit_response(volume, dark=dark, save_path='path/to/which/to/save/coefficients')

# P.S. if you wish to fit the response to a volume without first subtracting the 
# dark current (perhaps you already did this), the `dark` argument can be omitted

# P.S.S. If you don't wish to save the coefficients the `save` argument can also be omitted
```

The plots below show the calibration for 9 random pixels. These plots can be generated using `detectorcal.plot.plot_pixel_calibrations(volume, dark=dark, n_pixels=n)`, which will return fit lines and scatter plots showing measured- vs true intesity for `n` random pixels. 

![Calibrations for 9 random pixels](https://github.com/physlin/detector-calibration/blob/main/data/some_pixel_claibrations_0.png)

## Step 2: Correct

Once the coefficients for the detector are found, they can be used to correct data aquired by the detector using `correct_image`. If you provide a flat field image (i.e., mean of flat field [sample free] images aquired with the CT sequence), the function will apply flat field correction. For flat field correction, the flat field is first smoothed (using the same gaussian smoothing as in step 1), then used to normalise the corrected image. 

If there is likely to be any substantial changes in the optical system between aquiring the detector response volume and the sample image (e.g., dust on the sensor, etc), these changes can also be corrected using the flat field image. This is done by finding the residual differences between the estimated true flat field (smoothed) and the coefficient-corrected flat field (using coefficients found in step 1). The residuals can be removed from the smoothed flat field prior to flat field correction. Because the residuals themselves are prone to noise, removal of all residuals may introduce new artifacts. For this reason, only large (i.e., real) residuals should be removed. The user can define the number of standard deviations above which residuals will be removed (we suggest `sigma = 3`). By default the `sigma` argument is `None` meaning that the residual correction will not go ahead. 
 
```Python
from detectorcal import correct_image
from skimage.io import imread

# get the coefficients (either using the method above or from file)
coefficients = imread('path/to/coefficients')

# get the image you wish to correct (shape = (z, y, x) or (y, x))
image = imread('path/to/image')

# get the dark field and flat field images for flat field correction
dark = imread('path/to/dark/field') # shape = (y, x)
flat = imread('path/to/flat/field') # shape = (y, x)

# obtain the corrected image
corrected = correct_image(image, coefficients, dark=dark, flat=flat)

# if you wish to correct with residual correction
resid_corrected = correct_image(image, coefficients, dark=dark, flat=flat, sigma=3.)

```

![correction](https://github.com/physlin/detector-calibration/blob/main/data/detectorcal-correct.png)

A reconstructed image both with and without correction. Figure modified from Croton et al. (2019).

### Large Arrays
When the input volume is too large, the former code will raise a memory error. If this is the case, computation can still be completed by setting the `use_dask` flag as `True`. Using `dask` as a backend, this allows computations to be carried out and written to disk without ever exceeding RAM. If the array is too large to read in as a numpy array, please use `dask` or `dask_image`. When you input a dask array, the `use_dask` flag will automatically be set to `True`. Also note that when the corrected image is expected to be bigger-than-RAM, the file should be saved as a zarr or hdf5, as we do not currently support big tiffs. 

```Python
from dask_image.imread import imread
image = imread('path/to/image')

# save path as zarr or hdf5
save_path = 'path/to/which/to/save/corrected.zarr'

# obtain the corrected image
corrected = correct_image(image, coefficients, dark=dark, flat=flat, save_path=save_path)


```


## Citations

*If you use this software, please cite the following:*

Croton, L.C., Ruben, G., Morgan, K.S., Paganin, D.M. and Kitchen, M.J., 
    2019. Ring artifact suppression in X-ray computed tomography using a 
    simple, pixel-wise response correction. Optics express, 27(10), 
    pp.14231-14245. https://doi.org/10.1364/OE.27.014231

*In preparation:*

L.C.P. Croton, G. Ruben, K.S. Morgan, A.S. McGovern, and M.J. Kitchen, Detector calibration for CT ring artifact suppression with polychromatic X-ray sources, in prep


## Contribution

Community contributions are welcome! If you think of a way to imrpove efficiency or want to help us out with bugs, please reach out to us on the Issues page. Once you've done this, fork the repository and link the issue in your pull request. Contributions will be reviewed by the authors and will have to pass tests on GitHub Actions.  
 
                                                                                        
