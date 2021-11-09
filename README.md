# detectorcal

Python package for detector calibration. The calibration algorithm was initially developed for the correction of CT ring artifacts (Croton et al., 2019) but can be used more broadly for correcting data produced by other types of detectors. 


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

Using detectorcal is a two step process: (1) calibrating the detector response, which is done by `detectorcal.fit_response` ; and (2) applying the calibration to the image that you want to correct, which requires  `detectorcal.correct_image`. Input to these funtions is expected to be `np.ndarray`. Both functions can be used to save output directly when supplied with the `save_path` argument. We support saving output as `.tif`, `.hdf5`, and `.zarr` files. 


## About `detectorcal`

The algorithm implemented in this package corrects artefacts in images produced by a 2D detector by first finding the relationship between detector response and estimated true stimulus intensity. This relationship is then used this to 'reconstruct' an image aquired by the same detector. For this to work, you first need to aquire data that demonstrates the detector response across a range of stimulus intensities. For CT, this would be done by sweeping the detector through the X-ray beam to aquire images across the range of possible intensities. This stimulus-response data should be a 3D array of shape (z, y, x), where the z-axis represents the range of measurements for a single pixel and the x- and y-axes represent the position of each pixel within the detector. 

### Step 1: Fit

During the 'fit' stage of the callibration process, a linear relationship between detector response and true stimulus intensity is found for every pixel in the detector. We can estimate the true intensity by applying a Gaussian filter to the stimulus-response image, which removes local pixel-to-pixel intensity variations. 

<a href="http://www.youtube.com/watch?feature=player_embedded&v=rqQI8tXwAgc" target="_blank">
 <img src="https://github.com/physlin/detector-calibration/blob/finishing-touches/data/detectorcal-fit-thumbnail.png" alt="Watch the video" width="100%" height="100%" border="10" />
</a>


*Image redirects to a video showing an example of* `fit_response`. Animations were created using `napari-animation`.

Fitting is done using the `fit_response` function, for which example usage ish shown below. By default, the standard deviation for the Gaussian kernel used to smooth the image is 50 pixels. This can be adjusted using the optional `sigma` argument.

```Python
from detectorcal import fit_response
from skimage.io import imread # or any array reading modality of your choice

# read in your detector response data as a numpy array. shape = (z, y, x)
volume = imread('path/to/image/volume')

# read in the dark current image to be subtracted from each yx plane. shape = (y, x) 
dark = imread('path/to/dark/current/image')

# Get the linear coefficients for the detector. shape = (y, x)
coefficients = fit_response(volume, dark=dark, save='path/to/which/to/save/coefficients')

# P.S. if you wish to fit the response to a volume without first subtracking the 
# dark current (perhaps you already did this), the `dark` argument can be omitted

# P.S.S. If you don't wish to save the coefficients the `save` argument can also be omitted
```

The plots below show the calibration for 9 random pixels. These plots can be generated using `detectorcal.plot.plot_pixel_calibrations(volume, dark=dark, n_pixels=n)`, which will return fit lines and scatter plots showing measured- vs true intesity for `n` random pixels. 

![Calibrations for 9 random pixels](https://github.com/physlin/detector-calibration/blob/finishing-touches/data/some_pixel_claibrations_0.png)

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

The following figure shows a reconstructed image both with and without residual correction. 

[EMBED IMAGE]

## Citations

If you use this software, please cite the following paper. 

Croton, L. C., ... and Kitchen, M.J., 2021. Detector calibration for effective DT ring artefact supressionwith polychromatic X-ray sources. In Press. 

Croton, L.C., Ruben, G., Morgan, K.S., Paganin, D.M. and Kitchen, M.J., 
    2019. Ring artifact suppression in X-ray computed tomography using a 
    simple, pixel-wise response correction. Optics express, 27(10), 
    pp.14231-14245. https://doi.org/10.1364/OE.27.014231



## Contribution

Community contributions are welcome! If you think of a way to imrpove efficiency or want to help us out with bugs, please reach out to us on the Issues page. Once you've done this, fork the repository and link the issue in your pull request. Contributions will be reviewed by the authors and will have to pass tests on GitHub Actions.  
 
                                                                                        
