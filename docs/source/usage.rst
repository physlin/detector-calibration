Using Detectorcal
=================

About detectorcal
-----------------

The algorithm implemented in this package corrects artefacts in images produced by a 2D detector by first finding the relationship between detector response and estimated true stimulus intensity. This relationship is then used this to 'reconstruct' an image aquired by the same detector. For this to work, you first need to aquire data that demonstrates the detector response across a range of stimulus intensities. For CT, this would be done by sweeping the detector through the X-ray beam to aquire images across the range of possible intensities. This stimulus-response data should be a 3D array of shape (z, y, x), where the z-axis represents the range of measurements for a single pixel and the x- and y-axes represent the position of each pixel within the detector. 


Usage
-----

Using detectorcal is a two step process: (1) calibrating the detector response, which is done by ``detectorcal.fit_response`` ; and (2) applying the calibration to the image that you want to correct, which requires  ``detectorcal.correct_image``. Input to these funtions is expected to be ``np.ndarray``. Both functions can be used to save output directly when supplied with the ``save_path`` argument. We support saving output as ``.tif``, ``.hdf5``, and ``.zarr`` files. 

Step 1: Fit 
-----------

During the 'fit' stage of the callibration process, a linear relationship between detector response and true stimulus intensity is found for every pixel in the detector. We can estimate the true intensity by applying a Gaussian filter to the stimulus-response image, which removes local pixel-to-pixel intensity variations. g is done using the fit_response function, for which example usage ish shown below. By default, the standard deviation for the Gaussian kernel used to smooth the image is 50 pixels. This can be adjusted using the optional sigma argument.

Fitting is done using the ``fit_response`` function, for which example usage ish shown below. By default, the standard deviation for the Gaussian kernel used to smooth the image is 50 pixels. This can be adjusted using the optional ``sigma`` argument.

.. code-block:: python

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



Step 2: Correct 
--------------- 

Once the coefficients for the detector are found, they can be used to correct data aquired by the detector using ``correct_image``. If you provide a flat field image (i.e., mean of flat field [sample free] images aquired with the CT sequence), the function will apply flat field correction. For flat field correction, the flat field is first smoothed (using the same gaussian smoothing as in step 1), then used to normalise the corrected image. 

If there is likely to be any substantial changes in the optical system between aquiring the detector response volume and the sample image (e.g., dust on the sensor, etc), these changes can also be corrected using the flat field image. This is done by finding the residual differences between the estimated true flat field (smoothed) and the coefficient-corrected flat field (using coefficients found in step 1). The residuals can be removed from the smoothed flat field prior to flat field correction. Because the residuals themselves are prone to noise, removal of all residuals may introduce new artifacts. For this reason, only large (i.e., real) residuals should be removed. The user can define the number of standard deviations above which residuals will be removed (we suggest ``sigma = 3``). By default the ``sigma`` argument is ``None`` meaning that the residual correction will not go ahead. 

.. code-block:: python

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


Correcting Large Arrays
-----------------------

When the input volume is too large, the former code will raise a memory error. If this is the case, computation can still be completed by setting the ``use_dask`` flag as ``True``. Using ``dask`` as a backend, this allows computations to be carried out and written to disk without ever exceeding RAM. If the array is too large to read in as a numpy array, please use ``dask`` or ``dask_image``. When you input a dask array, the ``use_dask`` flag will automatically be set to ``True``. Also note that When the corrected image is expected to be bigger-than-RAM, the file should be saved as a zarr or hdf5, as we do not currently support big tiffs. 

.. code-block:: python
   
   from dask_image.imread import imread
   image = imread('path/to/image')
   
   # save path as zarr or hdf5
   save_path = 'path/to/which/to/save/corrected.zarr'
   
   # obtain the corrected image
   corrected = correct_image(image, coefficients, dark=dark, flat=flat, save_path=save_path)
