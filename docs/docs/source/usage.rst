Usage & About
=============

About detectorcal
-----------------

The algorithm implemented in this package corrects artefacts in images produced by a 2D detector by first finding the relationship between detector response and estimated true stimulus intensity. This relationship is then used this to 'reconstruct' an image aquired by the same detector. For this to work, you first need to aquire data that demonstrates the detector response across a range of stimulus intensities. In X-ray computed tomography, this would be done by sweeping the detector through the X-ray beam to aquire images across the range of possible intensities. This stimulus-response data should be a 3D array of shape (z, y, x), where the z-axis represents the range of measurements for a single pixel and the x- and y-axes represent the position of each pixel within the detector.


Step 1: Fit 
-----------

During the 'fit' stage of the callibration process, a linear relationship between detector response and true stimulus intensity is found for every pixel in the detector. If the stimulus beam can be assumed to emmit a smooth signal, we can estimate the true intensity by applying a Gaussian filter to the stimulus response image thereby removing local pixel-to-pixel intensity variations. 

Fitting is done using the fit_response function, for which example usage ish shown below. By default, the standard deviation for the Gaussian kernel used to smooth the image is 50 pixels. This can be adjusted using the optional sigma argument.

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

Once the coefficients for the detector are found these can be used to correct data aquired by the detector. The correction applies the linear coefficients to adjust the aquired data. When provided with a flat field image (i.e., mean of flat field [sample free] images aquired with the CT sequence), a the flat field can be used for flat field correction. For this, the flat field image is first smoothed (using the same gaussian smoothing as in step 1), which provides an estimate of the true flat field intesity. This can be used to normalise the corrected image. 

If there is likely to be any substantial changes in the optical system between aquiring the detector response volume and the sample image (e.g., dust on the sensor, etc), these changes can be corrected for by finding the residual differences between the estimate of the true flat field (smoothed) and the corrected flat field (i.e., original flat field corrected using coefficients). These residuals can be removed from the smoothed flat field prior to flat field correction. Because the residuals themselves are prone to noise, removal of all residuals may introduce new artifacts. For this reason, only large (i.e., real) residuals greater than a user defined number of standard deviationsfrom the mean are subtracted (we suggest sigma = 3).

Correction is performed using the correct_image function. Example usage is shown below:

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
