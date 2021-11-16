.. detectorcal documentation master file, created by
   sphinx-quickstart on Mon Oct 18 16:04:40 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to detectorcal's documentation!
=======================================


Python package for detector calibration. The algorithm was initially developed for the correction of CT ring artifacts (Croton et al., 2019) but can be used for correcting data produced by other types of detectors. 

.. note::

   We welcome community contributions. If you have any bug fixes or ideas please raise an issue and create a linked pull request. Pull requests will be reviewed by the package maintainers.


Installation
------------

detectorcal can be found on the Python Package Index (PyPI) and can be installed using pip using the following command:

.. code-block:: console

   # CPU only version
   $ pip install detectorcal

   # GPU version
   $ pip install detectorcal[gpu]

   # with testing suite
   $ pip install detectorcal[testing]

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Contents
--------

.. toctree::
   usage
   api/api
   
