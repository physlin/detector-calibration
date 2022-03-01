import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = 'detectorcal',
    version = '0.0.1',
    author = 'Abigail McGovern',
    author_email = 'Abigail.McGovern1@monash.edu',
    description = 'Ring artefact suppression in X-ray CT using pixel-wise linear correction',
    long_description = long_description,
    long_description_content_type = 'text/markdown', 
    license = 'BSD 2-Clause License',
    url = 'https://github.com/physlin/detector-calibration',
    project_urls = {
        'Bug Tracker' : 'https://github.com/physlin/detector-calibration/issues'
    },
    classifiers =
        ['Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License', 
        'Operating System :: OS Independent', 
        'Intended Audience :: Science/Research', 
        'Topic :: Scientific/Engineering', 
        'Topic :: Scientific/Engineering :: Image Processing', 
        'Topic :: Scientific/Engineering :: Physics'],
    packages = setuptools.find_packages(),
    python_requires = '>=3.6',
    install_requires =
        ['dask',
        'distributed', 
        'scipy',
        'numpy',
        'matplotlib',
        'scikit-image', 
        'pytest', 
        'sphinx', 
        'h5py', 
        'zarr', 
        'numba',
        'numpydoc'],
    extras_require={
        'testing': ['pytest', 'pytest-cov', 'coverage', 'asv'],
        'gpu': ['cupy'],  # pip install detectorcal[gpu]
        'docs': ['sphinx', 'furo', 'numpydoc'],
    },
)