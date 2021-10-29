from detectorcal.fit import fit_response
import numpy as np
from pathlib import Path
import pytest
import skimage.io as io

CURRENT_PATH = Path(__file__).parent.resolve()
SRC_PATH = CURRENT_PATH.parents[1]
BS_DATA = str(SRC_PATH / 'data/detectorcaleg_stp4_200-500_500-800_beamsweep.tif')
DARK_DATA = str(SRC_PATH / 'data/detectorcaleg_stp4_200-500_500-800_dark.tif')
COEFD_DATA = str(SRC_PATH / 'data/detectorcaleg_stp4_200-500_500-800_coeff_dkrm.tif')

bs = io.imread(BS_DATA)
dk = io.imread(DARK_DATA)
coeffd = io.imread(COEFD_DATA)


def test_fit_rmdk(bs=bs, dk=dk, coeff=coeffd):
    out = fit_response(bs, dark=dk)
    b = out == coeffd
    assert b.min() == True