from ..correct import correct_image
from pathlib import Path
import pytest
import skimage.io as io


CURRENT_PATH = Path(__file__).parent.resolve()
SRC_PATH = CURRENT_PATH.parents[1]
CT_DATA = SRC_PATH / 'data/detectorcaleg_stp4_200-500_500-800_CT.tif'
FLAT_DATA = SRC_PATH / 'data/detectorcaleg_stp4_200-500_500-800_flat.tif'
DARK_DATA = SRC_PATH / 'data/detectorcaleg_stp4_200-500_500-800_dark.tif'
CORR_DATA = SRC_PATH / 'data/detectorcaleg_stp4_200-500_500-800_corrected.tif'
COEF_DATA = SRC_PATH / 'data/detectorcaleg_stp4_200-500_500-800_coeff.tif'
COEFD_DATA = SRC_PATH / 'data/detectorcaleg_stp4_200-500_500-800_coeff_dkrm.tif'

ct = io.imread(CT_DATA)
flat = io.imread(FLAT_DATA)
dark = io.imread(DARK_DATA)
corr = io.imread(CORR_DATA)
coef = io.imread(COEF_DATA)

def test_correct():
    result = correct_image(ct, coef, dark, flat)
    b = result == corr
    assert b.min() == True
