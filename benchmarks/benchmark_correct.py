import numpy as np
from detectorcal.correct import correct_image


class CorrectSuite:
    '''Benchmarks for detectorcals correct module'''

    def setup(self):
        self.ct = np.random.random((50, 100, 100))
        self.ff = np.random.random((100, 100))
        self.dk = np.random.random((100, 100))
        self.cf = np.random.random((100, 100))

    
    def time_correct_no_gpu(self):
        _ = correct_image(self.ct, self.cf, self.dk, self.ff, gpu=False)


    def time_correct_gpu(self):
        try:
            import cupy as cp
            _ = correct_image(self.ct, self.cf, self.dk, self.ff, gpu=True)
        except ImportError:
            pass
