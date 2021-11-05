import numpy as np
from detectorcal.fit import fit_response, sequential_fit, sequential_gauss

class FitSuite:
    '''Benchmark for fit module methods in detectorcal'''
    
    def setup(self):
        self.bs = np.random.random((25, 100, 100))
        self.dark = np.random.random((100, 100))

    
    def time_fit_no_gpu(self):
        _ = fit_response(self.bs, self.dark, gpu=False)


    def time_fit_gpu(self):
        try:
            import cupy as cp
            _ = fit_response(self.bs, self.dark, gpu=True)
        except ImportError:
            pass

    
    def time_fit_sequential(self):
        smoothed = sequential_gauss(self.bs)
        _ = sequential_fit(self.bs, smoothed, 0.9)