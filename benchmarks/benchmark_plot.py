from detectorcal.plot import plot_pixel_calibrations
import numpy as np


class PlotSuite:
    '''Benchmarks for detectorcal's plot module'''

    def setup(self):
        self.bs = np.random.random((25, 100, 100))
        self.dark = np.random.random((100, 100))

    
    def test_plot(self):
        plot_pixel_calibrations(self.bs, self.dark, show=False)