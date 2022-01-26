'''
Since sampling from a Gaussian can be slow, and it is a central step in our model, I tried to use a supposedly
faster trick: it stores samples from a standard Gaussian, and scales them based on the mean and standard deviation of
the Gaussian. It is a 'look-up table', hence LUT.py.
'''

from scipy.stats import norm
import numpy as np


def prepareStdNormalLUT(interval):
    return np.asarray([norm.ppf(i) for i in np.arange(interval, 1-interval, interval)])


class SampleFromLUT:
    def __init__(self, LUT):
        self.LUT = LUT

    def __call__(self, mean, stdDev, numSamples):
        stdNormalSamples = np.random.RandomState().choice(self.LUT, size=numSamples)
        transformedSamples = stdNormalSamples * stdDev + mean

        return transformedSamples