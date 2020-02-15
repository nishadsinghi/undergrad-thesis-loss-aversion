from scipy.stats import norm
import numpy as np


def prepareStdNormalLUT(interval):
    return np.asarray([norm.ppf(i) for i in np.arange(interval, 1-interval, interval)])


class SampleFromLUT:
    def __init__(self, LUT):
        self.LUT = LUT

    def __call__(self, mean, stdDev, numSamples):
        stdNormalSamples = np.random.choice(self.LUT, size=numSamples)
        transformedSamples = stdNormalSamples * stdDev + mean

        return transformedSamples