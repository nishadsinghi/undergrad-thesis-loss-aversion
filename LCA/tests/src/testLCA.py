import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "..", "..", "src"))

import unittest
from ddt import ddt, data, unpack
import numpy as np

from LCA import UtilityFunction, GetValueInput, PrepareRecurrentWeights

@ddt
class TestUtilityFunction(unittest.TestCase):
    def setUp(self):
        self.k = 2
        self.gamma = 1
        self.utilityFunction = UtilityFunction(self.k, self.gamma)

    @data((2, 1.60943791243), (15, 3.43398720449), (0, 0), (-4, -2.19722), (-30, -4.110873))
    @unpack
    def testUtilityFunction(self, input, groundTruthOutput):
        self.assertAlmostEqual(self.utilityFunction(input), groundTruthOutput, 4)


@ddt
class TestGetValueInput(unittest.TestCase):
    def setUp(self):
        self.k = 2
        self.gamma = 1
        self.utilityFunction = UtilityFunction(self.k, self.gamma)
        self.getValueInput = GetValueInput(self.utilityFunction)

    @data(((0.5, 0.5), np.array([[1, 2], [3, 4]]), (1.35402510055, 2.0715673632)),
          ((0.5, 0.5), np.array([[5, 6], [7, 8]]), (2.48142231513, 2.77063177258)))
    @unpack
    def testGetValueInput(self, attributeProbabilities, choiceAttributes, groundTruthMean):
        allValueInputs = [self.getValueInput(attributeProbabilities, choiceAttributes) for _ in range(100000)]
        allValueInputs = np.vstack(allValueInputs)
        sampleMean = np.mean(allValueInputs, axis=0)

        [self.assertAlmostEqual(sampleMean[i], groundTruthMean[i], 2) for i in range(len(sampleMean))]


@ddt
class TestPrepareRecurrentWeights(unittest.TestCase):
    @data((2, 3, 4, np.asarray([[-3, -4], [-4, -3]])), (4, -1, -2, np.asarray([[1, 2, 2, 2], [2, 1, 2, 2], [2, 2, 1, 2], [2, 2, 2, 1]])))
    @unpack
    def testPrepareRecurrentWeights(self, size, decay, competition, groundTruthWeights):
        prepareRecurrentWeights = PrepareRecurrentWeights(size)
        recurrentWeights = prepareRecurrentWeights(decay, competition)

        np.testing.assert_array_equal(groundTruthWeights, recurrentWeights)





if __name__ == "__main__":
    unittest.main()