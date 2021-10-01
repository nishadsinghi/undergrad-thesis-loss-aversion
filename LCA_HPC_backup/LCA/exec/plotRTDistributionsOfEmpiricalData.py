import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "..", "src"))

from LCA import PrepareRecurrentWeights, RunLCASimulation, GetValueInputZeroReference, getValueInput2Attributes2Choices
from LUT import prepareStdNormalLUT, SampleFromLUT

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import chisquare

def selectConditionTrials(gain, loss, data):
    selectedData = data[data[:, 2] == gain][:]
    selectedData = selectedData[selectedData[:, 3] == loss][:]
    return selectedData

STAKES = (50, -10)

# actual data
actual = np.genfromtxt("../src/risk_data.csv", delimiter=',')[1:, 1:]

trials = selectConditionTrials(*STAKES, actual)
numTrials = np.shape(trials)[0]
RTMultiplier = np.full((numTrials, 1), 1)
RTMultiplier[trials[:, 0] == 0] = -1
trials[:, 1] *= RTMultiplier.flatten()

allGainValues = range(10, 110, 10)
allLossValues = range(-100, 0, 10)

sns.distplot(trials[:, 1], label='actual data')



# set up look-up table (LUT)
LUTInterval = 0.0001
numAccumulators = 2
stdNormalLUT = prepareStdNormalLUT(LUTInterval)
sampleFromLUT = SampleFromLUT(stdNormalLUT)
sampleFromZeroMeanLUT = lambda stdDev: sampleFromLUT(0, stdDev, numAccumulators)


identityUtilityFunction = lambda x: x
getValueInput = GetValueInputZeroReference(identityUtilityFunction)

maxTimeSteps = 1000
deltaT = 0.02
prepareRecurrentWeights = PrepareRecurrentWeights(numAccumulators)
lca = RunLCASimulation(getValueInput, sampleFromZeroMeanLUT, prepareRecurrentWeights, maxTimeSteps, deltaT)
# lca = RunLCASimulation(getValueInput2Attributes2Choices, sampleFromZeroMeanLUT, prepareRecurrentWeights, maxTimeSteps, deltaT)

attributeProbabilities = (0.5, 0.5)
startingActivation = (0, 0)

getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
# getAllThresholds = lambda threshold: [threshold]*numAccumulators
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold1, threshold2, constantInput, valueScalingParam: \
    lca(attributeProbabilities, getChoiceAttributes(valueScalingParam*gain, valueScalingParam*loss), startingActivation, decay, competition, constantInput,
        noiseStdDev, nonDecisionTime, [threshold1, threshold2])


parameters = [2.0030469987392214, 1.87962304276737, 54.73885211544135, 0.35203754748040317, 27.064738684547272, 46.333538116720824, 103.30686768305385, 1]#[2.1565684684920456, 4.07778103135741, 50.45170393195244, 0.9779209849111549, 10.890761051251813, 27.406060687010097, 80.84052623759135]

allSimulations = [lcaWrapper(*STAKES, *parameters) for _ in
                  range(200)]

_, allModelRTs, allModelResponses = zip(*allSimulations)
mask = np.full(np.size(allModelRTs), 1)
mask[np.array(allModelResponses) == 0] = -1
allModelRTs = np.array(allModelRTs)
allModelRTs *= mask
print(allModelRTs)

sns.distplot(allModelRTs, label='LCA')
plt.title(STAKES)
plt.legend()
plt.xlabel("RT")
plt.show()