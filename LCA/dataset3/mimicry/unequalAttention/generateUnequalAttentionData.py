import sys
import os

DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "..", "..", "..", "src"))

import numpy as np

from LCA import PrepareRecurrentWeights, RunLCASimulation, \
    GetValueInputZeroReference
from LUT import prepareStdNormalLUT, SampleFromLUT


experimentalData = np.genfromtxt("data_cleaned_250_10500_zScore3.csv", delimiter=',')[:, 1:]
allStakes = np.unique(experimentalData[:, -2:], axis=0)


# LCA
def filterFunction(tup):
    if tup[-1] != -1:
        return True
    else:
        return False

numSimulationsPerCondition = 200#150

# set up look-up table (LUT)
LUTInterval = 0.0001
numAccumulators = 2
stdNormalLUT = prepareStdNormalLUT(LUTInterval)
sampleFromLUT = SampleFromLUT(stdNormalLUT)
sampleFromZeroMeanLUT = lambda stdDev: sampleFromLUT(0, stdDev, numAccumulators)

# set-up the LCA
identityUtilityFunction = lambda x: x
getValueInput = GetValueInputZeroReference(identityUtilityFunction)

maxTimeSteps = 1000
deltaT = 0.02
prepareRecurrentWeights = PrepareRecurrentWeights(numAccumulators)
lca = RunLCASimulation(getValueInput, sampleFromZeroMeanLUT, prepareRecurrentWeights, maxTimeSteps, deltaT)

startingActivation = (0, 0)
getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
getAllThresholds = lambda threshold: [threshold]*numAccumulators

def getStartingActivation(startingBias, threshold):
    if startingBias < 0:
        return [-1*startingBias*threshold, 0]
    else:
        return [0, startingBias*threshold]


lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, constantInput, attribute1Prob: \
    lca((attribute1Prob, 1-attribute1Prob), getChoiceAttributes(gain, loss), getStartingActivation(0, threshold),
        decay, competition, constantInput, noiseStdDev, nonDecisionTime, getAllThresholds(threshold))


parameters = [ 2.80603906,  3.56727085,  8.01231325,  0.47240811,  2.66769535, 10.20452434, 0.34151331]

allModelData = np.zeros(4)
for stakes in allStakes:
    gainValue, lossValue = stakes
    allSimulations = [lcaWrapper(gainValue, lossValue, *parameters) for _ in
                      range(numSimulationsPerCondition)]
    allValidResponseSimulations = list(filter(filterFunction, allSimulations))
    numValidResponses = len(allValidResponseSimulations)
    allActivations, allModelRTs, allModelResponses = zip(*allValidResponseSimulations)
    modelDataForStakes = np.hstack(
        (np.array(allModelResponses).reshape(-1, 1), np.array(allModelRTs).reshape(-1, 1),
         np.array([gainValue] * numValidResponses).reshape(-1, 1),
         np.array([lossValue] * numValidResponses).reshape(-1, 1)))
    print(np.shape(modelDataForStakes))
    allModelData = np.vstack((allModelData, modelDataForStakes))

allModelData = allModelData[1:, :]
np.savetxt("simulatedData/unequalAttention_repeat.csv", allModelData, delimiter=",")