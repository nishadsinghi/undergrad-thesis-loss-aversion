import sys
import os

DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "..", "..", "..", "src"))

import numpy as np

from LCA import PrepareRecurrentWeights, RunLCASimulation, \
    GetValueInputZeroReference
from LUT import prepareStdNormalLUT, SampleFromLUT


experimentalData = np.genfromtxt("risk_data_cleaned.csv", delimiter=',')[1:, 1:]
allStakes = np.unique(experimentalData[:, -2:], axis=0)


# LCA
def filterFunction(tup):
    if tup[-1] != -1:
        return True
    else:
        return False

numSimulationsPerCondition = 150

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


parameters = [1.55893187e+00, 2.59678066e+00, 1.36329544e+02, 1.25511453e-01, 4.71199432e+01, 1.11088120e+02, 3.57679855e-01]

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
np.savetxt("unequalAttentionSimulatedData.csv", allModelData, delimiter=",")