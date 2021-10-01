import sys
import os

DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "..", "..", "..", "src"))

import numpy as np

from LCA import PrepareRecurrentWeights, RunLCASimulation, \
    GetValueInputZeroReference
from LUT import prepareStdNormalLUT, SampleFromLUT


experimentalData = np.genfromtxt("risk_data_cleaned.csv", delimiter=',')[:, 1:]
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


lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, constantInput, lossWeight: \
    lca((0.5, 0.5), getChoiceAttributes(gain, lossWeight*loss), getStartingActivation(0, threshold),
        decay, competition, constantInput, noiseStdDev, nonDecisionTime, getAllThresholds(threshold))


parameters = [2.97553172,   4.76614439, 148.09895125,   0.19744745, 33.2914632 , 101.92218251,   2.17809482]

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
np.savetxt("lossAversionSimulatedData.csv", allModelData, delimiter=",")