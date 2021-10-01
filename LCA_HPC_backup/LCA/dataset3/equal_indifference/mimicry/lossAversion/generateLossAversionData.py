import sys
import os

DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "..", "..", "..", "..", "src"))

import numpy as np

from LCA import PrepareRecurrentWeights, RunLCASimulation, \
    GetValueInputZeroReference
from LUT import prepareStdNormalLUT, SampleFromLUT


experimentalData = np.genfromtxt("data_preprocessed_250_zScore3.csv", delimiter=',')[:, 1:]
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

maxTimeSteps = 300
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


parameters = [ 3.3304184,   4.76703779, 26.95523898,  0.45817259, 15.71979804, 64.94022609, 1.52706622]


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