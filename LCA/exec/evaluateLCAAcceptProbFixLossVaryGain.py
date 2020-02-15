import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "..", "src"))

from LCA import PrepareRecurrentWeights, RunLCASimulation, GetValueInput
from LUT import prepareStdNormalLUT, SampleFromLUT

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle

def filterFunction(tup):
    if tup[-1] != -1:
        return True
    else:
        return False

LUTInterval = 0.0001
numAccumulators = 2
stdNormalLUT = prepareStdNormalLUT(LUTInterval)
sampleFromLUT = SampleFromLUT(stdNormalLUT)
sampleFromZeroMeanLUT = lambda stdDev: sampleFromLUT(0, stdDev, numAccumulators)

# set-up the LCA
identityUtilityFunction = lambda x: x
getValueInput = GetValueInput(identityUtilityFunction)

maxTimeSteps = 750
deltaT = 0.01
prepareRecurrentWeights = PrepareRecurrentWeights(numAccumulators)
lca = RunLCASimulation(getValueInput, sampleFromZeroMeanLUT, prepareRecurrentWeights, maxTimeSteps, deltaT)

attributeProbabilities = (0.5, 0.5)
startingActivation = (0, 0)

getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
getAllThresholds = lambda threshold: [threshold]*numAccumulators
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, constantInput: \
    lca(attributeProbabilities, getChoiceAttributes(gain, loss), startingActivation, decay, competition, constantInput,
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

numSimulationsPerCondition = 200
paramFile = open('../data/newAttempt.pickle', 'rb')
params = pickle.load(paramFile)
params = (params[-1:, :])[0]
# params = [1, 1, 10, 0, 5, 25]
print("PARAMS = ", params)

# lossValue = -20
# allGainValues = range(10, 110, 10)
#
# allPModel = []
# for gainValue in allGainValues:
#     allSimulations = [lcaWrapper(gainValue, lossValue, *params) for _ in
#                       range(numSimulationsPerCondition)]
#     allValidResponseSimulations = list(filter(filterFunction, allSimulations))
#     _, allModelRTs, allModelResponses = zip(*allValidResponseSimulations)
#     P_model = np.mean(allModelResponses)
#     allPModel.append(P_model)
#
# plt.plot(list(allGainValues), allPModel)
# plt.show()