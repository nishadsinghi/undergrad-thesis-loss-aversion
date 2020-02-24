import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "..", "src"))

import numpy as np
from matplotlib import pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

from LCA import PrepareRecurrentWeights, RunLCASimulation, GetValueInputZeroReference, getValueInput2Attributes2Choices
from LUT import prepareStdNormalLUT, SampleFromLUT

data = np.genfromtxt("../src/risk_data.csv", delimiter=',')[1:, :]

def computeParticipantAllLogRTResidual(participantIndex):
    trialIndicesOfParticipant = np.asarray(np.where(data[:, 0] == participantIndex))[0]
    participantData = data[trialIndicesOfParticipant]
    participantAllRT = participantData[:, 2]
    participantAllLogRT = np.log(participantAllRT).reshape(-1, 1)
    participantAllGainLoss = participantData[:, 3:5]

    regressor = LinearRegression()
    regressor.fit(participantAllGainLoss, participantAllLogRT)

    participantAllPredictedLogRT = regressor.predict(participantAllGainLoss)
    participantAllLogRTResidual = participantAllLogRT - participantAllPredictedLogRT

    return np.ndarray.flatten(participantAllLogRTResidual)


def extractParticipantResponses(participantIndex):
    trialIndicesOfParticipant = np.asarray(np.where(data[:, 0] == participantIndex))[0]
    participantData = data[trialIndicesOfParticipant]
    participantResponses = participantData[:, 1]

    return participantResponses


def prepareParticipantDictPAcceptLogRTResidual(participantIndex):
    participantAllLogRTResidual = computeParticipantAllLogRTResidual(participantIndex)
    participantResponses = extractParticipantResponses(participantIndex)

    dictPAcceptLogRTResidual = {residual: pAccept for pAccept, residual in
                                zip(participantResponses, participantAllLogRTResidual)}

    return dictPAcceptLogRTResidual


sortDictUsingKeys = lambda dictionary: {key: value for key, value in sorted(dictionary.items(), key=lambda item: item[0])}
prepareParticipantSortedDict = lambda participantIndex: sortDictUsingKeys(prepareParticipantDictPAcceptLogRTResidual(participantIndex))


def computeParticipantMeanPAcceptForBinnedRT(participantIndex, numBins):
    participantSortedDict = prepareParticipantSortedDict(participantIndex)
    allResponses = np.array(list(participantSortedDict.values()))
    binnedResponses = np.array_split(allResponses, numBins)
    binPAccept = [np.mean(binResponses) for binResponses in binnedResponses]

    return binPAccept


allParticipantMeanPAcceptForBinnedRT = np.array([computeParticipantMeanPAcceptForBinnedRT(participantIndex, 5) for participantIndex in range(1, 50)])
meanPAcceptForBinnedRT = np.mean(allParticipantMeanPAcceptForBinnedRT, 0)

plt.plot(meanPAcceptForBinnedRT, marker='o', label='Observed Responses')
plt.ylabel("P(accept)")
plt.xlabel("Choice factor adjusted RT")
plt.ylim(0, 1)


                                                    # SIMULATION

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
getValueInput = GetValueInputZeroReference(identityUtilityFunction)

maxTimeSteps = 750
deltaT = 0.01
prepareRecurrentWeights = PrepareRecurrentWeights(numAccumulators)
# lca = RunLCASimulation(getValueInput, sampleFromZeroMeanLUT, prepareRecurrentWeights, maxTimeSteps, deltaT)
lca = RunLCASimulation(getValueInput2Attributes2Choices, sampleFromZeroMeanLUT, prepareRecurrentWeights, maxTimeSteps, deltaT)

attributeProbabilities = (0.5, 0.5)
startingActivation = (0, 0)

getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
getAllThresholds = lambda threshold: [threshold]*numAccumulators
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold1, threshold2, constantInput: \
    lca(attributeProbabilities, getChoiceAttributes(gain, loss), startingActivation, decay, competition, constantInput,
        noiseStdDev, nonDecisionTime, [threshold1, threshold2])

for metropolisRun in range(1, 2):
    numSimulationsPerCondition = 100
    paramFile = open('../data/differentialLCAUnequalThresholds.pickle'.format(metropolisRun), 'rb')
    params = pickle.load(paramFile)
    params = (params[-1:, :])[0]
    print("PARAMS = ", params)
    params = [-2.83570389, 5.449, 62.15105359, 0.38892113, 24.95, 48.04508622, 4.10653348]

    allGainValues = list(range(10, 110, 10))
    allLossValues = list(range(-100, 0, 10))

    stakesValues = np.zeros(2)
    modelLogRT = np.zeros(1)
    modelResponses = np.zeros(1)

    for gainValue in allGainValues:
        for lossValue in allLossValues:
            allSimulations = [lcaWrapper(gainValue, lossValue, *params) for _ in
                              range(numSimulationsPerCondition)]
            allValidResponseSimulations = list(filter(filterFunction, allSimulations))
            _, allModelRTs, allModelResponses = zip(*allValidResponseSimulations)
            allModelLogRTs = np.reshape(np.log(allModelRTs), (-1, 1))
            allModelResponses = np.reshape(allModelResponses, (-1, 1))
            stakes = np.hstack((np.full((numSimulationsPerCondition, 1), gainValue), np.full((numSimulationsPerCondition, 1), lossValue)))
            stakesValues = np.vstack((stakesValues, stakes))
            modelLogRT = np.vstack((modelLogRT, allModelLogRTs))
            modelResponses = np.vstack((modelResponses, allModelResponses))

    stakesValues = stakesValues[1:, :]

    modelLogRT = modelLogRT[1:, :]
    modelResponses = modelResponses[1:, :]

    regressor = LinearRegression()
    regressor.fit(stakesValues, modelLogRT)

    predictedLogRT = regressor.predict(stakesValues)
    allLogRTResidual = modelLogRT - predictedLogRT

    print("Proportion of gambles accepted by the model: ", np.mean(modelResponses)*100)

    dict = {residual: response for residual, response in zip(np.ndarray.flatten(allLogRTResidual), np.ndarray.flatten(modelResponses))}
    sortedDict = {k: v for k, v in sorted(dict.items(), key=lambda item: item[0])}
    allResponses = np.array(list(sortedDict.values()))
    numBins = 5
    modelBinnedResponses = np.array_split(allResponses, numBins)
    modelBinPAccept = [np.mean(binResponses) for binResponses in modelBinnedResponses]
    plt.plot(modelBinPAccept, marker='o', linestyle='dashed', label='LCA Simulation {}'.format(metropolisRun))

plt.legend()
plt.show()