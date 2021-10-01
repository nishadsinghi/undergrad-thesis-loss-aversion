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

from LCA import PrepareRecurrentWeights, RunLCASimulation, GetValueInputZeroReference#, getValueInput2Attributes2Choices
from LUT import prepareStdNormalLUT, SampleFromLUT

data = np.genfromtxt("../src/risk_data.csv", delimiter=',')[1:, :]

allGainValues = list(range(10, 110, 10))
allLossValues = list(range(-100, 0, 10))


# def computeParticipantAllLogRTResidual(participantIndex):
#     trialIndicesOfParticipant = np.asarray(np.where(data[:, 0] == participantIndex))[0]
#     participantData = data[trialIndicesOfParticipant]
#     participantAllRT = participantData[:, 2]
#     participantAllLogRT = np.log(participantAllRT).reshape(-1, 1)
#     participantAllGainLoss = participantData[:, 3:5]
#
#     regressor = LinearRegression()
#     regressor.fit(participantAllGainLoss, participantAllLogRT)
#
#     participantAllPredictedLogRT = regressor.predict(participantAllGainLoss)
#     participantAllLogRTResidual = participantAllLogRT - participantAllPredictedLogRT
#
#     return np.ndarray.flatten(participantAllLogRTResidual)
#
#
# def extractParticipantResponses(participantIndex):
#     trialIndicesOfParticipant = np.asarray(np.where(data[:, 0] == participantIndex))[0]
#     participantData = data[trialIndicesOfParticipant]
#     participantResponses = participantData[:, 1]
#
#     return participantResponses
#
#
# def computeParticipantMeanPAcceptForBinnedRT(participantIndex, numBins):
#     participantAllLogRTResidual = computeParticipantAllLogRTResidual(participantIndex)
#     participantResponses = extractParticipantResponses(participantIndex)
#     _, sortedResponses = (list(t) for t in zip(*sorted(zip(participantAllLogRTResidual.tolist(), participantResponses.tolist()))))
#     binnedResponses = np.array_split(sortedResponses, numBins)
#     binPAccept = [np.mean(binResponses) for binResponses in binnedResponses]
#
#     return binPAccept
#
#
# allParticipantMeanPAcceptForBinnedRT = np.array([computeParticipantMeanPAcceptForBinnedRT(participantIndex, 5) for participantIndex in range(1, 50)])
# meanPAcceptForBinnedRT = np.mean(allParticipantMeanPAcceptForBinnedRT, 0)
#
# plt.plot(meanPAcceptForBinnedRT, marker='o', label='Observed Responses')
# plt.ylabel("P(accept)")
# plt.xlabel("Choice factor adjusted RT")
# plt.ylim(0, 1)



allLogRT = np.log(data[:, 2]).reshape(-1, 1)
allGainLoss = data[:, 3:5]
regressor = LinearRegression()
regressor.fit(allGainLoss, allLogRT)
allPredictedLogRT = regressor.predict(allGainLoss)
allLogRTResidual = allLogRT - allPredictedLogRT

responses = data[:, 1]
_, sortedResponses = (list(t) for t in zip(*sorted(zip(allLogRTResidual.tolist(), responses.tolist()))))

binnedResponses = np.array_split(sortedResponses, 5)
binPAccept = [np.mean(binResponses) for binResponses in binnedResponses]

plt.plot(binPAccept, marker='o', label='All data clubbed together')



data = np.genfromtxt("/Users/nishadsinghi/undergrad-project-loss-aversion/DDM/allParticipantDataClubbedSimulatedData.csv", delimiter=',')[1:, :]
allLogRT = np.log(data[:, 2]).reshape(-1, 1)
allGainLoss = data[:, 3:5]
regressor = LinearRegression()
regressor.fit(allGainLoss, allLogRT)
allPredictedLogRT = regressor.predict(allGainLoss)
allLogRTResidual = allLogRT - allPredictedLogRT

responses = data[:, 1]
_, sortedResponses = (list(t) for t in zip(*sorted(zip(allLogRTResidual.tolist(), responses.tolist()))))

binnedResponses = np.array_split(sortedResponses, 5)
binPAccept = [np.mean(binResponses) for binResponses in binnedResponses]

plt.plot(binPAccept, marker='o', label='DDM', linestyle='--')


# data = np.genfromtxt("../src/simulatedData.csv", delimiter=',')[1:, :]
#
# allLogRT = np.log(data[:, 2]).reshape(-1, 1)
# allGainLoss = data[:, 3:5]
# regressor = LinearRegression()
# regressor.fit(allGainLoss, allLogRT)
# allPredictedLogRT = regressor.predict(allGainLoss)
# allLogRTResidual = allLogRT - allPredictedLogRT
#
# responses = data[:, 1]
# _, sortedResponses = (list(t) for t in zip(*sorted(zip(allLogRTResidual.tolist(), responses.tolist()))))
#
# binnedResponses = np.array_split(sortedResponses, 5)
# binPAccept = [np.mean(binResponses) for binResponses in binnedResponses]
#
# plt.plot(binPAccept, marker='o', label='simulated data clubbed together', linestyle='--')
# plt.ylabel("P(accept)")
# plt.xlabel("Choice factor adjusted RT")
# plt.ylim(0, 1)
# plt.legend()
# plt.show()


# SIMULATION

def filterFunction(tup):
    if tup[-1] != -1:
        return True
    else:
        return False

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
# lca = RunLCASimulation(getValueInput2Attributes2Choices, sampleFromZeroMeanLUT, prepareRecurrentWeights, maxTimeSteps, deltaT)

attributeProbabilities = (0.5, 0.5)
# startingActivation = (0, 0)
#
# getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
# lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold1, threshold2, constantInput: \
#     lca(attributeProbabilities, getChoiceAttributes(gain, loss), startingActivation, decay, competition, constantInput,
#         noiseStdDev, nonDecisionTime, [threshold1, threshold2])

def getStartingActivation(startingBias, threshold):
    if startingBias < 0:
        return [-1*startingBias*threshold, 0]
    else:
        return [0, startingBias*threshold]

getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
getAllThresholds = lambda threshold: [threshold]*numAccumulators
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, startingBias, constantInput: \
    lca(attributeProbabilities, getChoiceAttributes(gain, loss), getStartingActivation(startingBias, threshold), decay, competition, constantInput,
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))


def plotPAcceptVsAdjustedRT(params, label):
    for metropolisRun in range(1, 2):
        numSimulationsPerCondition = 150
        print("PARAMS = ", params)

        stakesValues = np.zeros(2)
        modelLogRT = np.zeros(1)
        modelResponses = np.zeros(1)

        for gainValue in allGainValues:
            for lossValue in allLossValues:
                allSimulations = [lcaWrapper(gainValue, lossValue, *params) for _ in
                                  range(numSimulationsPerCondition)]
                allValidResponseSimulations = list(filter(filterFunction, allSimulations))
                _, allModelRTs, allModelResponses = zip(*allValidResponseSimulations)
                numValidResponses = np.shape(allModelRTs)[0]
                allModelLogRTs = np.reshape(np.log(allModelRTs), (-1, 1))
                allModelResponses = np.reshape(allModelResponses, (-1, 1))
                stakes = np.hstack((np.full((numValidResponses, 1), gainValue), np.full((numValidResponses, 1), lossValue)))
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

        allRTs, allResponses = (list(t) for t in zip(*sorted(zip(allLogRTResidual.tolist(), modelResponses.tolist()))))
        numBins = 5
        modelBinnedResponses = np.array_split(allResponses, numBins)
        modelBinPAccept = [np.mean(binResponses) for binResponses in modelBinnedResponses]
        plt.plot(modelBinPAccept, marker='o', linestyle='dashed', label=label)


plotPAcceptVsAdjustedRT([1.21160582e+00,  3.43006180e+00,  9.62417214e+01,  1.97180120e-01, 3.21544934e+01, -2.48226397e-01,  5.68393017e+01], "LCA")

plt.ylim(0, 1)
plt.legend()
plt.show()