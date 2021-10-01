import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "..", "..", "src"))

import numpy as np
from matplotlib import pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import seaborn as sns

from LCA import PrepareRecurrentWeights, RunLCASimulation, GetValueInputZeroReference#, getValueInput2Attributes2Choices
from LUT import prepareStdNormalLUT, SampleFromLUT

data = np.genfromtxt("data_preprocessed.csv", delimiter=',')[1:, 2:]

# def computeParticipantAllLogRTResidual(participantIndex):
#     trialIndicesOfParticipant = np.asarray(np.where(data[:, 0] == participantIndex))[0]
#     participantData = data[trialIndicesOfParticipant]
#     participantAllRT = participantData[:, 2]
#     participantAllLogRT = np.log(participantAllRT).reshape(-1, 1)
#     participantAllGainLoss = participantData[:, 3:5]
#
#     regressor = LinearRegression()
#
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
# allParticipantMeanPAcceptForBinnedRT = np.array([computeParticipantMeanPAcceptForBinnedRT(participantIndex, 5) for participantIndex in range(1, 40)])
# # print(allParticipantMeanPAcceptForBinnedRT)
# meanPAcceptForBinnedRT = np.mean(allParticipantMeanPAcceptForBinnedRT, 0)

# i = 0
# for row in range(5):
#     for column in range(8):
#         if i >= 39:
#             break
#         ax = plt.subplot2grid(shape=(5, 8), loc=(row, column))
#         ax.plot(allParticipantMeanPAcceptForBinnedRT[i], marker='o')
#         ax.set_ylim(0, 1)
#         i += 1
#
# plt.show()


# plt.plot(meanPAcceptForBinnedRT, marker='o', label='Observed Responses')
plt.ylabel("P(accept)")
plt.xlabel("Choice factor adjusted RT")
plt.ylim(0, 1)


allLogRT = np.log(data[:, 1]).reshape(-1, 1)
allGainLoss = data[:, 2:]
regressor = LinearRegression()
regressor.fit(allGainLoss, allLogRT)
allPredictedLogRT = regressor.predict(allGainLoss)
allLogRTResidual = allLogRT - allPredictedLogRT

responses = data[:, 0]
_, sortedResponses = (list(t) for t in zip(*sorted(zip(allLogRTResidual.tolist(), responses.tolist()))))

binnedResponses = np.array_split(sortedResponses, 5)
binPAccept = [np.mean(binResponses) for binResponses in binnedResponses]

plt.plot(binPAccept, marker='o', label='All data clubbed together')

plt.show()
exit()



# data = np.genfromtxt("/Users/nishadsinghi/undergrad-project-loss-aversion/newDataset/dataZeroSureDDMSim.csv", delimiter=',')[1:, :]
# allLogRT = np.log(data[:, 3]).reshape(-1, 1)
# allGainLoss = data[:, 4:]
# regressor = LinearRegression()
# regressor.fit(allGainLoss, allLogRT)
# allPredictedLogRT = regressor.predict(allGainLoss)
# allLogRTResidual = allLogRT - allPredictedLogRT
#
# responses = data[:, 2]
# _, sortedResponses = (list(t) for t in zip(*sorted(zip(allLogRTResidual.tolist(), responses.tolist()))))
#
# binnedResponses = np.array_split(sortedResponses, 5)
# binPAccept = [np.mean(binResponses) for binResponses in binnedResponses]

# plt.plot(binPAccept, marker='o', label='DDM', linestyle='--')








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
# lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, startingBias, constantInput: \
#     lca(attributeProbabilities, getChoiceAttributes(gain, loss), getStartingActivation(startingBias, threshold), decay, competition, constantInput,
#         noiseStdDev, nonDecisionTime, getAllThresholds(threshold))


lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, startingBias, lossWeight, attribute1Prob, constantInput1, constantInput2: \
        lca((attribute1Prob, 1-attribute1Prob), getChoiceAttributes(gain, lossWeight*loss), getStartingActivation(startingBias, threshold), decay, competition, (constantInput1, constantInput2),
            noiseStdDev, nonDecisionTime, getAllThresholds(threshold))



allStakes = np.unique(data[:, -2:], axis=0)

def plotPAcceptVsAdjustedRT(params, label):
    for metropolisRun in range(1, 2):
        numSimulationsPerCondition = 100
        print("PARAMS = ", params)

        stakesValues = np.zeros(2)
        modelLogRT = np.zeros(1)
        modelResponses = np.zeros(1)

        for stakes in allStakes:
            print('stakes: ', stakes)
            gainValue, lossValue = stakes
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
        # plt.savefig('fig.png')
        # plt.close()


plotPAcceptVsAdjustedRT([3.65181484,  4.42065135, 26.57513611,  0.27369398,  6.8341132, -0.23144024,  3.30918601, 0.5, 24.18245027, 27.05303325], "LCA")

plt.ylim(0, 1)
plt.legend()
# plt.show()
plt.savefig('fig.png')
plt.close()