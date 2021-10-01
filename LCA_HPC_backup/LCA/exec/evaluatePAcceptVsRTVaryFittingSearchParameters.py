import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "..", "src"))

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pickle

from LCA import PrepareRecurrentWeights, RunLCASimulation, GetValueInputZeroReference, getValueInput2Attributes2Choices
from LUT import prepareStdNormalLUT, SampleFromLUT
from sklearn.linear_model import LinearRegression

########################################## PLOT EMPIRICAL DATA #########################################################

data = np.genfromtxt("../src/risk_data.csv", delimiter=',')[1:, :]

allGainValues = list(range(10, 110, 10))
allLossValues = list(range(-100, 0, 10))

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


def computeParticipantMeanPAcceptForBinnedRT(participantIndex, numBins):
    participantAllLogRTResidual = computeParticipantAllLogRTResidual(participantIndex)
    participantResponses = extractParticipantResponses(participantIndex)
    _, sortedResponses = (list(t) for t in zip(*sorted(zip(participantAllLogRTResidual.tolist(), participantResponses.tolist()))))
    binnedResponses = np.array_split(sortedResponses, numBins)
    binPAccept = [np.mean(binResponses) for binResponses in binnedResponses]

    return binPAccept


allParticipantMeanPAcceptForBinnedRT = np.array([computeParticipantMeanPAcceptForBinnedRT(participantIndex, 5) for participantIndex in range(1, 50)])
empiricalMeanPAcceptForBinnedRT = np.mean(allParticipantMeanPAcceptForBinnedRT, 0)
print("P(accept) in empirical data: ", np.mean(empiricalMeanPAcceptForBinnedRT))
plt.plot(empiricalMeanPAcceptForBinnedRT, label='Empirical data', marker='o')


########################################## PLOT SIMULATED DATA #########################################################

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
lca = RunLCASimulation(getValueInput, sampleFromZeroMeanLUT, prepareRecurrentWeights, maxTimeSteps, deltaT)

attributeProbabilities = (0.5, 0.5)
startingActivation = (0, 0)

getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
getAllThresholds = lambda threshold: [threshold]*numAccumulators
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold1, threshold2, constantInput: \
    lca(attributeProbabilities, getChoiceAttributes(gain, loss), startingActivation, decay, competition, constantInput,
        noiseStdDev, nonDecisionTime, [threshold1, threshold2])


numSimulationsPerCondition = 100

# function to return the binned values for plotting
def computeBinnedValuesForPlotting(params):
    stakesValues = np.zeros(2)
    modelLogRT = np.zeros(1)
    modelResponses = np.zeros(1)

    for gainValue in allGainValues:
        for lossValue in allLossValues:
            allSimulations = [lcaWrapper(gainValue, lossValue, *params) for _ in
                              range(numSimulationsPerCondition)]
            allValidResponseSimulations = list(filter(filterFunction, allSimulations))
            numValidResponses = len(allValidResponseSimulations)
            # print("Number of valid responses: ", len(allValidResponseSimulations))
            if numValidResponses < numSimulationsPerCondition/2:
                print("TOO FEW RECEIVED")
                print("Number of valid responses: ", len(allValidResponseSimulations))
                return -1
            _, allModelRTs, allModelResponses = zip(*allValidResponseSimulations)
            allModelLogRTs = np.reshape(np.log(allModelRTs), (-1, 1))
            allModelResponses = np.reshape(allModelResponses, (-1, 1))
            stakes = np.hstack(
                (np.full((numValidResponses, 1), gainValue), np.full((numValidResponses, 1), lossValue)))
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

    allRTs, allResponses = (list(t) for t in zip(*sorted(zip(allLogRTResidual.tolist(), modelResponses.tolist()))))

    numBins = 5
    modelBinnedResponses = np.array_split(allResponses, numBins)
    modelBinPAccept = [np.mean(binResponses) for binResponses in modelBinnedResponses]

    return modelBinPAccept

# all parameters forced to be positive
file = open("data/fitLCAAllParametersForcedToBePositiveAndNonZeroStartingPoint.pickle", "rb")
record = pickle.load(file)
finalParams = record[-1, :]
toBePlotted = computeBinnedValuesForPlotting(finalParams)
print("P(Accept) in All positive: ", np.mean(toBePlotted))
plt.plot(toBePlotted, label='Only positive parameters are allowed', linestyle='--', marker='o')

# starting point is away from zero
file = open("data/fitLCANonZeroStartingPoint.pickle", "rb")
record = pickle.load(file)
finalParams = record[-1, :]
toBePlotted = computeBinnedValuesForPlotting(finalParams)
print("P(Accept) in starting point away from zero: ", np.mean(toBePlotted))
plt.plot(toBePlotted, label='Starting point is away from zero', linestyle='--', marker='o')


# best fit
finalParams = [ -5.14274066, 2.41055527, 46.18926038, 1.28197494, 70.1362561, 98.49542825, 120.06259734]
toBePlotted = computeBinnedValuesForPlotting(finalParams)
print("P(Accept) in best fit: ", np.mean(toBePlotted))
plt.plot(toBePlotted, label='Best fit', linestyle='--', marker='o')


# set-up the LCA
identityUtilityFunction = lambda x: x
# getValueInput = GetValueInputZeroReference(identityUtilityFunction)

maxTimeSteps = 750
deltaT = 0.01
prepareRecurrentWeights = PrepareRecurrentWeights(numAccumulators)
lca = RunLCASimulation(getValueInput2Attributes2Choices, sampleFromZeroMeanLUT, prepareRecurrentWeights, maxTimeSteps, deltaT)

attributeProbabilities = (0.5, 0.5)
startingActivation = (0, 0)

getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
getAllThresholds = lambda threshold: [threshold]*numAccumulators
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold1, threshold2, constantInput: \
    lca(attributeProbabilities, getChoiceAttributes(gain, loss), startingActivation, decay, competition, constantInput,
        noiseStdDev, nonDecisionTime, [threshold1, threshold2])


# function to return the binned values for plotting
def computeBinnedValuesForPlotting(params):
    stakesValues = np.zeros(2)
    modelLogRT = np.zeros(1)
    modelResponses = np.zeros(1)

    for gainValue in allGainValues:
        for lossValue in allLossValues:
            allSimulations = [lcaWrapper(gainValue, lossValue, *params) for _ in
                              range(numSimulationsPerCondition)]
            allValidResponseSimulations = list(filter(filterFunction, allSimulations))
            numValidResponses = len(allValidResponseSimulations)
            # print("Number of valid responses: ", len(allValidResponseSimulations))
            if numValidResponses < numSimulationsPerCondition/2:
                print("TOO FEW RECEIVED")
                print("Number of valid responses: ", len(allValidResponseSimulations))
                return -1
            _, allModelRTs, allModelResponses = zip(*allValidResponseSimulations)
            allModelLogRTs = np.reshape(np.log(allModelRTs), (-1, 1))
            allModelResponses = np.reshape(allModelResponses, (-1, 1))
            stakes = np.hstack(
                (np.full((numValidResponses, 1), gainValue), np.full((numValidResponses, 1), lossValue)))
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

    allRTs, allResponses = (list(t) for t in zip(*sorted(zip(modelLogRT.tolist(), modelResponses.tolist()))))
    numBins = 5
    modelBinnedResponses = np.array_split(allResponses, numBins)
    modelBinPAccept = [np.mean(binResponses) for binResponses in modelBinnedResponses]

    return modelBinPAccept


# differential LCA
file = open("data/fitDifferentialLCA.pickle", "rb")
record = pickle.load(file)
finalParams = record[-1, :]
toBePlotted = computeBinnedValuesForPlotting(finalParams)
print("P(Accept) in differential LCA: ", np.mean(toBePlotted))
plt.plot(toBePlotted, label='DifferentialLCA', linestyle='--', marker='o')

plt.legend()
plt.show()
