import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "..", "src"))

import numpy as np
from matplotlib import pyplot as plt
import pickle

from LCA import PrepareRecurrentWeights, RunLCASimulation, GetValueInputZeroReference, getValueInput2Attributes2Choices
from LUT import prepareStdNormalLUT, SampleFromLUT
from sklearn.linear_model import LinearRegression

########################################## PLOT EMPIRICAL DATA #########################################################

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
empiricalMeanPAcceptForBinnedRT = np.mean(allParticipantMeanPAcceptForBinnedRT, 0)

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

# function to return the binned values for plotting
numSimulationsPerCondition = 50

allGainValues = list(range(10, 110, 10))
allLossValues = list(range(-100, 0, 10))

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

    dict = {residual: response for residual, response in
            zip(np.ndarray.flatten(allLogRTResidual), np.ndarray.flatten(modelResponses))}
    sortedDict = {k: v for k, v in sorted(dict.items(), key=lambda item: item[0])}
    allResponses = np.array(list(sortedDict.values()))
    numBins = 5
    modelBinnedResponses = np.array_split(allResponses, numBins)
    modelBinPAccept = [np.mean(binResponses) for binResponses in modelBinnedResponses]

    return modelBinPAccept

term1AllValues = [1, 25, 250, 400]
term2AllValues = [2, 0.02, 0.0002]

fig, ax = plt.subplots(2, 2)

axisLocationForTerm1 = {1: (0, 0), 25: (0, 1), 250: (1, 0), 400: (1, 1)}

for term1 in term1AllValues:
    ax_ = ax[axisLocationForTerm1[term1][0]][axisLocationForTerm1[term1][1]]
    ax_.plot(empiricalMeanPAcceptForBinnedRT, label='empirical data', marker='o')
    for term2 in term2AllValues:
        print(term1, term2)
        if term1 == 250 and term2 == 2:
            continue
        paramFile = open('../data/varyCostFunction/{}_1_{}.pickle'.format(term1, term2), 'rb')
        paramRecord = pickle.load(paramFile)
        bestParams = paramRecord[-1, :]

        print("Number of iterations: ", np.shape(paramRecord)[0])
        print("Best parameters: ", bestParams)

        binnedValuesForPlotting = computeBinnedValuesForPlotting(bestParams)
        if binnedValuesForPlotting == -1:
            continue
        else:
            ax_.plot(binnedValuesForPlotting, linestyle='--', label=str(term2), marker='o')
            ax_.set_title(term1)
            ax_.set_xlabel("bin")
            ax_.set_ylabel("P(Accept)")
            ax_.legend()
        print("-----------------------------")

plt.show()



