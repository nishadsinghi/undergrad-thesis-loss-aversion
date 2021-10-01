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


def computeParticipantMeanPAcceptForBinnedRT(participantIndex, numBins):
    participantAllLogRTResidual = computeParticipantAllLogRTResidual(participantIndex)
    participantResponses = extractParticipantResponses(participantIndex)
    _, sortedResponses = (list(t) for t in zip(*sorted(zip(participantAllLogRTResidual.tolist(), participantResponses.tolist()))))
    binnedResponses = np.array_split(sortedResponses, numBins)
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
def lossAvertUtilityFunction(lossWeight, x):
    if x > 0:
        return x
    else:
        return lossWeight*x

getUtilityFunction = lambda lossWeight: lambda x: lossAvertUtilityFunction(lossWeight, x)
getValueInputWrapper = lambda lossWeight: GetValueInputZeroReference(getUtilityFunction(lossWeight))

maxTimeSteps = 750
deltaT = 0.01
prepareRecurrentWeights = PrepareRecurrentWeights(numAccumulators)

startingActivation = (0, 0)

getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
getAllThresholds = lambda threshold: [threshold]*numAccumulators
getLCA = lambda lossWeight: RunLCASimulation(getValueInputWrapper(lossWeight), sampleFromZeroMeanLUT,
                                             prepareRecurrentWeights, maxTimeSteps, deltaT)

getAllAttributeProbabilities = lambda attribute1Prob: (attribute1Prob, 1-attribute1Prob)

def getLCAWrapperForLossWeight(lossWeight):
    lca = getLCA(lossWeight)
    lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold1, threshold2, constantInput, attribute1Prob: \
        lca(getAllAttributeProbabilities(attribute1Prob), getChoiceAttributes(gain, loss), startingActivation, decay, competition, constantInput,
            noiseStdDev, nonDecisionTime, (threshold1, threshold2))

    return lcaWrapper


# function to take model params and compute values to be plotted
def computeModelBinPAccept(lcaWrapper, params):
    numSimulationsPerCondition = 150
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

    allRTs, allResponses = (list(t) for t in zip(*sorted(zip(modelLogRT.tolist(), modelResponses.tolist()))))
    numBins = 5
    modelBinnedResponses = np.array_split(allResponses, numBins)
    modelBinPAccept = [np.mean(binResponses) for binResponses in modelBinnedResponses]

    return modelBinPAccept


def getParamsAt4kFromFilePath(path):
    readfile = open(path, "rb")
    params = pickle.load(readfile)
    return (params[-1, :])#[0]

# no assumptions relaxed
print("no assumptions relaxed")
path1 = '../data/varyAssumptions/noAssumptionRelaxed.pickle'
params1 = getParamsAt4kFromFilePath(path1)
lcaWrapper1 = getLCAWrapperForLossWeight(1)
finalParams1 = (params1[0], params1[1], params1[2], params1[3], params1[4], params1[4], params1[5], 0.5)
plt.plot(computeModelBinPAccept(lcaWrapper1, finalParams1), marker='o', linestyle='dashed', label='Baseline')

# loss aversion
print("loss aversion")
path2 = '../data/varyAssumptions/flexibleLossWeight.pickle'
params2 = getParamsAt4kFromFilePath(path2)
lossWeight = params2[-1]
lcaWrapper2 = getLCAWrapperForLossWeight(lossWeight)
finalParams2 = (params1[0], params1[1], params1[2], params1[3], params1[4], params1[4], params1[5], 0.5)
plt.plot(computeModelBinPAccept(lcaWrapper2, finalParams2), marker='o', linestyle='dashed', label='Loss Aversion')

# flexible focus
print("flexible focus")
path3 = '../data/varyAssumptions/variableProbabilityOfFocus.pickle'
params3 = getParamsAt4kFromFilePath(path3)
lcaWrapper3 = getLCAWrapperForLossWeight(1)
finalParams3 = (params3[0], params3[1], params3[2], params3[3], params3[4], params3[4], params3[5], params3[6])
plt.plot(computeModelBinPAccept(lcaWrapper3, finalParams3), marker='o', linestyle='dashed', label='Asymmetric focus on loss, gain')

# different thresholds
print("different thresholds")
path4 = '../data/varyCostFunction/furtherFineTuningCorona/250_1_0.7.pickle'
params4 = getParamsAt4kFromFilePath(path4)
lcaWrapper4 = getLCAWrapperForLossWeight(1)
#decay, competition, noiseStdDev, nonDecisionTime, threshold1, threshold2, constantInput:
#decay, competition, noiseStdDev, nonDecisionTime, threshold1, threshold2, constantInput, attribute1Prob:
finalParams4 = np.reshape(np.hstack((params4, 0.5)), (1, -1))[0]
print(finalParams4)
plt.plot(computeModelBinPAccept(lcaWrapper4, finalParams4), marker='o', linestyle='dashed', label='different thresholds')

plt.legend()
plt.show()