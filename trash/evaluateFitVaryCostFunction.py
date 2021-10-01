import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "..", "src"))

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pickle

from LCA import PrepareRecurrentWeights, RunLCASimulation, GetValueInputZeroReference#, getValueInput2Attributes2Choices
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

# data = np.genfromtxt("../src/risk_data.csv", delimiter=',')[1:, 1:]
#
# def selectConditionTrials(gain, loss):
#     selectedData = data[data[:, 2] == gain][:]
#     selectedData = selectedData[selectedData[:, 3] == loss][:]
#     return selectedData
#
# def computePData(gain, loss):
#     conditionTrialData = selectConditionTrials(gain, loss)
#     responses = conditionTrialData[:, 0]
#     return np.mean(responses)
#
# ActualRatioAndProbRecord = np.zeros(2)
# for gainValue in allGainValues:
#     for lossValue in allLossValues:
#         gainLossRatio = -1 * gainValue / lossValue
#         probAccept = computePData(gainValue, lossValue)
#         ActualRatioAndProbRecord = np.vstack((ActualRatioAndProbRecord, (gainLossRatio, probAccept)))
#
# ActualRatioAndProbRecord = ActualRatioAndProbRecord[1:, :]
# df = pd.DataFrame(ActualRatioAndProbRecord)
# ratioCombinedDf = df.groupby(0).mean()
# ratioCombinedDfEmpirical = ratioCombinedDf.rename(columns={1: 'Actual Probability'})

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

# def plotPAcceptVsGainLossRatio(params, axForPlot, plotLabel):
#     modelRecord = np.zeros(2)
#     for gainValue in allGainValues:
#         for lossValue in allLossValues:
#             gainLossRatio = -1 * gainValue / lossValue
#             allSimulations = [lcaWrapper(gainValue, lossValue, *params) for _ in
#                               range(numSimulationsPerCondition)]
#             allValidResponseSimulations = list(filter(filterFunction, allSimulations))
#             _, allModelRTs, allModelResponses = zip(*allValidResponseSimulations)
#             P_model = np.mean(allModelResponses)
#             modelRecord = np.vstack((modelRecord, (gainLossRatio, P_model)))
#
#     modelRecord = modelRecord[1:, :]
#     dfSim = pd.DataFrame(modelRecord)
#     ratioCombinedDfSim = dfSim.groupby(0).mean()
#     ratioCombinedDfSim = ratioCombinedDfSim.rename(columns={1: plotLabel})
#     ratioCombinedDfSim.plot(ax=axForPlot, logx=True, linestyle='dashed')

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
    print("number of responses of model: ", np.shape(allResponses)[0])
    modelBinnedResponses = np.array_split(allResponses, numBins)
    modelBinPAccept = [np.mean(binResponses) for binResponses in modelBinnedResponses]

    return modelBinPAccept

term1Value = 250
term2AllValues = [[0.2, 0.5, 0.7, 0.9], [0.02, 0.05, 0.07, 0.09], [0.0002, 0.0005, 0.0007, 0.0009]]

fig, ax = plt.subplots(1, 3)

for subplot in range(len(term2AllValues)):
    ax_ = ax[subplot]
    ax_.plot(empiricalMeanPAcceptForBinnedRT, label='empirical data', marker='o')
    # ax_ = ratioCombinedDfEmpirical.plot(logx=True, ax=ax_)
    for term2 in term2AllValues[subplot]:
        paramFile = open('../data/varyCostFunction/furtherFineTuningCorona/{}_1_{}.pickle'.format(term1Value, term2), 'rb')
        paramRecord = pickle.load(paramFile)
        numIterations = np.shape(paramRecord)[0]
        if numIterations < 4000:
            bestParams = paramRecord[-1, :]
        else:
            bestParams = paramRecord[3999, :]

        print("Term1: ", term1Value, " Term2: ", term2)
        print("Number of iterations: ", numIterations)
        if np.shape(paramRecord)[0] < 3000:
            continue
        print("Best parameters: ", bestParams)

        binnedValuesForPlotting = computeBinnedValuesForPlotting(bestParams)
        # plotPAcceptVsGainLossRatio(bestParams, ax_, str(term2))
        if binnedValuesForPlotting == -1:
            continue

        else:
            print("Model P(Accept): ", np.mean(binnedValuesForPlotting))
            ax_.plot(binnedValuesForPlotting, linestyle='--', label=str(term2), marker='o')
        # ax_.set_title(term1)
        ax_.set_xlabel("bin")
        ax_.set_ylabel("P(Accept)")
        ax_.legend()
        ax_.set_ylim(0, 1)
        print("-----------------------------")

plt.show()



