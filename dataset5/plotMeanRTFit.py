import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "..", "src"))

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle
import matplotlib.gridspec as gridspec

data = np.genfromtxt("data_preprocessed.csv", delimiter=',')[1:, 2:]
allStakes = np.unique(data[:, -2:], axis=0)
NUMBINS = 5

def filterFunction(tup):
    if tup[-1] != -1:
        return True
    else:
        return False

gs = gridspec.GridSpec(2, 2)
separateLabels = ['Reject', 'Accept']
plt.figure()

def plotData(data, label, linestyle):
    for choice in range(2):
        ax = plt.subplot(gs[0, choice])
        ax.set_xlabel("Gain/Loss")
        ax.set_ylabel("RT")
        ax.set_ylim(0.75, 2.25)
        ax.set_title(separateLabels[choice])
        trialsForThisChoice = data[data[:, 0] == choice]
        allRatiosForThisChoice = -1*trialsForThisChoice[:, -2]/trialsForThisChoice[:, -1]
        allReactionTimesForThisChoice = trialsForThisChoice[:, 1].flatten()
        allRatiosForThisChoiceSorted, allRTsSortedByRatio = (list(t) for t in zip(*sorted(zip(allRatiosForThisChoice.tolist(), allReactionTimesForThisChoice.tolist()))))
        RTsBinned = np.array_split(allRTsSortedByRatio, NUMBINS)
        meanRTs = [np.mean(_) for _ in RTsBinned]
        plt.plot(meanRTs, label=label, linestyle=linestyle, marker='o')

    ax = plt.subplot(gs[1, :])
    allRatios = -1 * data[:, -2] / data[:, -1]
    allRTs = data[:, 1].flatten()
    allRatiosSorted, allRTsSortedByRatio = (list(t) for t in zip(*sorted(zip(allRatios.tolist(), allRTs.tolist()))))
    RTsBinned = np.array_split(allRTsSortedByRatio, NUMBINS)
    meanRTs = [np.mean(_) for _ in RTsBinned]
    plt.plot(meanRTs, label=label, linestyle=linestyle, marker='o')
    plt.xlabel("Gain/Loss ratio")
    plt.ylabel("RT")
    plt.title("All data")


plotData(data, 'Actual data', '-')


fullModelData = np.genfromtxt("simulatedData/fullModel.csv", delimiter=',')[:, 1:]
plotData(fullModelData, 'Full Model', '--')

noLossAversionData = np.genfromtxt("simulatedData/noLossAversion.csv", delimiter=',')[:, 1:]
plotData(noLossAversionData, 'No Loss Aversion', '--')

noPredecisionalBiasData = np.genfromtxt("simulatedData/noPredecisionalBias.csv", delimiter=',')[:, 1:]
plotData(noPredecisionalBiasData, 'No Predecisional Bias', '--')

noAlphaData = np.genfromtxt("simulatedData/noAlpha.csv", delimiter=',')[:, 1:]
plotData(noAlphaData, 'No Fixed Utility Bias', '--')

plt.legend()
plt.show()
exit()

# simData = np.genfromtxt("/Users/nishadsinghi/undergrad-project-loss-aversion/DDM/allParticipantDataClubbedSimulatedData.csv", delimiter=',')[1:, 1:]
# simData = np.genfromtxt("/Users/nishadsinghi/undergrad-project-loss-aversion/DDM/DDMSimCleanedDataAllClubbedOnlyStartingBias.csv", delimiter=',')[1:, 1:]
# simData = np.genfromtxt("/Users/nishadsinghi/undergrad-project-loss-aversion/newDataset/dataZeroSureDDMSim.csv", delimiter=',')[1:, 2:]
# plotData(simData, 'DDM', '--')


# LCA
numSimulationsPerCondition = 100

def generateLCAData(lcaWrapper, params):
    allModelData = np.zeros(4)
    for stakes in allStakes:
        print(stakes)
        gainValue, lossValue = stakes
        allSimulations = [lcaWrapper(gainValue, lossValue, *params) for _ in
                          range(numSimulationsPerCondition)]
        allValidResponseSimulations = list(filter(filterFunction, allSimulations))
        numValidResponses = len(allValidResponseSimulations)
        _, allModelRTs, allModelResponses = zip(*allValidResponseSimulations)

        allModelResponses = np.array(allModelResponses)
        allModelRTs = np.array(allModelRTs)
        modelStakes = np.hstack((np.full((numValidResponses, 1), gainValue), np.full((numValidResponses, 1), lossValue)))
        modelDataForStakes = np.hstack((np.array(allModelResponses).reshape(-1, 1), np.array(allModelRTs).reshape(-1, 1), modelStakes))
        allModelData = np.vstack((allModelData, modelDataForStakes))

    allModelData = allModelData[1:, :]
    return allModelData


# set up look-up table (LUT)
LUTInterval = 0.0001
numAccumulators = 2
stdNormalLUT = prepareStdNormalLUT(LUTInterval)
sampleFromLUT = SampleFromLUT(stdNormalLUT)
sampleFromZeroMeanLUT = lambda stdDev: sampleFromLUT(0, stdDev, numAccumulators)

# set-up the LCA
identityUtilityFunction = lambda x: x
getValueInput = GetValueInputZeroReference(identityUtilityFunction)
# getValueInput = GetValueInputZeroReference(identityUtilityFunction)

maxTimeSteps = 1000
deltaT = 0.02
prepareRecurrentWeights = PrepareRecurrentWeights(numAccumulators)
lca = RunLCASimulation(getValueInput, sampleFromZeroMeanLUT, prepareRecurrentWeights, maxTimeSteps, deltaT)
# lca = RunLCASimulation(getValueInput2Attributes2Choices, sampleFromZeroMeanLUT, prepareRecurrentWeights, maxTimeSteps, deltaT)

attributeProbabilities = (0.5, 0.5)
startingActivation = (0, 0)

# # UNEQUAL THRESHOLDS
# getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
# lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold1, threshold2, constantInput: \
#     lca(attributeProbabilities, getChoiceAttributes(gain, loss), startingActivation, decay, competition, constantInput,
#         noiseStdDev, nonDecisionTime, [threshold1, threshold2])
#
# params = [2.9222192 ,  1.97753516, 17.59714645,  0.18920589,  7.08568937, 8.62839634, 28.39363504]
# unequalThresholdsData = generateLCAData(lcaWrapper, params)
# # plotData(unequalThresholdsData, 'Unequal thresholds', '--')


# # STARTING BIAS
# def getStartingActivation(startingBias, threshold):
#     if startingBias < 0:
#         return [-1*startingBias*threshold, 0]
#     else:
#         return [0, startingBias*threshold]
#
# getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
# getAllThresholds = lambda threshold: [threshold]*numAccumulators
# lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, startingBias, constantInput: \
#     lca(attributeProbabilities, getChoiceAttributes(gain, loss), getStartingActivation(startingBias, threshold), decay, competition, constantInput,
#         noiseStdDev, nonDecisionTime, getAllThresholds(threshold))
#
#
# # params = [1.77895768e+00,  3.20906638e+00,  9.58128213e+01,  1.96114120e-01, 3.40131479e+01, -3.41666518e-01,  7.71126655e+01]
# params = [2.06155217e+00,  3.18317118e+00,  2.08697307e+01,  1.27605889e-01, 1.04339815e+01, -1.95599890e-01,  3.01482276e+01]
# startingBiasData = generateLCAData(lcaWrapper, params)
# plotData(startingBiasData, 'LCA with starting point bias', '--')
#
# plt.legend()
# plt.tight_layout()
# plt.savefig('RTFitsDataset2.png', bbox_inches='tight')
# exit()



# LOSS AVERSION
attributeProbabilities = (0.5, 0.5)
startingActivation = (0, 0)
getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
getAllThresholds = lambda threshold: [threshold]*numAccumulators

lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, lossWeight, constantInput: \
    lca(attributeProbabilities, getChoiceAttributes(gain, lossWeight*loss), startingActivation, decay, competition, constantInput,
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

# params = [2.97553172,   4.76614439, 148.09895125,   0.19744745, 33.2914632 , 101.92218251,   2.17809482]
params = [3.70756359,  3.45232175, 22.60422262,  0.28049256,  7.20611797, 2.17565966, 31.180381]
lossAversionData = generateLCAData(lcaWrapper, params)
plotData(lossAversionData, 'Unequal weights', '--')


# LOSS ATTENTION
startingActivation = (0, 0)
getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
getAllThresholds = lambda threshold: [threshold]*numAccumulators

lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, constantInput, attribute1Prob: \
    lca((attribute1Prob, 1-attribute1Prob), getChoiceAttributes(gain, loss), startingActivation, decay, competition, constantInput,
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

# params = [1.55893187e+00, 2.59678066e+00, 1.36329544e+02, 1.25511453e-01, 4.71199432e+01, 1.11088120e+02, 3.57679855e-01]
params = [1.42469771,  2.3423496 , 17.30099185,  0.12683208,  7.85291344, 17.59723663,  0.26822441]
lossAttentionData = generateLCAData(lcaWrapper, params)
plotData(lossAttentionData, 'Unequal attention', '--')

plt.legend()
plt.savefig('dataset2OnlyLossAversionAndOnlyLossAttentionRTVsGainLossRatio.png', bbox_inches='tight')
plt.close()
exit()



#
#
# # CONSTANT OFFSET
# attributeProbabilities = (0.5, 0.5)
# startingActivation = (0, 0)
# getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
# getAllThresholds = lambda threshold: [threshold]*numAccumulators
#
# lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, constantInput1, constantInput2: \
#     lca(attributeProbabilities, getChoiceAttributes(gain, loss), startingActivation, decay, competition,
#         (constantInput1, constantInput2), noiseStdDev, nonDecisionTime, getAllThresholds(threshold))
#
# # params = [7.09735461e-01, 1.67785047e+00, 1.45887027e+02, 1.35412016e-03, 7.75527680e+01, 1.35863504e+02, 1.20132635e+02]
# params = [4.03268051,  4.85055685, 18.00478189,  0.30715232,  6.25095921, 30.3708972 , 28.10031369]
# constantOffsetData = generateLCAData(lcaWrapper, params)
# plotData(constantOffsetData, 'Constant offset', '--')







plt.legend()
plt.show()