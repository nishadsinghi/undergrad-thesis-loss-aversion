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
import matplotlib
from scipy.stats import skew, chisquare
import seaborn as sns

from LCA import PrepareRecurrentWeights, RunLCASimulation, GetValueInputZeroReference, getValueInput2Attributes2Choices
from LUT import prepareStdNormalLUT, SampleFromLUT

data = np.genfromtxt("dataZeroSure_cleaned_250_10500_zScore3.csv", delimiter=',')[:, 1:]
allStakes = np.unique(data[:, -2:], axis=0)

def selectConditionTrials(data, gain, loss):
    selectedData = data[data[:, 2] == gain][:]
    selectedData = selectedData[selectedData[:, 3] == loss][:]
    return selectedData

def computePData(data, gain, loss):
    conditionTrialData = selectConditionTrials(data, gain, loss)
    responses = conditionTrialData[:, 0]
    return np.mean(responses)

def computeRTStatsData(data, gain, loss, choice):
    conditionTrialData = selectConditionTrials(data, gain, loss)
    dataForThisChoice = conditionTrialData[conditionTrialData[:, 0] == choice]
    if np.shape(dataForThisChoice)[0] == 0:
        return (0, 0, 0)
    reactionTimes = dataForThisChoice[:, 1]
    return (np.mean(reactionTimes), np.std(reactionTimes), skew(reactionTimes))

def filterFunction(tup):
    if tup[-1] != -1:
        return True
    else:
        return False

def computeCost(allModelData):
    totalCost = 0
    quantiles = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    observedProportionsChoiceWise = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
    for stakes in allStakes:
        gain, loss = stakes
        observedTrials = selectConditionTrials(data, gain, loss)
        numObservedTrials = np.shape(observedTrials)[0]
        modelTrials = selectConditionTrials(allModelData, gain, loss)
        numModelTrials = np.shape(modelTrials)[0]
        for choice in range(2):
            observedTrialsForChoice = observedTrials[observedTrials[:, 0] == choice]
            observedRTsForChoice = observedTrialsForChoice[:, 1]
            numObservedRTsForChoice = np.size(observedRTsForChoice)
            observedPOfThisChoice = numObservedRTsForChoice / numObservedTrials

            if numObservedRTsForChoice < 5:
                continue

            quantilesBoundaries = np.quantile(observedRTsForChoice, quantiles)

            expectedFrequencies = \
                np.histogram(observedRTsForChoice, bins=np.concatenate(([0], quantilesBoundaries, [100])))[
                    0] / numObservedTrials

            if numObservedRTsForChoice == 5:
                expectedFrequencies = observedProportionsChoiceWise * observedPOfThisChoice

            modelTrialsForChoice = modelTrials[modelTrials[:, 0] == choice]
            modelRTsForChoice = modelTrialsForChoice[:, 1]
            numModelRTsForChoice = np.size(modelRTsForChoice)
            modelFrequencies = \
            np.histogram(modelRTsForChoice, bins=np.concatenate(([0], quantilesBoundaries, [100])))[
                0] / numModelTrials

            totalCost += chisquare(modelFrequencies, expectedFrequencies)[0]

    return totalCost

                # SIMULATION
numSimulationsPerCondition = 200

def filterFunction(tup):
    if tup[-1] != -1:
        return True
    else:
        return False


def generateModelData(lcaWrapper, params):
    allModelData = np.zeros(4)
    for stakes in allStakes:
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


LUTInterval = 0.0001
numAccumulators = 2
stdNormalLUT = prepareStdNormalLUT(LUTInterval)
sampleFromLUT = SampleFromLUT(stdNormalLUT)
sampleFromZeroMeanLUT = lambda stdDev: sampleFromLUT(0, stdDev, numAccumulators)

# set-up the LCA
identityUtilityFunction = lambda x: x
getValueInput = GetValueInputZeroReference(identityUtilityFunction)

maxTimeSteps = 750
deltaT = 0.02
prepareRecurrentWeights = PrepareRecurrentWeights(numAccumulators)
lca = RunLCASimulation(getValueInput, sampleFromZeroMeanLUT, prepareRecurrentWeights, maxTimeSteps, deltaT)

labels = []
costs = []


# FULL MODEL
def getStartingActivation(startingBias, threshold):
    if startingBias < 0:
        return [-1 * startingBias * threshold, 0]
    else:
        return [0, startingBias * threshold]


getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])


def getAllThresholds(thresholdBias, threshold):
    if thresholdBias < 0:
        return [(1 + thresholdBias) * threshold, threshold]
    else:
        return [threshold, threshold * (1 - thresholdBias)]


lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, constantInput1, lossWeight, startingBias, constantInput2, thresholdBias, attribute1Prob: \
    lca((attribute1Prob, 1 - attribute1Prob), getChoiceAttributes(gain, lossWeight * loss),
        getStartingActivation(startingBias, threshold), decay, competition, (constantInput1, constantInput2),
        noiseStdDev, nonDecisionTime, getAllThresholds(thresholdBias, threshold))

params = [ 1.23038319,  2.75410135, 23.01916933,  0.06933039,  7.92593087, 11.47578506, 1.80954224,  0.0339803,  13.11288405, -0.17487191,  0.30492408]
modelSimulationData = generateModelData(lcaWrapper, params)
np.savetxt("simulatedData/5BiasFullModel.csv", modelSimulationData, delimiter=",")
# cost = computeCost(modelSimulationData)
# print(cost)
# labels.append("No Fixed \nUtility bias")
# costs.append(cost)


# sns.set(font_scale=2.2)
# sns.set_style("whitegrid")
# fig, ax = plt.subplots(figsize=(12, 12))
# splot = sns.barplot(labels, costs, ax=ax)
#
# for p in splot.patches:
#     splot.annotate(format(p.get_height(), '.1f'),
#                    (p.get_x() + p.get_width() / 2., p.get_height()),
#                    ha = 'center', va = 'center',
#                    xytext = (0, -12),
#                    textcoords = 'offset points')
#
# plt.xlabel("Model / Hypothesis")
# plt.ylabel(r"$\chi^2$")
# plt.title(r"$\chi^2$ fit statistic for Dataset-1")
# plt.savefig('chiSquared.png', bbox_inches='tight')
