import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "..", "..", "src"))

from LCA import PrepareRecurrentWeights, RunLCASimulation, GetValueInputZeroReference, \
    getValueInput2Attributes2Choices
from LUT import prepareStdNormalLUT, SampleFromLUT

import numpy as np
import time
import pickle
from scipy.stats import skew, truncnorm, chisquare

# read observed data
data = np.genfromtxt("../../src/dataZeroSure_cleaned.csv", delimiter=',')[1:, 1:]
allStakes = np.unique(data[:, -2:], axis=0)

np.random.seed()

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

def getStartingActivation(startingBias, threshold):
    if startingBias < 0:
        return [-1*startingBias*threshold, 0]
    else:
        return [0, startingBias*threshold]

getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
getAllThresholds = lambda threshold: [threshold]*numAccumulators

lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, startingBias, lossWeight, attribute1Prob, constantInput1, constantInput2: \
    lca((attribute1Prob, 1-attribute1Prob), getChoiceAttributes(gain, lossWeight*loss), getStartingActivation(startingBias, threshold), decay, competition, (constantInput1, constantInput2),
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))


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


# define the cost function
class CostFunction:
    def __init__(self, numSimulationsPerCondition, allStakes):
        self.numSimulationsPerCondition = numSimulationsPerCondition
        self.allStakes = allStakes

    def __call__(self, parameters):
        allModelData = np.zeros(4)
        for stakes in self.allStakes:
            gainValue, lossValue = stakes
            allSimulations = [lcaWrapper(gainValue, lossValue, *parameters) for _ in
                              range(self.numSimulationsPerCondition)]
            allValidResponseSimulations = list(filter(filterFunction, allSimulations))
            numValidResponses = len(allValidResponseSimulations)
            if numValidResponses < self.numSimulationsPerCondition / 3:
                return (-1, parameters[3])
            _, allModelRTs, allModelResponses = zip(*allValidResponseSimulations)
            modelStakes = np.hstack(
                (np.full((numValidResponses, 1), gainValue), np.full((numValidResponses, 1), lossValue)))
            modelDataForStakes = np.hstack(
                (np.array(allModelResponses).reshape(-1, 1), np.array(allModelRTs).reshape(-1, 1), modelStakes))
            allModelData = np.vstack((allModelData, modelDataForStakes))

        allModelData = allModelData[1:, :]

        actualDataMeanRT = np.mean(data[:, 1])
        simDataMeanRT = np.mean(allModelData[:, 1])
        delta = simDataMeanRT - actualDataMeanRT
        if delta > parameters[3]:
            delta = parameters[3]

        allModelData[:, 1] = allModelData[:, 1] - delta

        totalCost = 0
        quantiles = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        observedProportionsChoiceWise = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
        for stakes in self.allStakes:
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
                modelFrequencies = \
                np.histogram(modelRTsForChoice, bins=np.concatenate(([0], quantilesBoundaries, [100])))[
                    0] / numModelTrials

                totalCost += chisquare(modelFrequencies, expectedFrequencies)[0]
        return (totalCost, parameters[3] - delta)


costFunction = CostFunction(75, allStakes)

print(costFunction([2.66338518,  2.91922264, 21.42119226,  0.34500113,  5.86430269, -0.16472694,  2.6974676 ,  0.56216513, 17.15150966, 17.15150966]))