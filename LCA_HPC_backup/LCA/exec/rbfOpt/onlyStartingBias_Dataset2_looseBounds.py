import sys
import os

os.environ['OMP_NUM_THREADS'] = '1'

DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "..", "..", "src"))

from LCA_randomResponse import PrepareRecurrentWeights, RunLCASimulation, GetValueInputZeroReference, \
    getValueInput2Attributes2Choices
from LUT import prepareStdNormalLUT, SampleFromLUT

import numpy as np
from scipy.stats import skew, truncnorm, chisquare
import rbfopt




# read observed data
data = np.genfromtxt("../../src/dataZeroSure_RT_StdDevLessThan2.5_250_10500.csv", delimiter=',')[1:, 1:]
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
attributeProbabilities = [0.5, 0.5]

lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, startingBias, constantInput: \
    lca(attributeProbabilities, getChoiceAttributes(gain, loss), getStartingActivation(startingBias, threshold), decay, competition, constantInput,
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
                return 1e6
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

                if numObservedRTsForChoice < 5:
                    continue

                quantilesBoundaries = np.quantile(observedRTsForChoice, quantiles)
                expectedFrequencies = \
                    np.histogram(observedRTsForChoice, bins=np.concatenate(([0], quantilesBoundaries, [100])))[0]

                if numObservedRTsForChoice == 5:
                    expectedFrequencies = observedProportionsChoiceWise * numObservedRTsForChoice

                modelTrialsForChoice = modelTrials[modelTrials[:, 0] == choice]
                modelRTsForChoice = modelTrialsForChoice[:, 1]
                modelFrequencies = \
                    np.histogram(modelRTsForChoice, bins=np.concatenate(([0], quantilesBoundaries, [100])))[0] * (
                                numObservedTrials / numModelTrials)

                totalCost += chisquare(modelFrequencies, expectedFrequencies)[0]

        return totalCost

costFunction = CostFunction(5, allStakes)
bounds = ((0, 20), (0, 20), (0, 100), (0, 2), (0, 100), (-1, 0), (0, 125))   #decay, competition, noiseStdDev, nonDecisionTime, threshold, startingBias, constantInput

settings = rbfopt.RbfoptSettings(minlp_solver_path='~/bonmin-linux64/bonmin', nlp_solver_path='~/ipopt-linux64/ipopt', num_cpus=10, eps_impr=25, max_clock_time=(100*3600 - 10*60), save_state_file='onlyStartingBias_dataset2_looseBounds.dat', save_state_interval=10, max_evaluations=25000, max_iterations=25000)

low, high = list(zip(*bounds))
bb = rbfopt.RbfoptUserBlackBox(7, low, high, np.array(['R']*7), costFunction)
alg = rbfopt.RbfoptAlgorithm(settings, bb)
val, x, itercount, evalcount, fast_evalcount = alg.optimize()

print(val, x, itercount, evalcount, fast_evalcount)
