import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "..", "src"))

from LCA import PrepareRecurrentWeights, RunLCASimulation, GetValueInputZeroReference, getValueInput2Attributes2Choices
from LUT import prepareStdNormalLUT, SampleFromLUT

from matplotlib import pyplot as plt
import numpy as np
import time
import pickle
from scipy.stats import skew, truncnorm, chisquare

SAVEFILENAME = '../data/chiSquareCorrected_bound=5_contd.pickle'

# read observed data
data = np.genfromtxt("../src/risk_data.csv", delimiter=',')[1:, 1:]

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
startingActivation = (0, 0)

getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
# getAllThresholds = lambda threshold: [threshold]*numAccumulators
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold1, threshold2, constantInput: \
    lca(attributeProbabilities, getChoiceAttributes(gain, loss), startingActivation, decay, competition, constantInput,
        noiseStdDev, nonDecisionTime, [threshold1, threshold2])


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
        return (0, 0)
    reactionTimes = dataForThisChoice[:, 1]
    return (np.mean(reactionTimes), np.std(reactionTimes), skew(reactionTimes))

def filterFunction(tup):
    if tup[-1] != -1:
        return True
    else:
        return False


# define the cost function
class CostFunction:
    def __init__(self, numSimulationsPerCondition, allGainValues, allLossValues):
        self.numSimulationsPerCondition = numSimulationsPerCondition
        self.allGainValues = allGainValues
        self.allLossValues = allLossValues

    def __call__(self, parameters):
        allModelData = np.zeros(4)
        for gainValue in self.allGainValues:
            for lossValue in self.allLossValues:
                allSimulations = [lcaWrapper(gainValue, lossValue, *parameters) for _ in
                                  range(self.numSimulationsPerCondition)]
                allValidResponseSimulations = list(filter(filterFunction, allSimulations))
                numValidResponses = len(allValidResponseSimulations)
                if numValidResponses < self.numSimulationsPerCondition / 3:
                    return (-1, parameters[3])
                _, allModelRTs, allModelResponses = zip(*allValidResponseSimulations)
                modelStakes = np.hstack((np.full((numValidResponses, 1), gainValue), np.full((numValidResponses, 1), lossValue)))
                modelDataForStakes = np.hstack((np.array(allModelResponses).reshape(-1, 1), np.array(allModelRTs).reshape(-1, 1), modelStakes))
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
        for gain in allGainValues:
            for loss in allLossValues:
                observedTrials = selectConditionTrials(data, gain, loss)
                numObservedTrials = np.shape(observedTrials)[0]
                modelTrials = selectConditionTrials(allModelData, gain, loss)
                numModelTrials = np.shape(modelTrials)[0]
                for choice in range(2):
                    observedTrialsForChoice = observedTrials[observedTrials[:, 0] == choice]
                    observedRTsForChoice = observedTrialsForChoice[:, 1]
                    numObservedRTsForChoice = np.size(observedRTsForChoice)
                    # observedPOfThisChoice = numObservedRTsForChoice / numObservedTrials
                    quantilesBoundaries = np.quantile(observedRTsForChoice, quantiles)
                    # observedProportionsWeighted = observedProportionsChoiceWise * observedPOfThisChoice
                    expectedFrequencies = \
                    np.histogram(observedRTsForChoice, bins=np.concatenate(([0], quantilesBoundaries, [100])))[0]

                    if numObservedRTsForChoice == 5:
                        expectedFrequencies = observedProportionsChoiceWise * numObservedRTsForChoice

                    elif numObservedRTsForChoice < 5:
                        continue

                    modelTrialsForChoice = modelTrials[modelTrials[:, 0] == choice]
                    modelRTsForChoice = modelTrialsForChoice[:, 1]
                    numModelRTsForChoice = np.size(modelRTsForChoice)
                    print(numModelRTsForChoice)
                    # modelProportions = np.histogram(modelRTsForChoice, bins=np.concatenate(([0], quantilesBoundaries, [100])))[
                    #                        0] / numModelTrials
                    if numModelRTsForChoice != 0:
                        modelFrequencies = \
                        np.histogram(modelRTsForChoice, bins=np.concatenate(([0], quantilesBoundaries, [100])))[0] * (
                                    numObservedRTsForChoice / np.size(modelRTsForChoice))
                    else:
                        modelFrequencies = \
                        np.histogram(modelRTsForChoice, bins=np.concatenate(([0], quantilesBoundaries, [100])))[0]

                    totalCost += chisquare(modelFrequencies, expectedFrequencies)[0]


        return (totalCost, parameters[3] - delta)


allGainValues = list(range(10, 110, 10))
allLossValues = list(range(-100, 0, 10))
costFunction = CostFunction(75, allGainValues, allLossValues)

bounds = ((0, 5), (0, 5), (0, 100), (0, 5), (0, 200), (0, 200), (0, 150))

def generateProposalParams(currentParams, variances):
    newParams = []
    for i in range(len(currentParams)):
        myclip_a = bounds[i][0]
        myclip_b = bounds[i][1]
        my_mean = currentParams[i]
        my_std = variances[i]
        if my_std == 0:
            newParam = my_mean
        else:
            a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
            newParam = truncnorm.rvs(a, b, loc=my_mean, scale=my_std)
        newParams.append(newParam)

    return newParams


# minimization algorithm
initialParameterVariance = (2, 2, 20, 0, 40, 40, 50) * (np.array((0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99)) ** 35)

# counter = 0
# while True:
#     print(counter)
#     counter += 1
#     initialParameterValues = generateProposalParams((1, 1, 50, 0.5, 100, 100, 50), initialParameterVariance)
#     initialCost, adjustedContTime = costFunction(initialParameterValues)
#     if initialCost != -1:
#         initialParameterValues[3] = adjustedContTime
#         break

initialParameterValues = [2.4010140468546757, 2.491310846596547, 98.0134508156539, 0.30319488081373636, 23.102203624006513, 33.22456607580175, 73.96977722602814]
initialCost = 2896.95638051285
print("Initial value identified: ", initialParameterValues, " Cost: ", initialCost)

bestParameters = initialParameterValues
parameterVariance = np.asarray(initialParameterVariance)
minimumCost = initialCost
recordOfBestParameters = bestParameters
recordOfBestCost = minimumCost

# try:
#     saveFile = open(SAVEFILENAME, "rb")
#     recordOfBestParameters = pickle.load(saveFile)
#     bestParameters = recordOfBestParameters[-1, :]
#     print("Best parameters: ", bestParameters)
#     numIterationsPrevious = np.shape(recordOfBestParameters)[0]
#     print("NumIterations: ", numIterationsPrevious)
#     minimumCost = costFunction(bestParameters)
#     parameterVarianceMultiplier = np.array((1, 1, 0.99, 0.99, 0.99, 0.99, 0.99))**(numIterationsPrevious//100)
#     print("Num Iterations: ", numIterationsPrevious, " Multiplier: ", parameterVarianceMultiplier)
#     parameterVariance = np.multiply(np.array((1, 1, 10, 0.25, 20, 20, 20)), parameterVarianceMultiplier)
# except:
#     exit()

startTimeOverall = time.time()
numIterationsMinAlgo = 20000
# for minimizationIteration in range(numIterationsPrevious, numIterationsMinAlgo+numIterationsPrevious):
for minimizationIteration in range(numIterationsMinAlgo):
    lca.simulationTimes = []
    lca.numIterations = []
    startTime = time.time()
    # decrease the value of variance
    if minimizationIteration % 100 == 0 and minimizationIteration != 0:
        parameterVariance = (0.99, 0.99, 0.98, 0.99, 0.98, 0.98, 0.99) * parameterVariance

    # propose new parameters
    newParameters = generateProposalParams(bestParameters, parameterVariance)

    # compute cost
    cost, adjustedNonDecisionTime = costFunction(newParameters)

    if cost != -1:
        # update parameters
        if cost < minimumCost:
            print("updating parameters and cost")
            bestParameters = newParameters
            bestParameters[3] = adjustedNonDecisionTime
            minimumCost = cost

    # record best parameters
    recordOfBestParameters = np.vstack((recordOfBestParameters, bestParameters))
    recordOfBestCost = np.vstack((recordOfBestCost, minimumCost))
    finalRecord = np.hstack((recordOfBestParameters, recordOfBestCost))
    saveFile = open(SAVEFILENAME, 'wb')
    pickle.dump(finalRecord, saveFile)
    saveFile.close()

    # time for iteration
    endTime = time.time()
    iterationTime = endTime - startTime
    print("Iteration: {} Time: {}".format(minimizationIteration, iterationTime))
    print("Proposed Value of params: ", newParameters)
    print("Best Value of params: ", bestParameters)
    print("Cost: ", cost)
    print("Best cost: ", minimumCost)
    print("-------------------------------")

endTimeOverall = time.time()
overallTime = endTimeOverall - startTimeOverall

print("Time taken for {} iterations: {} seconds".format(numIterationsMinAlgo, overallTime))
[plt.plot(recordOfBestParameters[:, i]) for i in range(np.shape(recordOfBestParameters)[1])]
plt.show()