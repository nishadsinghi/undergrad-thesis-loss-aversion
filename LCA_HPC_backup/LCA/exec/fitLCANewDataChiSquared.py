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

SAVEFILENAME = '../data/phase2/newDataSet_ChiSquared_proportions_bounds=10_10_150_5_150_150_150.pickle'

# read observed data
data = np.genfromtxt("../../newDataset/dataZeroSure_cleaned.csv", delimiter=',')[1:, 1:]
allStakes = np.unique(data[:, -2:], axis=0)

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
    def __init__(self, numSimulationsPerCondition, allStakes):
        self.numSimulationsPerCondition = numSimulationsPerCondition
        self.allStakes = allStakes

    def __call__(self, parameters):
        allModelData = np.zeros(4)
        for stakes in self.allStakes:
            gainValue, lossValue = stakes
            print("gain: {}, loss: {}".format(gainValue, lossValue))
            allSimulations = [lcaWrapper(gainValue, lossValue, *parameters) for _ in
                              range(self.numSimulationsPerCondition)]
            allValidResponseSimulations = list(filter(filterFunction, allSimulations))
            numValidResponses = len(allValidResponseSimulations)
            print("number of valid responses: ", numValidResponses)
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

        print("shape of combined model data: ", np.shape(allModelData))


        totalCost = 0
        quantiles = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        observedProportionsChoiceWise = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
        for stakes in self.allStakes:
            gain, loss = stakes
            observedTrials = selectConditionTrials(data, gain, loss)
            numObservedTrials = np.shape(observedTrials)[0]
            print("observed trials: ", numObservedTrials)
            modelTrials = selectConditionTrials(allModelData, gain, loss)
            numModelTrials = np.shape(modelTrials)[0]
            print("model trials: ", numModelTrials)
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
                print("total model frequencies: {}, expected frequencies: {}".format(np.sum(modelFrequencies), np.sum(expectedFrequencies)))
            print("-------------------------")

        return (totalCost, parameters[3] - delta)

allStakes = np.unique(data[:, -2:], axis=0)
costFunction = CostFunction(75, allStakes)

costFunction([2.27530735e+00, 6.18845147e+00, 1.42463935e+02, 2.45541032e-02, 3.42787557e+01, 4.31549031e+01, 8.41641920e+01])
exit()

bounds = ((0, 10), (0, 10), (0, 150), (0, 5), (0, 150), (0, 150), (0, 150))

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
initialParameterVariance = (3, 3, 50, 0, 50, 50, 50)

counter = 0
while True:
    print(counter)
    counter += 1
    initialParameterValues = generateProposalParams((5, 5, 75, 0.5, 75, 75, 75), initialParameterVariance)
    initialCost, adjustedContTime = costFunction(initialParameterValues)
    if initialCost != -1:
        initialParameterValues[3] = adjustedContTime
        break

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
