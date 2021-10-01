# import sys
# import os
# DIRNAME = os.path.dirname(__file__)
# sys.path.append(os.path.join(DIRNAME, "..", "src"))

from LCA import PrepareRecurrentWeights, RunLCASimulation, GetValueInputZeroReference#, getValueInput2Attributes2Choices
from LUT import prepareStdNormalLUT, SampleFromLUT

from matplotlib import pyplot as plt
import numpy as np
import time
import pickle

# read observed data
data = np.genfromtxt("risk_data.csv", delimiter=',')[1:, 1:]

# set up look-up table (LUT)
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
# lca = RunLCASimulation(getValueInput2Attributes2Choices, sampleFromZeroMeanLUT, prepareRecurrentWeights, maxTimeSteps, deltaT)

attributeProbabilities = (0.5, 0.5)
startingActivation = (0, 0)
# constantInput = (100, 100)

getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
# getAllThresholds = lambda threshold: [threshold]*numAccumulators
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold1, threshold2, constantInput: \
    lca(attributeProbabilities, getChoiceAttributes(gain, loss), startingActivation, decay, competition, constantInput,
        noiseStdDev, nonDecisionTime, [threshold1, threshold2])


def selectConditionTrials(gain, loss):
    selectedData = data[data[:, 2] == gain][:]
    selectedData = selectedData[selectedData[:, 3] == loss][:]
    return selectedData

def computePData(gain, loss):
    conditionTrialData = selectConditionTrials(gain, loss)
    responses = conditionTrialData[:, 0]
    return np.mean(responses)

def computeRTStatsData(gain, loss):
    conditionTrialData = selectConditionTrials(gain, loss)
    reactionTimes = conditionTrialData[:, 1]
    return (np.mean(reactionTimes), np.std(reactionTimes))

def filterFunction(tup):
    if tup[-1] != -1:
        return True
    else:
        return False

term1Coeff = 300
term3Coeff = 0.02

# define the cost function
class CostFunction:
    def __init__(self, numSimulationsPerCondition, allGainValues, allLossValues):
        self.numSimulationsPerCondition = numSimulationsPerCondition
        self.allGainValues = allGainValues
        self.allLossValues = allLossValues

    def __call__(self, parameters):
        cost = 0
        for gainValue in self.allGainValues:
            for lossValue in self.allLossValues:
                allSimulations = [lcaWrapper(gainValue, lossValue, *parameters) for _ in
                                  range(self.numSimulationsPerCondition)]
                allValidResponseSimulations = list(filter(filterFunction, allSimulations))
                print("Number of valid responses: ", len(allValidResponseSimulations))
                if len(allValidResponseSimulations) < self.numSimulationsPerCondition/3:
                    return -1 #1000000
                _, allModelRTs, allModelResponses = zip(*allValidResponseSimulations)
                P_model = np.mean(allModelResponses)
                RT_model_mean = np.mean(allModelRTs)
                RT_model_stdDev = np.std(allModelRTs)
                P_data = computePData(gainValue, lossValue)
                RT_data_mean, RT_data_stdDev = computeRTStatsData(gainValue, lossValue)
                conditionCost = term1Coeff * ((P_model - P_data) ** 2) + ((RT_data_mean - RT_model_mean) ** 2) / (
                            RT_data_stdDev ** 2) \
                                + term3Coeff * ((RT_data_stdDev - RT_model_stdDev) ** 2)
                cost += conditionCost

        return cost

allGainValues = list(range(10, 110, 10))
allLossValues = list(range(-100, 0, 10))
costFunction = CostFunction(75, allGainValues, allLossValues)

# minimization algorithm
initialParameterVariance = (1, 1, 10, 0.25, 20, 20, 20)

while True:
    initialParameterValues = np.random.normal(0, initialParameterVariance)
    if initialParameterValues[6] < 0:
        initialParameterValues[6] = -initialParameterValues[6]

    if initialParameterValues[5] < 0:
        initialParameterValues[5] = -initialParameterValues[5]

    if initialParameterValues[4] < 0:
        initialParameterValues[4] = -initialParameterValues[4]

    if initialParameterValues[3] < 0:
        initialParameterValues[3] = -initialParameterValues[3]

    if initialParameterValues[2] < 0:
        initialParameterValues[2] = -initialParameterValues[2]

    initialCost = costFunction(initialParameterValues)
    
    if initialCost != -1:
        break

bestParameters = initialParameterValues
parameterVariance = np.asarray(initialParameterVariance)
minimumCost = initialCost
recordOfBestParameters = bestParameters

startTimeOverall = time.time()
numIterationsMinAlgo = 20000
for minimizationIteration in range(numIterationsMinAlgo):
    lca.simulationTimes = []
    lca.numIterations = []
    startTime = time.time()
    # decrease the value of variance
    if minimizationIteration % 100 == 0:
        parameterVariance = (1, 1, 0.99, 0.99, 0.99, 0.99, 0.99) * parameterVariance

    # propose new parameters
    newParameters = np.random.normal(loc=bestParameters, scale=parameterVariance)

    # # check values of new parameters
    if newParameters[5] < 0:
        newParameters[5] = -newParameters[5]

    if newParameters[4] < 0:
        newParameters[4] = -newParameters[4]

    if newParameters[3] < 0:
        newParameters[3] = -newParameters[3]

    if newParameters[2] < 0:
        newParameters[2] = -newParameters[2]

    # compute cost
    cost = costFunction(newParameters)
    if cost == -1:
        continue

    # update parameters
    if cost < minimumCost:
        bestParameters = newParameters
        minimumCost = cost

    # record best parameters
    recordOfBestParameters = np.vstack((recordOfBestParameters, bestParameters))
    saveFile = open('{}_1_{}_new.pickle'.format(term1Coeff, term3Coeff), 'wb')
    pickle.dump(recordOfBestParameters, saveFile)
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