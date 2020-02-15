import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(DIRNAME)

from LCA import PrepareRecurrentWeights, RunLCASimulation, getValueInput
from LUT import prepareStdNormalLUT, SampleFromLUT

import numpy as np
import pickle
from matplotlib import pyplot as plt

# read observed data
data = np.genfromtxt("risk_data.csv", delimiter=',')[1:, 1:]

def selectConditionTrials(gain, loss):
    selectedData = data[data[:, 2] == gain][:]
    selectedData = selectedData[selectedData[:, 3] == loss][:]
    return selectedData

def responseRTForCondition(gain, loss, response):
    selectedDataForCondition = selectConditionTrials(gain, loss)
    responseRT = (selectedDataForCondition[selectedDataForCondition[:, 0] == response][:, 1:]).T[0]
    return responseRT

def RTQuantilesForResponse(gain, loss, response, numQuantiles):
    responseRTs = responseRTForCondition(gain, loss, response)
    quantiles = [np.quantile(responseRTs, i) for i in np.arange(1/numQuantiles, 1+(1/numQuantiles), 1/numQuantiles)]
    return quantiles


# set up look-up table (LUT)
LUTInterval = 0.0001
numAccumulators = 2
stdNormalLUT = prepareStdNormalLUT(LUTInterval)
sampleFromLUT = SampleFromLUT(stdNormalLUT)
sampleFromZeroMeanLUT = lambda stdDev: sampleFromLUT(0, stdDev, numAccumulators)

# set-up the LCA
maxTimeSteps = 750
deltaT = 0.01
prepareRecurrentWeights = PrepareRecurrentWeights(numAccumulators)
lca = RunLCASimulation(getValueInput, sampleFromZeroMeanLUT, prepareRecurrentWeights, maxTimeSteps, deltaT)

attributeProbabilities = (0.5, 0.5)
startingActivation = (0, 0)

getChoiceAttributes = lambda gain, loss: np.array([[gain, loss], [0, 0]])
getAllThresholds = lambda threshold: [threshold]*numAccumulators
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, constantInput: \
    lca(attributeProbabilities, getChoiceAttributes(gain, loss), startingActivation, decay, competition, constantInput,
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))


def filterFunction(tup):
    if tup[-1] != -1:
        return True
    else:
        return False

# class CostFunction:
#     def __init__(self, allGainValues, allLossValues):
#         self.allGainValues = allGainValues
#         self.allLossValues = allLossValues
#
#     def __call__(self, parameters):
#         cost = 0
#         for gainValue in self.allGainValues:
#             for lossValue in self.allLossValues:
#                 numEmpiricalObservations = np.shape(selectConditionTrials(gainValue, lossValue))[0]
#                 allSimulations = [lcaWrapper(gainValue, lossValue, *parameters) for _ in
#                                   range(numEmpiricalObservations)]
#                 allValidResponseSimulations = list(filter(filterFunction, allSimulations))
#                 print("Number of valid responses: {}, Number of empirical responses: {}".format(len(allValidResponseSimulations), numEmpiricalObservations))
#                 _, allModelRTs, allModelResponses = zip(*allValidResponseSimulations)
#                 allModelRTs = np.array(allModelRTs)
#                 allModelResponses = np.array(allModelResponses)
#                 modelRTsReject = allModelRTs[(np.where(allModelResponses == 0))[0]]
#                 hist, _ = np.histogram(modelRTsReject, np.concatenate())
#         #         conditionCost = 400 * ((P_model - P_data) ** 2) + ((RT_data_mean - RT_model_mean) ** 2) / (
#         #                     RT_data_stdDev ** 2) \
#         #                         + 0.0002 * ((RT_data_stdDev - RT_model_stdDev) ** 2)
#         #         cost += conditionCost
#         # #
#         # # return cost


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
                # print("Number of valid response simulations: ", len(allValidResponseSimulations))
                if len(allValidResponseSimulations) == 0:
                    return -1
                _, allModelRTs, allModelResponses = zip(*allValidResponseSimulations)
                P_model = np.mean(allModelResponses)
                RT_model_mean = np.mean(allModelRTs)
                RT_model_stdDev = np.std(allModelRTs)
                P_data = computePData(gainValue, lossValue)
                RT_data_mean, RT_data_stdDev = computeRTStatsData(gainValue, lossValue)
                conditionCost = 400 * ((P_model - P_data) ** 2) + ((RT_data_mean - RT_model_mean) ** 2) / (
                            RT_data_stdDev ** 2) \
                                + 0.0002 * ((RT_data_stdDev - RT_model_stdDev) ** 2)
                cost += conditionCost

        return cost

allGainValues = list(range(10, 110, 10))
allLossValues = list(range(-100, 0, 10))
costFunction = CostFunction(100, allGainValues, allLossValues)

allCost = []

for trial in range(1, 11):
    readFile = open("BTPpickles/{}.pickle".format(trial), 'rb')
    record = pickle.load(readFile)
    finalParams = record[-1, :]
    cost = costFunction(finalParams)
    allCost.append(cost)
    print("Cost for trial {} = {}".format(trial, cost))

plt.hist(allCost)
plt.title("Final cost using Metropolis Algorithm (10 trials)")
plt.xlabel("Cost")
plt.ylabel("Frequency")
plt.show()
