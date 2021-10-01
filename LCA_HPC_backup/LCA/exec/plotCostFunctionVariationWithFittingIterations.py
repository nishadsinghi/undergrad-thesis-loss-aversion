import pickle
import numpy as np
from scipy.stats import skew, chisquare

file = open("/Users/nishadsinghi/undergrad-project-loss-aversion/LCA/data/varyCostFunction/furtherFineTuningCorona/data from Gantavya HPC/fitLCAAllParametersForcedToBePositiveAndNonZeroStartingPoint.pickle", 'rb')
record = pickle.load(file)

diff = np.diff(record, axis=0)
change = np.apply_along_axis(all, 1, diff).astype(int)
change = np.append([1], change)
argchange = np.argwhere(change == 1).flatten()

import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "..", "src"))

from LCA import PrepareRecurrentWeights, RunLCASimulation, GetValueInputZeroReference, getValueInput2Attributes2Choices
from LUT import prepareStdNormalLUT, SampleFromLUT

from matplotlib import pyplot as plt
import numpy as np

TERM1COEFF = 150
TERM2COEFF = 1
TERM3COEFF = 2.5
TERM4COEFF = 0.2

# read observed data
data = np.genfromtxt("../src/risk_data.csv", delimiter=',')[1:, 1:]


# set up look-up table (LUT)
LUTInterval = 0.0001
numAccumulators = 2
stdNormalLUT = prepareStdNormalLUT(LUTInterval)
sampleFromLUT = SampleFromLUT(stdNormalLUT)
sampleFromZeroMeanLUT = lambda stdDev: sampleFromLUT(0, stdDev, numAccumulators)


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
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold1, threshold2, constantInput, valueScalingParam: \
    lca(attributeProbabilities, getChoiceAttributes(valueScalingParam*gain, valueScalingParam*loss), startingActivation, decay, competition, constantInput,
        noiseStdDev, nonDecisionTime, [threshold1, threshold2])

# lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold1, threshold2, constantInput: \
#     lca(attributeProbabilities, getChoiceAttributes(gain, loss), getStartingActivations([threshold1, threshold2]), decay, competition, constantInput,
#         noiseStdDev, nonDecisionTime, [threshold1, threshold2])


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
                    observedPOfThisChoice = numObservedRTsForChoice / numObservedTrials
                    quantilesBoundaries = np.quantile(observedRTsForChoice, quantiles, interpolation='higher')
                    observedProportionsWeighted = observedProportionsChoiceWise * observedPOfThisChoice

                    modelTrialsForChoice = modelTrials[modelTrials[:, 0] == choice]
                    modelRTsForChoice = modelTrialsForChoice[:, 1]
                    modelProportions = np.histogram(modelRTsForChoice, bins=np.concatenate(([0], quantilesBoundaries, [100])))[
                                           0] / numModelTrials
                    totalCost += chisquare(modelProportions, observedProportionsWeighted)[0]

        return (totalCost, parameters[3] - delta)
        # cost = 0
        # t1 = 0
        # t2 = 0
        # t3 = 0
        # t4 = 0
        # allModelData = np.zeros(4)
        # for gainValue in self.allGainValues:
        #     for lossValue in self.allLossValues:
        #         allSimulations = [lcaWrapper(gainValue, lossValue, *parameters) for _ in
        #                           range(self.numSimulationsPerCondition)]
        #         allValidResponseSimulations = list(filter(filterFunction, allSimulations))
        #         numValidResponses = len(allValidResponseSimulations)
        #         print("Gain: {}, Loss: {}, NumValidResponses: {}".format(gainValue, lossValue, numValidResponses))
        #         if numValidResponses < self.numSimulationsPerCondition / 3:
        #             return (-1, parameters[3])
        #         _, allModelRTs, allModelResponses = zip(*allValidResponseSimulations)
        #         modelStakes = np.hstack((np.full((numValidResponses, 1), gainValue), np.full((numValidResponses, 1), lossValue)))
        #         modelDataForStakes = np.hstack((np.array(allModelResponses).reshape(-1, 1), np.array(allModelRTs).reshape(-1, 1), modelStakes))
        #         allModelData = np.vstack((allModelData, modelDataForStakes))
        #
        # allModelData = allModelData[1:, :]
        #
        # actualDataMeanRT = np.mean(data[:, 1])
        # simDataMeanRT = np.mean(allModelData[:, 1])
        # delta = simDataMeanRT - actualDataMeanRT
        # if delta > parameters[3]:
        #     delta = parameters[3]
        #
        # allModelData[:, 1] = allModelData[:, 1] - delta
        #
        # for gain in allGainValues:
        #     for loss in allLossValues:
        #         for choice in range(2):
        #             if (computePData(allModelData, gain, loss) == 1 and choice == 0) or (computePData(allModelData, gain, loss) == 0 and choice == 1):
        #                 continue
        #             dataSD = computeRTStatsData(data, gain, loss, choice)[1]
        #             t1 += ((computePData(data, gain, loss) - computePData(allModelData, gain, loss)) ** 2)
        #             if dataSD <= 0.1:
        #                 t2 += ((computeRTStatsData(data, gain, loss, choice)[0] - computeRTStatsData(allModelData, gain, loss, choice)[0]) ** 2)
        #             else:
        #                 t2 += ((computeRTStatsData(data, gain, loss, choice)[0] -
        #                         computeRTStatsData(allModelData, gain, loss, choice)[0]) ** 2) / (dataSD ** 2)
        #             t3 += ((computeRTStatsData(data, gain, loss, choice)[1] -
        #                     computeRTStatsData(allModelData, gain, loss, choice)[1]) ** 2)
        #             t4 += (computeRTStatsData(data, gain, loss, choice)[2] - computeRTStatsData(allModelData, gain, loss, choice)[2])**2
        #
        #             print("Choice: {}, P_model: {}, meanRT_model: {}, stdRT_model: {}, skew_model: {}".format(choice, computePData(allModelData, gain, loss), *computeRTStatsData(allModelData, gain, loss, choice)))
        #             print("Choice: {}, P_data: {}, meanRT_data: {}, stdRT_data: {}, skew_data: {}".format(choice, computePData(data, gain, loss), *computeRTStatsData(data, gain, loss, choice)))
        #         print("---------------------")
        #
        #
        # cost = TERM1COEFF*t1 + TERM2COEFF*t2 + TERM3COEFF*t3 + TERM4COEFF*t4
        # print(t1, t2, t3, t4)
        # return (cost, parameters[3] - delta)

allGainValues = list(range(10, 110, 10))
allLossValues = list(range(-100, 0, 10))

costFunction = CostFunction(500, allGainValues, allLossValues)
# c1 = costFunction([1.15466437e-01, 2.36121712e-01, 9.12961129e+01, 9.28031547e-01, 5.60606893e+01, 6.35346529e+01, 1.40481640e+02])
# c2 = costFunction([0.2245408945202022, 2.561492713707515, 75.28143580606088, 1.010847547480403, 28.39706525583036, 36.47742752290966, 122.4132270325614])
# c3 = costFunction([0.25, 1.1, 90, 0.66, 70.1362561/1.4, 95/1.4, 100])
# print("1382")
# print("Cost: ", c1)
# print("-----------")
# print("1331")
# print("Cost: ", c2)
# print("-----------")
# print("manual")
# print("Cost: ", c3)

print(costFunction([4.660387297855354, 3.0612528095458145, 90.61660242686926, 0.30574954748040306, 20.50894987699313, 27.963092785111176, 107.68779883432228, 1]))


# print(costFunction([0.25, 1.1, 80, 0.66, 70.1362561/1.5, 95/1.5, 100])) # 0.79, 1.908, x so that overall 251
exit()

# costs = []
# for i in argchange:
#     print("Index: ", i)
#     params = record[i, :].flatten()
#     cost = costFunction(params)
#     costs.append(cost)
#
# argchangeMod = [0]
# for i in range(1, len(argchange)):
#     argchangeMod.extend([argchange[i]-1, argchange[i]])
#
# costsMod = []
# for i in range(len(costs) - 1):
#     costsMod.extend([costs[i], costs[i]])
# costsMod.append(costs[-1])
#
#
# plt.plot(argchangeMod, costsMod, linestyle='-')
# plt.xlabel("Fitting Iteration")
# plt.ylabel("Cost")
# plt.xscale("log")
# plt.show()
