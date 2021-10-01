import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "..", "src"))

from LCA import PrepareRecurrentWeights, RunLCASimulation, GetValueInputZeroReference
from LUT import prepareStdNormalLUT, SampleFromLUT

import numpy as np
import pickle
import time

# read observed data
data = np.genfromtxt("../src/risk_data.csv", delimiter=',')[1:, 1:]

TERM1COEFF = 200
TERM2COEFF = 1
TERM3COEFF = 1

# set up look-up table (LUT)
LUTInterval = 0.0001
numAccumulators = 2
stdNormalLUT = prepareStdNormalLUT(LUTInterval)
sampleFromLUT = SampleFromLUT(stdNormalLUT)
sampleFromZeroMeanLUT = lambda stdDev: sampleFromLUT(0, stdDev, numAccumulators)

# set-up the LCA
maxTimeSteps = 1000
deltaT = 0.01
prepareRecurrentWeights = PrepareRecurrentWeights(numAccumulators)
identityUtilityFunction = lambda x: 0.2*x
getValueInput = GetValueInputZeroReference(identityUtilityFunction)
lca = RunLCASimulation(getValueInput, sampleFromZeroMeanLUT, prepareRecurrentWeights, maxTimeSteps, deltaT)

attributeProbabilities = (0.5, 0.5)
startingActivation = (0, 0)

getChoiceAttributes = lambda gain, loss: np.array([[gain, loss], [0, 0]])
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
    return (np.mean(reactionTimes), np.std(reactionTimes))

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
        self.functionCalls = 0

    def __call__(self, parameters):
        t1 = 0
        t2 = 0
        t3 = 0
        allModelData = np.zeros(4)
        for gainValue in self.allGainValues:
            for lossValue in self.allLossValues:
                allSimulations = [lcaWrapper(gainValue, lossValue, *parameters) for _ in
                                  range(self.numSimulationsPerCondition)]
                allValidResponseSimulations = list(filter(filterFunction, allSimulations))
                numValidResponses = len(allValidResponseSimulations)
                if numValidResponses < self.numSimulationsPerCondition / 3:
                    return 1000000000
                _, allModelRTs, allModelResponses = zip(*allValidResponseSimulations)
                modelStakes = np.hstack((np.full((numValidResponses, 1), gainValue), np.full((numValidResponses, 1), lossValue)))
                modelDataForStakes = np.hstack((np.array(allModelResponses).reshape(-1, 1), np.array(allModelRTs).reshape(-1, 1), modelStakes))
                allModelData = np.vstack((allModelData, modelDataForStakes))

        allModelData = allModelData[1:, :]

        actualDataMeanRT = np.mean(data[:, 1])
        simDataMeanRT = np.mean(allModelData[:, 1])
        delta = simDataMeanRT - actualDataMeanRT

        allModelData[:, 1] = allModelData[:, 1] - delta

        for gain in allGainValues:
            for loss in allLossValues:
                for choice in range(2):
                    t1 += ((computePData(data, gain, loss) - computePData(allModelData, gain, loss)) ** 2)
                    dataSD = computeRTStatsData(data, gain, loss, choice)[1]
                    if dataSD <= 0.1:
                        t2 += ((computeRTStatsData(data, gain, loss, choice)[0] - computeRTStatsData(allModelData, gain, loss, choice)[0]) ** 2)
                    else:
                        t2 += ((computeRTStatsData(data, gain, loss, choice)[0] -
                                computeRTStatsData(allModelData, gain, loss, choice)[0]) ** 2) / (dataSD ** 2)
                    t3 += ((computeRTStatsData(data, gain, loss, choice)[1] -
                            computeRTStatsData(allModelData, gain, loss, choice)[1]) ** 2)


        cost = TERM1COEFF*t1 + TERM2COEFF*t2 + TERM3COEFF*t3
        print(cost)
        return cost

allGainValues = list(range(30, 70, 10))
allLossValues = list(range(-70, -20, 10))                       ########################################################
costFunction = CostFunction(50, allGainValues, allLossValues)

import scipy

min = scipy.optimize.fmin(costFunction, [1.2198117, 0.93995038, 55.62458808, -1.78430894, 24.26204649, 31.98458065, 40.44150842], disp=True, xtol=0.01, ftol=5)

startTime = time.time()


print(min[0])
print(min[1])

saveFile = open('simplexFinalResult.pickle', 'wb')
pickle.dump(min, saveFile)
saveFile.close()


# nelder mead: [0.52672552, 0.50870876, 55.2412187, 1.04319053, 4.8159347, 11.78213358, 113.1203357]
# powell: [9.73903503e-02, -9.03268148e+00. 7.59758537e+03, 1.28281772e+00, 1.63019236e+01, 1.89570158e+01, 1.03809397e+02]
