from LCA import PrepareRecurrentWeights, RunLCASimulation, getValueInput
from LUT import prepareStdNormalLUT, SampleFromLUT

import numpy as np
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
                    return 1000000
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
costFunction = CostFunction(50, allGainValues, allLossValues)

from scipy.optimize import minimize
x0 = (0.5, 0.5, 20, 0.5, 10, 25)
min = minimize(costFunction, x0, method='nelder-mead', options={'disp':True})

print(min.x)

saveFile = open('simplexFinalResult.pickle', 'wb')
pickle.dump(min, saveFile)
saveFile.close()
