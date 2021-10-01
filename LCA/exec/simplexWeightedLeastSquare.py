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
from scipy.optimize import minimize

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
                    # return (-1, parameters[3])
                    return 100000000
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
        quantileWeights = [2, 2, 1, 1, 0.5]
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
                    numModelRTs = np.size(modelRTsForChoice)
                    if numModelRTs < 5:
                        continue
                    modelProportions = np.histogram(modelRTsForChoice, bins=np.concatenate(([0], quantilesBoundaries, [100])))[
                                           0] / numModelTrials

                    P_model = np.size(modelRTsForChoice) / numModelTrials

                    modelQuantileBoundaries = np.quantile(modelRTsForChoice, quantiles, interpolation='higher')
                    totalCost += 4 * ((P_model - observedPOfThisChoice) ** 2) + \
                                 observedPOfThisChoice * np.sum(
                        [weight * ((modelQ - dataQ) ** 2) for weight, modelQ, dataQ in
                         zip(quantileWeights, modelQuantileBoundaries, quantilesBoundaries)])

                    # totalCost += chisquare(modelProportions, observedProportionsWeighted)[0]

        print("parameters: ", parameters, " cost: ", totalCost)
        return totalCost#, parameters[3] - delta)


allGainValues = list(range(10, 110, 10))
allLossValues = list(range(-100, 0, 10))

costFunction = CostFunction(200, allGainValues, allLossValues)

x0 = [1.95186167,   2.44990787, 141.03682298,   0.36185888, 25.49401448,  37.39690481,  64.3019711]
x1 = [4.59485926,   3.10097219,  89.9451625,    0.30719986,  20.46732939, 28.62127894, 108.34603301]

print("Cost of x0: ", costFunction(x0))

startTime = time.time()
res = minimize(costFunction, x0, method='Nelder-Mead', options={'disp': True, 'return_all': True, 'xatol': 0.01, 'fatol': 1, 'adaptive': True})
endTime = time.time()

print("-------------------------")
print("END")
print(res)
print("time taken: ", endTime-startTime)

saveFile = open('simplexWeightedLeastSquare.pickle', 'wb')
pickle.dump(res, saveFile)
saveFile.close()


# nelder mead: [0.52672552, 0.50870876, 55.2412187, 1.04319053, 4.8159347, 11.78213358, 113.1203357]
# powell: [9.73903503e-02, -9.03268148e+00. 7.59758537e+03, 1.28281772e+00, 1.63019236e+01, 1.89570158e+01, 1.03809397e+02]
