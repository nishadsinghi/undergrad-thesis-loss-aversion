import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "..", "src"))

import numpy as np
from matplotlib import pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import matplotlib
from scipy.stats import skew, chisquare
import seaborn as sns

from LCA import PrepareRecurrentWeights, RunLCASimulation, GetValueInputZeroReference, getValueInput2Attributes2Choices
from LUT import prepareStdNormalLUT, SampleFromLUT

data = np.genfromtxt("../src/risk_data_cleaned.csv", delimiter=',')[:, 1:]
allStakes = np.unique(data[:, -2:], axis=0)

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

def computeCost(allModelData):
    totalCost = 0
    quantiles = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    observedProportionsChoiceWise = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
    for stakes in allStakes:
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
            numModelRTsForChoice = np.size(modelRTsForChoice)
            modelFrequencies = \
            np.histogram(modelRTsForChoice, bins=np.concatenate(([0], quantilesBoundaries, [100])))[
                0] / numModelTrials

            totalCost += chisquare(modelFrequencies, expectedFrequencies)[0]

    return totalCost

                # SIMULATION
numSimulationsPerCondition = 150

def filterFunction(tup):
    if tup[-1] != -1:
        return True
    else:
        return False


def generateModelData(lcaWrapper, params):
    allModelData = np.zeros(4)
    for stakes in allStakes:
        gainValue, lossValue = stakes
        allSimulations = [lcaWrapper(gainValue, lossValue, *params) for _ in
                          range(numSimulationsPerCondition)]
        allValidResponseSimulations = list(filter(filterFunction, allSimulations))
        numValidResponses = len(allValidResponseSimulations)
        _, allModelRTs, allModelResponses = zip(*allValidResponseSimulations)

        allModelResponses = np.array(allModelResponses)
        allModelRTs = np.array(allModelRTs)
        modelStakes = np.hstack((np.full((numValidResponses, 1), gainValue), np.full((numValidResponses, 1), lossValue)))
        modelDataForStakes = np.hstack((np.array(allModelResponses).reshape(-1, 1), np.array(allModelRTs).reshape(-1, 1), modelStakes))
        allModelData = np.vstack((allModelData, modelDataForStakes))

    allModelData = allModelData[1:, :]
    return allModelData


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

labels = []
costs = []

# LOSS AVERSION
attributeProbabilities = (0.5, 0.5)
startingActivation = (0, 0)
getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
getAllThresholds = lambda threshold: [threshold]*numAccumulators

lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, constantInput, lossWeight: \
    lca(attributeProbabilities, getChoiceAttributes(gain, lossWeight*loss), startingActivation, decay, competition, constantInput,
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

params = [2.97553172,   4.76614439, 148.09895125,   0.19744745, 33.2914632 , 101.92218251,   2.17809482]
# params = [2.68790632,  1.00256435, 25.88580797,  0.12270113,  9.05514628, 24.59621147,  2.8273047]
modelSimulationData = generateModelData(lcaWrapper, params)
cost = computeCost(modelSimulationData)
print(cost)
labels.append("Loss Aversion")
costs.append(cost)


# LOSS ATTENTION
startingActivation = (0, 0)
getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
getAllThresholds = lambda threshold: [threshold]*numAccumulators

lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, constantInput, attribute1Prob: \
    lca((attribute1Prob, 1-attribute1Prob), getChoiceAttributes(gain, loss), startingActivation, decay, competition, constantInput,
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

params = [1.55893187e+00, 2.59678066e+00, 1.36329544e+02, 1.25511453e-01, 4.71199432e+01, 1.11088120e+02, 3.57679855e-01]
# params = [1.42469771,  2.3423496 , 17.30099185,  0.12683208,  7.85291344, 17.59723663,  0.26822441]
modelSimulationData = generateModelData(lcaWrapper, params)
cost = computeCost(modelSimulationData)
print(cost)
labels.append("Attention \nbias")
costs.append(cost)



# UNEQUAL STARTING POINTS
attributeProbabilities = (0.5, 0.5)

def getStartingActivation(startingBias, threshold):
    if startingBias < 0:
        return [-1*startingBias*threshold, 0]
    else:
        return [0, startingBias*threshold]

getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
getAllThresholds = lambda threshold: [threshold]*numAccumulators
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, startingBias, constantInput: \
    lca(attributeProbabilities, getChoiceAttributes(gain, loss), getStartingActivation(startingBias, threshold), decay, competition, constantInput,
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

params = [1.77895768e+00,  3.20906638e+00,  9.58128213e+01,  1.96114120e-01, 3.40131479e+01, -3.41666518e-01,  7.71126655e+01]
# params = [2.06155217e+00,  3.18317118e+00,  2.08697307e+01,  1.27605889e-01, 1.04339815e+01, -1.95599890e-01,  3.01482276e+01]
modelSimulationData = generateModelData(lcaWrapper, params)
cost = computeCost(modelSimulationData)
print(cost)
labels.append("Predecisional \nbias")
costs.append(cost)


# UNEQUAL THRESHOLDS
attributeProbabilities = (0.5, 0.5)
startingActivation = (0, 0)

getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
# getAllThresholds = lambda threshold: [threshold]*numAccumulators
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold1, threshold2, constantInput: \
    lca(attributeProbabilities, getChoiceAttributes(gain, loss), startingActivation, decay, competition, constantInput,
        noiseStdDev, nonDecisionTime, [threshold1, threshold2])

params = [1.87624118e+00, 1.50234826e+00, 1.25197286e+02, 1.66428120e-01, 3.29702771e+01, 4.85272755e+01, 8.80408658e+01]
# params = [2.9222192 ,  1.97753516, 17.59714645,  0.18920589,  7.08568937, 8.62839634, 28.39363504]
modelSimulationData = generateModelData(lcaWrapper, params)
cost = computeCost(modelSimulationData)
print(cost)
labels.append("Unequal \nThresholds")
costs.append(cost)



# CONSTANT OFFSET
attributeProbabilities = (0.5, 0.5)
startingActivation = (0, 0)
getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
getAllThresholds = lambda threshold: [threshold]*numAccumulators

lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, constantInput1, constantInput2: \
    lca(attributeProbabilities, getChoiceAttributes(gain, loss), startingActivation, decay, competition,
        (constantInput1, constantInput2), noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

params = [7.09735461e-01, 1.67785047e+00, 1.45887027e+02, 1.35412016e-03, 7.75527680e+01, 1.35863504e+02, 1.20132635e+02]
# params = [4.03268051,  4.85055685, 18.00478189,  0.30715232,  6.25095921, 30.3708972 , 28.10031369]
modelSimulationData = generateModelData(lcaWrapper, params)
cost = computeCost(modelSimulationData)
print(cost)
labels.append("Fixed utility \nbias")
costs.append(cost)

plt.bar(labels, costs, color=['black', 'red', 'green', 'blue', 'cyan'])
plt.xlabel("Model / Hypothesis", fontsize=17)
plt.ylabel(r"$\chi^2$", fontsize=17)
plt.title("Dataset - 1", fontsize=17)
# plt.legend()
plt.show()




sns.set(font_scale=1.9)
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(12, 12))
splot = sns.barplot(labels, costs, ax=ax)

for p in splot.patches:
    splot.annotate(format(p.get_height(), '.1f'),
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'center',
                   xytext = (0, -16),
                   textcoords = 'offset points')

plt.xlabel("Model / Hypothesis")
plt.ylabel(r"$\chi^2$")
plt.title(r"$\chi^2$ fit statistic for Dataset-1")
plt.savefig('chiSquaredSingleBiasModels.png', bbox_inches='tight')





# plt.xticks((0, 1, 2, 3, 4))
# plt.ylabel("P(accept)", fontsize=18)
# plt.xlabel("Choice-factor adjusted RT", fontsize=18)
# plt.title("Dataset - 1", fontsize=18)
# plt.legend(fontsize=16)
# plt.ylim(0, 1)
# plt.show()





