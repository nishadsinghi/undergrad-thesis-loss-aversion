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
from scipy import stats

from LCA import PrepareRecurrentWeights, RunLCASimulation, GetValueInputZeroReference, getValueInput2Attributes2Choices
from LUT import prepareStdNormalLUT, SampleFromLUT

# experimentalData = np.genfromtxt("../src/risk_data.csv", delimiter=',')[1:, 1:]
experimentalData = np.genfromtxt("../../newDataset/dataZeroSure_cleaned.csv", delimiter=',')[1:, 1:]
allStakes = np.unique(experimentalData[:, -2:], axis=0)

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return h

def plotData(ax, data, label, linestyle):
    allLogRT = np.log(data[:, 1]).reshape(-1, 1)
    allGainLoss = data[:, 2:]
    regressor = LinearRegression()
    regressor.fit(allGainLoss, allLogRT)
    allPredictedLogRT = regressor.predict(allGainLoss)
    allLogRTResidual = allLogRT - allPredictedLogRT

    responses = data[:, 0]
    _, sortedResponses = (list(t) for t in zip(*sorted(zip(allLogRTResidual.tolist(), responses.tolist()))))

    binnedResponses = np.array_split(sortedResponses, 5)
    binPAccept = [np.mean(binResponses) for binResponses in binnedResponses]
    CI = [mean_confidence_interval(binResponses) for binResponses in binnedResponses]

    ax.errorbar(x=(0, 1, 2, 3, 4), y=binPAccept, yerr=CI, marker='o', label=label, linestyle=linestyle)
    ax.set_xticks((0, 1, 2, 3, 4))
    ax.set_ylabel("P(accept)", fontsize=16)
    ax.set_xlabel("Choice-factor adjusted RT", fontsize=16)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=16)

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

# LOSS AVERSION
attributeProbabilities = (0.5, 0.5)
startingActivation = (0, 0)
getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
getAllThresholds = lambda threshold: [threshold]*numAccumulators

lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, constantInput, lossWeight: \
    lca(attributeProbabilities, getChoiceAttributes(gain, lossWeight*loss), startingActivation, decay, competition, constantInput,
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

# params = [2.97553172,   4.76614439, 148.09895125,   0.19744745, 33.2914632 , 101.92218251,   2.17809482]
params = [2.68790632,  1.00256435, 25.88580797,  0.12270113,  9.05514628, 24.59621147,  2.8273047]
modelSimulationData = generateModelData(lcaWrapper, params)
ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
plotData(ax1, modelSimulationData, 'LCA', '--')
plotData(ax1, experimentalData, 'Actual data', '-')
ax1.set_title("Unequal weight", fontsize=16)


# LOSS ATTENTION
startingActivation = (0, 0)
getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
getAllThresholds = lambda threshold: [threshold]*numAccumulators

lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, constantInput, attribute1Prob: \
    lca((attribute1Prob, 1-attribute1Prob), getChoiceAttributes(gain, loss), startingActivation, decay, competition, constantInput,
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

# params = [1.55893187e+00, 2.59678066e+00, 1.36329544e+02, 1.25511453e-01, 4.71199432e+01, 1.11088120e+02, 3.57679855e-01]
params = [1.42469771,  2.3423496 , 17.30099185,  0.12683208,  7.85291344, 17.59723663,  0.26822441]
modelSimulationData = generateModelData(lcaWrapper, params)
ax2 = plt.subplot2grid(shape=(2,6), loc=(0,2), colspan=2)
plotData(ax2, modelSimulationData, 'LCA', '--')
plotData(ax2, experimentalData, 'Actual data', '-')
ax2.set_title("Unequal attention", fontsize=16)



# CONSTANT OFFSET
attributeProbabilities = (0.5, 0.5)
startingActivation = (0, 0)
getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
getAllThresholds = lambda threshold: [threshold]*numAccumulators

lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, constantInput1, constantInput2: \
    lca(attributeProbabilities, getChoiceAttributes(gain, loss), startingActivation, decay, competition,
        (constantInput1, constantInput2), noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

# params = [7.09735461e-01, 1.67785047e+00, 1.45887027e+02, 1.35412016e-03, 7.75527680e+01, 1.35863504e+02, 1.20132635e+02]
params = [4.03268051,  4.85055685, 18.00478189,  0.30715232,  6.25095921, 30.3708972 , 28.10031369]
modelSimulationData = generateModelData(lcaWrapper, params)

ax3 = plt.subplot2grid(shape=(2,6), loc=(0,4), colspan=2)
plotData(ax3, modelSimulationData, 'LCA', '--')
plotData(ax3, experimentalData, 'Actual data', '-')
ax3.set_title("Unequal constant input", fontsize=16)




# UNEQUAL THRESHOLDS
attributeProbabilities = (0.5, 0.5)
startingActivation = (0, 0)

getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
# getAllThresholds = lambda threshold: [threshold]*numAccumulators
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold1, threshold2, constantInput: \
    lca(attributeProbabilities, getChoiceAttributes(gain, loss), startingActivation, decay, competition, constantInput,
        noiseStdDev, nonDecisionTime, [threshold1, threshold2])

# params = [1.87624118e+00, 1.50234826e+00, 1.25197286e+02, 1.66428120e-01, 3.29702771e+01, 4.85272755e+01, 8.80408658e+01]
params = [2.9222192 ,  1.97753516, 17.59714645,  0.18920589,  7.08568937, 8.62839634, 28.39363504]
modelSimulationData = generateModelData(lcaWrapper, params)
ax4 = plt.subplot2grid(shape=(2,6), loc=(1,0), colspan=3)
plotData(ax4, modelSimulationData, 'LCA', '--')
plotData(ax4, experimentalData, 'Actual data', '-')
ax4.set_title("Unequal thresholds", fontsize=16)


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

# params = [1.77895768e+00,  3.20906638e+00,  9.58128213e+01,  1.96114120e-01, 3.40131479e+01, -3.41666518e-01,  7.71126655e+01]
params = [2.06155217e+00,  3.18317118e+00,  2.08697307e+01,  1.27605889e-01, 1.04339815e+01, -1.95599890e-01,  3.01482276e+01]
modelSimulationData = generateModelData(lcaWrapper, params)
ax5 = plt.subplot2grid(shape=(2,6), loc=(1,3), colspan=3)
plotData(ax5, modelSimulationData, 'LCA', '--')
plotData(ax5, experimentalData, 'Actual data', '-')
ax5.set_title("Unequal starting point", fontsize=16)





plt.xticks((0, 1, 2, 3, 4))
# plt.ylabel("P(accept)", fontsize=18)
# plt.xlabel("Choice-factor adjusted RT", fontsize=18)
# plt.title("Dataset - 1", fontsize=18)
# plt.legend(fontsize=16)
plt.ylim(0, 1)
plt.show()





