import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "..", "..", "src"))

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

experimentalData = np.genfromtxt("data_preprocessed.csv", delimiter=',')[1:, 2:]
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
    ax.set_ylabel("P(accept)", fontsize=10)
    ax.set_xlabel("Choice-factor adjusted RT", fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)

                                                    # SIMULATION
numSimulationsPerCondition = 100

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

# FULL MODEL
def getStartingActivation(startingBias, threshold):
    if startingBias < 0:
        return [-1 * startingBias * threshold, 0]
    else:
        return [0, startingBias * threshold]


getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
getAllThresholds = lambda threshold: [threshold] * numAccumulators
attributeProbabilities = [0.5, 0.5]

lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, startingBias, lossWeight, constantInput1, constantInput2: \
    lca(attributeProbabilities, getChoiceAttributes(gain, lossWeight * loss),
        getStartingActivation(startingBias, threshold), decay, competition, (constantInput1, constantInput2),
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))


params = [1.74549528e+00,  3.17903030e+00,  2.80430931e+01,  2.14425490e-01, 2.56822918e+01, -3.71411918e-02,  1.74424428e+00,  6.61264367e+01, 6.93722980e+01]
modelSimulationData = generateModelData(lcaWrapper, params)
ax1 = plt.subplot2grid(shape=(2,2), loc=(0,0), colspan=1)
plotData(ax1, modelSimulationData, 'LCA', '--')
plotData(ax1, experimentalData, 'Actual data', '-')
ax1.set_title("Full LCA Model", fontsize=10)



# NO LOSS AVERSION
def getStartingActivation(startingBias, threshold):
    if startingBias < 0:
        return [-1 * startingBias * threshold, 0]
    else:
        return [0, startingBias * threshold]


getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
getAllThresholds = lambda threshold: [threshold] * numAccumulators
attributeProbabilities = [0.5, 0.5]

lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, startingBias, constantInput1, constantInput2: \
    lca(attributeProbabilities, getChoiceAttributes(gain, loss), getStartingActivation(startingBias, threshold), decay,
        competition, (constantInput1, constantInput2),
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

params = [2.61577469e+00, 4.30567645e+00, 5.11963733e+01, 4.24286552e-01, 2.00665808e+01, 5.50324662e-03, 5.74473121e+01, 5.57126277e+01]
modelSimulationData = generateModelData(lcaWrapper, params)
ax2 = plt.subplot2grid(shape=(2,2), loc=(0,1), colspan=1)
plotData(ax2, modelSimulationData, 'LCA', '--')
plotData(ax2, experimentalData, 'Actual data', '-')
ax2.set_title("No Loss Aversion", fontsize=10)



# NO PREDECISIONAL BIAS
def getStartingActivation(startingBias, threshold):
    if startingBias < 0:
        return [-1 * startingBias * threshold, 0]
    else:
        return [0, startingBias * threshold]


getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
getAllThresholds = lambda threshold: [threshold] * numAccumulators
attributeProbabilities = [0.5, 0.5]

lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, lossWeight, constantInput1, constantInput2: \
    lca(attributeProbabilities, getChoiceAttributes(gain, lossWeight * loss), getStartingActivation(0, threshold),
        decay, competition, (constantInput1, constantInput2),
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

params = [1.72537805,   3.14166275,  29.70051179,   0.45390954, 18.93316093,   1.86035748,  49.49073978,  52.2559686]
modelSimulationData = generateModelData(lcaWrapper, params)

ax3 = plt.subplot2grid(shape=(2,2), loc=(1,0), colspan=1)
plotData(ax3, modelSimulationData, 'LCA', '--')
plotData(ax3, experimentalData, 'Actual data', '-')
ax3.set_title("No Predecisional bias", fontsize=10)




# NO FIXED UTILIY BIAS
def getStartingActivation(startingBias, threshold):
    if startingBias < 0:
        return [-1 * startingBias * threshold, 0]
    else:
        return [0, startingBias * threshold]


getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
getAllThresholds = lambda threshold: [threshold] * numAccumulators
attributeProbabilities = [0.5, 0.5]

lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, startingBias, lossWeight, constantInput: \
    lca(attributeProbabilities, getChoiceAttributes(gain, lossWeight * loss),
        getStartingActivation(startingBias, threshold), decay, competition, constantInput,
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

params = [2.74790253e+00,  4.01386458e+00,  2.43103345e+01,  5.78575468e-01, 1.20616881e+01, -3.26133146e-02,  1.37598125e+00,  4.26439994e+01]
modelSimulationData = generateModelData(lcaWrapper, params)
ax4 = plt.subplot2grid(shape=(2,2), loc=(1,1), colspan=1)
plotData(ax4, modelSimulationData, 'LCA', '--')
plotData(ax4, experimentalData, 'Actual data', '-')
ax4.set_title("No Fixed Utility bias", fontsize=10)


plt.xticks((0, 1, 2, 3, 4))
plt.ylim(0, 1)

plt.legend()

plt.tight_layout()
plt.savefig("LCABehaviouralMarkers.png", bbox_inches='tight')
plt.close()





