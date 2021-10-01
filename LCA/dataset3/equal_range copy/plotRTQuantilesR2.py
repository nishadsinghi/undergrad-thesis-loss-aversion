import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "..", "..", "src"))

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
import matplotlib

from LCA import PrepareRecurrentWeights, RunLCASimulation, GetValueInputZeroReference, getValueInput2Attributes2Choices
from LUT import prepareStdNormalLUT, SampleFromLUT


data = np.genfromtxt("data_preprocessed.csv", delimiter=',')[1:, 2:]
allStakes = np.unique(data[:, -2:], axis=0)

# matplotlib.rcParams.update({'font.size': 22})


def filterFunction(tup):
    if tup[-1] != -1:
        return True
    else:
        return False

QUANTILES = [0.1, 0.3, 0.5, 0.7, 0.9]
def plotR2(ax, modelRecord, data):
    def selectConditionTrials(gain, loss):
        selectedData = data[data[:, 2] == gain][:]
        selectedData = selectedData[selectedData[:, 3] == loss][:]
        return selectedData

    allModelQuantiles = []
    allActualQuantiles = []
    for stakes in allStakes:
        gain, loss = stakes
        modelTrialsForThisStakes = modelRecord[np.logical_and(modelRecord[:, 2] == gain, modelRecord[:, 3] == loss)]
        modelRTs = modelTrialsForThisStakes[:, 1].flatten()
        modelQuantiles = [np.quantile(modelRTs, quantile) for quantile in QUANTILES]
        allModelQuantiles.append(modelQuantiles)

        actualTrialsForThisStakes = data[np.logical_and(data[:, 2] == gain, data[:, 3] == loss)]
        actualRTs = actualTrialsForThisStakes[:, 1].flatten()
        actualQuantiles = [np.quantile(actualRTs, quantile) for quantile in QUANTILES]
        allActualQuantiles.append(actualQuantiles)

    r2Score = r2_score(np.array(allActualQuantiles).flatten(), np.array(allModelQuantiles).flatten())

    ax.scatter(np.array(allActualQuantiles).flatten(), np.array(allModelQuantiles).flatten())
    ax.plot((0, 4), (0, 4), linestyle='--', color='r')
    ax.annotate("R2 Score: %.2f" % r2Score, (0, 3.6))
    ax.set_xlabel("Empirical RT Quantiles")
    ax.set_ylabel("Model RT Quantiles")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)


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

attributeProbabilities = (0.5, 0.5)
startingActivation = (0, 0)
getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
getAllThresholds = lambda threshold: [threshold]*numAccumulators

def getStartingActivation(startingBias, threshold):
    if startingBias < 0:
        return [-1*startingBias*threshold, 0]
    else:
        return [0, startingBias*threshold]


numSimulationsPerCondition = 100

def generateLCAData(lcaWrapper, params):
    modelRecord = np.zeros(4)
    for stakes in allStakes:
        gainValue, lossValue = stakes
        allSimulations = [lcaWrapper(gainValue, lossValue, *params) for _ in
                          range(numSimulationsPerCondition)]
        allValidResponseSimulations = list(filter(filterFunction, allSimulations))
        _, allModelRTs, allModelResponses = zip(*allValidResponseSimulations)
        numValidSimulations = np.size(allModelRTs)
        modelRecordForStakes = np.vstack((allModelResponses, allModelRTs, np.full(numValidSimulations, gainValue), np.full(numValidSimulations, lossValue))).T
        modelRecord = np.vstack((modelRecord, modelRecordForStakes))

    modelRecord = modelRecord[1:, :]

    return modelRecord

# FULL LCA MODEL
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, startingBias, lossWeight, constantInput1, constantInput2: \
    lca(attributeProbabilities, getChoiceAttributes(gain, lossWeight * loss),
        getStartingActivation(startingBias, threshold), decay, competition, (constantInput1, constantInput2),
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))


params = [1.74549528e+00,  3.17903030e+00,  2.80430931e+01,  2.14425490e-01, 2.56822918e+01, -3.71411918e-02,  1.74424428e+00,  6.61264367e+01, 6.93722980e+01]
ax1 = plt.subplot2grid(shape=(2,2), loc=(0,0), colspan=1)
# plotData(ax1, data, '-', 'Actual data')
modelRecord = generateLCAData(lcaWrapper, params)
# plotData(ax1, modelRecord, '--', 'Full LCA')
plotR2(ax1, modelRecord, data)
ax1.set_title("Full LCA")


# NO LOSS AVERSION
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, startingBias, constantInput1, constantInput2: \
    lca(attributeProbabilities, getChoiceAttributes(gain, loss), getStartingActivation(startingBias, threshold), decay,
        competition, (constantInput1, constantInput2),
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

params = [2.61577469e+00, 4.30567645e+00, 5.11963733e+01, 4.24286552e-01, 2.00665808e+01, 5.50324662e-03, 5.74473121e+01, 5.57126277e+01]
ax2 = plt.subplot2grid(shape=(2,2), loc=(0,1), colspan=1)
# plotData(ax2, data, '-', 'Actual data')
modelRecord = generateLCAData(lcaWrapper, params)
# plotData(ax2, modelRecord, '--', 'No Loss Aversion')
plotR2(ax2, modelRecord, data)
ax2.set_title("No Loss Aversion")


# NO PREDECISIONAL BIAS
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, lossWeight, constantInput1, constantInput2: \
    lca(attributeProbabilities, getChoiceAttributes(gain, lossWeight * loss), getStartingActivation(0, threshold),
        decay, competition, (constantInput1, constantInput2),
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

params = [1.72537805,   3.14166275,  29.70051179,   0.45390954, 18.93316093,   1.86035748,  49.49073978,  52.2559686]
ax3 = plt.subplot2grid(shape=(2,2), loc=(1,0), colspan=1)
# plotData(ax3, data, '-', 'Actual data')
modelRecord = generateLCAData(lcaWrapper, params)
# plotData(ax3, modelRecord, '--', 'No Starting bias')
plotR2(ax3, modelRecord, data)
ax3.set_title("No Starting point bias")


# NO FIXED UTILIY BIAS
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, startingBias, lossWeight, constantInput: \
    lca(attributeProbabilities, getChoiceAttributes(gain, lossWeight * loss),
        getStartingActivation(startingBias, threshold), decay, competition, constantInput,
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

params = [2.74790253e+00,  4.01386458e+00,  2.43103345e+01,  5.78575468e-01, 1.20616881e+01, -3.26133146e-02,  1.37598125e+00,  4.26439994e+01]
ax4 = plt.subplot2grid(shape=(2,2), loc=(1,1), colspan=1)
# plotData(ax4, data, '-', 'Actual data')
modelRecord = generateLCAData(lcaWrapper, params)
# plotData(ax4, modelRecord, '--', 'No Fixed Utility bias')
plotR2(ax4, modelRecord, data)
ax4.set_title("No Fixed Utility bias")


# plt.legend()

plt.tight_layout()
plt.savefig("RTQuantileFitsR2.png", bbox_inches='tight')
plt.close()
