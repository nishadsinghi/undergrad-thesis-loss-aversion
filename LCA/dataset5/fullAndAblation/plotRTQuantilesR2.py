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

import seaborn as sns
sns.set(font_scale=2)

data = np.genfromtxt("data_cleaned_250_10500_zScore3.csv", delimiter=',')[:, 1:]
allStakes = np.unique(data[:, -2:], axis=0)

fig = plt.figure(figsize=(10,12))
fig.suptitle("RT quantile fits - Dataset 3")



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

    for i in range(len(QUANTILES)):
        sns.scatterplot(np.array(allActualQuantiles)[:, i].flatten(), np.array(allModelQuantiles)[:, i].flatten(), alpha=0.6)

    r2Score = r2_score(np.array(allActualQuantiles).flatten(), np.array(allModelQuantiles).flatten())

    # ax.scatter(np.array(allActualQuantiles).flatten(), np.array(allModelQuantiles).flatten())
    ax.plot((0, 5), (0, 5), linestyle='--', color='r')
    ax.annotate("R^2 = %.2f" % r2Score, (0, 4.6))
    ax.set_xlabel("Empirical RT Quantiles")
    ax.set_ylabel("Model RT Quantiles")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_aspect(1)


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


numSimulationsPerCondition = 150

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
def getStartingActivation(startingBias, threshold):
    if startingBias < 0:
        return [-1 * startingBias * threshold, 0]
    else:
        return [0, startingBias * threshold]


getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
getAllThresholds = lambda threshold: [threshold] * numAccumulators
attributeProbabilities = [0.5, 0.5]

lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, constantInput1, lossWeight, startingBias, constantInput2: lca(attributeProbabilities, getChoiceAttributes(gain, lossWeight * loss), getStartingActivation(startingBias, threshold), decay, competition, (constantInput1, constantInput2), noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

params = [ 1.31829065e+00,  2.95010964e+00,  1.40725240e+02,  1.23863248e-01, 4.49504742e+01,  7.85510677e+01,  2.12667308e+00, -3.00291888e-01, 9.60420894e+01]
ax1 = plt.subplot2grid(shape=(2,2), loc=(0,0), colspan=1)
# plotData(ax1, data, '-', 'Actual data')
modelRecord = generateLCAData(lcaWrapper, params)
# plotData(ax1, modelRecord, '--', 'Full LCA')
plotR2(ax1, modelRecord, data)
ax1.set_title("Full LCA")


# NO LOSS AVERSION
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, constantInput1, startingBias, constantInput2: \
    lca(attributeProbabilities, getChoiceAttributes(gain, loss), getStartingActivation(startingBias, threshold), decay,
        competition, (constantInput1, constantInput2),
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

params = [  1.76782392,   2.1726818,  121.83892296,   0.18328725,  39.03185679, 88.67449832,  -0.31405664,  79.67387437]
ax2 = plt.subplot2grid(shape=(2,2), loc=(0,1), colspan=1)
# plotData(ax2, data, '-', 'Actual data')
modelRecord = generateLCAData(lcaWrapper, params)
# plotData(ax2, modelRecord, '--', 'No Loss Aversion')
plotR2(ax2, modelRecord, data)
ax2.set_title("No Loss Aversion")


# NO PREDECISIONAL BIAS
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, constantInput1, lossWeight, constantInput2: \
    lca(attributeProbabilities, getChoiceAttributes(gain, lossWeight * loss), getStartingActivation(0, threshold),
        decay, competition, (constantInput1, constantInput2),
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

params = [1.14149950e+00, 2.03757892e+00, 1.44904015e+02, 1.06872484e-02, 5.68171073e+01, 1.08937572e+02, 1.44172499e+00, 1.00102342e+02]
ax3 = plt.subplot2grid(shape=(2,2), loc=(1,0), colspan=1)
# plotData(ax3, data, '-', 'Actual data')
modelRecord = generateLCAData(lcaWrapper, params)
# plotData(ax3, modelRecord, '--', 'No Starting bias')
plotR2(ax3, modelRecord, data)
ax3.set_title("No Predecisional bias")


# NO FIXED UTILIY BIAS
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, constantInput, lossWeight, startingBias: \
    lca(attributeProbabilities, getChoiceAttributes(gain, lossWeight * loss),
        getStartingActivation(startingBias, threshold), decay, competition, constantInput,
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

params = [  2.88887094,   3.95190123, 129.26699843,   0.26548458,  31.52055003, 96.08235669,   1.3552764,   -0.33520937]
ax4 = plt.subplot2grid(shape=(2,2), loc=(1,1), colspan=1)
# plotData(ax4, data, '-', 'Actual data')
modelRecord = generateLCAData(lcaWrapper, params)
# plotData(ax4, modelRecord, '--', 'No Fixed Utility bias')
plotR2(ax4, modelRecord, data)
ax4.set_title("No Fixed Utility bias")


# plt.legend()

plt.tight_layout()
plt.savefig("fig/RTQuantileFitsR2.png", bbox_inches='tight')
plt.close()
