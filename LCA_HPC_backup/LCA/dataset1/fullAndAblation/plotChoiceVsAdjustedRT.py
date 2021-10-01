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
import seaborn as sns

from LCA import PrepareRecurrentWeights, RunLCASimulation, GetValueInputZeroReference, getValueInput2Attributes2Choices
from LUT import prepareStdNormalLUT, SampleFromLUT

experimentalData = np.genfromtxt("risk_data_cleaned.csv", delimiter=',')[:, 1:]
allStakes = np.unique(experimentalData[:, -2:], axis=0)

fig = plt.figure(figsize=(7, 7))
fig.suptitle("Choice vs. Choice-factor Adjusted RT - Dataset 1")

sns.set()

# def mean_confidence_interval(data, confidence=0.95):
#     a = 1.0 * np.array(data)
#     n = len(a)
#     m, se = np.mean(a), stats.sem(a)
#     h = se * stats.t.ppf((1 + confidence) / 2., n-1)
#     return h

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
    # CI = [mean_confidence_interval(binResponses) for binResponses in binnedResponses]

    plt.plot((1, 2, 3, 4, 5), binPAccept, marker='o', label=label, linestyle=linestyle)
    ax.set_xticks((1, 2, 3, 4, 5))
    ax.set_ylabel("P(accept)")
    ax.set_xlabel("Choice-factor adjusted RT")
    ax.set_ylim(0, 1)
    ax.legend()

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
modelSimulationData = generateModelData(lcaWrapper, params)
ax1 = plt.subplot2grid(shape=(2,2), loc=(0,0), colspan=1)
plotData(ax1, experimentalData, 'Actual data', '-')
plotData(ax1, modelSimulationData, 'LCA', '--')
ax1.set_title("Full LCA Model")



# NO LOSS AVERSION
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, constantInput1, startingBias, constantInput2: \
    lca(attributeProbabilities, getChoiceAttributes(gain, loss), getStartingActivation(startingBias, threshold), decay,
        competition, (constantInput1, constantInput2),
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

params = [  1.76782392,   2.1726818,  121.83892296,   0.18328725,  39.03185679, 88.67449832,  -0.31405664,  79.67387437]
modelSimulationData = generateModelData(lcaWrapper, params)
ax2 = plt.subplot2grid(shape=(2,2), loc=(0,1), colspan=1)
plotData(ax2, experimentalData, 'Actual data', '-')
plotData(ax2, modelSimulationData, 'LCA', '--')
ax2.set_title("No Loss Aversion")



# NO PREDECISIONAL BIAS
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, constantInput1, lossWeight, constantInput2: \
    lca(attributeProbabilities, getChoiceAttributes(gain, lossWeight * loss), getStartingActivation(0, threshold),
        decay, competition, (constantInput1, constantInput2),
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

params = [1.14149950e+00, 2.03757892e+00, 1.44904015e+02, 1.06872484e-02, 5.68171073e+01, 1.08937572e+02, 1.44172499e+00, 1.00102342e+02]
modelSimulationData = generateModelData(lcaWrapper, params)

ax3 = plt.subplot2grid(shape=(2,2), loc=(1,0), colspan=1)
plotData(ax3, experimentalData, 'Actual data', '-')
plotData(ax3, modelSimulationData, 'LCA', '--')
ax3.set_title("No Predecisional bias")




# NO FIXED UTILIY BIAS
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, constantInput, lossWeight, startingBias: \
    lca(attributeProbabilities, getChoiceAttributes(gain, lossWeight * loss),
        getStartingActivation(startingBias, threshold), decay, competition, constantInput,
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

params = [  2.88887094,   3.95190123, 129.26699843,   0.26548458,  31.52055003, 96.08235669,   1.3552764,   -0.33520937]
modelSimulationData = generateModelData(lcaWrapper, params)
ax4 = plt.subplot2grid(shape=(2,2), loc=(1,1), colspan=1)
plotData(ax4, experimentalData, 'Actual data', '-')
plotData(ax4, modelSimulationData, 'LCA', '--')
ax4.set_title("No Fixed Utility bias")


plt.xticks((1, 2, 3, 4, 5))
plt.ylim(0, 1)

plt.legend()

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("fig/LCABehaviouralMarkers.png", bbox_inches='tight')
plt.close()





