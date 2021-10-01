import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "..", "..", "..", "src"))

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
from sklearn.metrics import mean_absolute_error as MAE

experimentalData = np.genfromtxt("data_preprocessed_250_zScore3.csv", delimiter=',')[:, 1:]
allStakes = np.unique(experimentalData[:, -2:], axis=0)

fig = plt.figure(figsize=(10, 10))
# fig.suptitle("Choice vs. Choice-factor Adjusted RT - Dataset 2")

sns.set(font_scale=1.5)


# def mean_confidence_interval(data, confidence=0.95):
#     a = 1.0 * np.array(data)
#     n = len(a)
#     m, se = np.mean(a), stats.sem(a)
#     h = se * stats.t.ppf((1 + confidence) / 2., n-1)
#     return h

exp_allLogRT = np.log(experimentalData[:, 1]).reshape(-1, 1)
exp_allGainLoss = experimentalData[:, 2:]
exp_regressor = LinearRegression()
exp_regressor.fit(exp_allGainLoss, exp_allLogRT)
exp_allPredictedLogRT = exp_regressor.predict(exp_allGainLoss)
exp_allLogRTResidual = exp_allLogRT - exp_allPredictedLogRT

exp_responses = experimentalData[:, 0]
_, exp_sortedResponses = (list(t) for t in zip(*sorted(zip(exp_allLogRTResidual.tolist(), exp_responses.tolist()))))

exp_binnedResponses = np.array_split(exp_sortedResponses, 5)
exp_binPAccept = [np.mean(binResponses) for binResponses in exp_binnedResponses]


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

    if linestyle == '--':
        mae = MAE(binPAccept, exp_binPAccept)
        ax.annotate("MAE = %.3f" % mae, (1, 0.9))

    # CI = [mean_confidence_interval(binResponses) for binResponses in binnedResponses]

    plt.plot((1, 2, 3, 4, 5), binPAccept, marker='o', label=label, linestyle=linestyle)
    ax.set_xticks((1, 2, 3, 4, 5))
    # ax.set_ylabel("P(accept)")
    # ax.set_xlabel("Choice-factor adjusted RT")
    ax.set_ylim(0, 1)

                                                    # SIMULATION
numSimulationsPerCondition = 200

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

params = [9.23846699e-01,  2.67337343e+00,  1.20874040e+01,  1.76918681e-01, 1.86583204e+01,  4.19019503e+01,  1.13950931e+00, -1.95275769e-02, 4.29881239e+01]
modelSimulationData = generateModelData(lcaWrapper, params)
ax1 = plt.subplot2grid(shape=(2,2), loc=(0,0), colspan=1)
plotData(ax1, modelSimulationData, 'Model', '--')
plotData(ax1, experimentalData, 'Observed', '-')
ax1.set_title("Full LCA Model")



# NO LOSS AVERSION
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, constantInput1, startingBias, constantInput2: \
    lca(attributeProbabilities, getChoiceAttributes(gain, loss), getStartingActivation(startingBias, threshold), decay,
        competition, (constantInput1, constantInput2),
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

params = [ 2.98204048,  4.56211096, 13.53315285,  0.50794486,  7.61376135, 29.59920162, -0.19561436, 31.23151038]
modelSimulationData = generateModelData(lcaWrapper, params)
ax2 = plt.subplot2grid(shape=(2,2), loc=(0,1), colspan=1)
plotData(ax2, modelSimulationData, 'Model', '--')
plotData(ax2, experimentalData, 'Observed', '-')
ax2.set_title("No Loss Aversion")



# NO PREDECISIONAL BIAS
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, constantInput1, lossWeight, constantInput2: \
    lca(attributeProbabilities, getChoiceAttributes(gain, lossWeight * loss), getStartingActivation(0, threshold),
        decay, competition, (constantInput1, constantInput2),
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

params = [ 3.43372772,  4.83008857, 16.23918899,  0.40279549,  9.28365665, 42.13510508, 1.16084044, 42.6948477 ]
modelSimulationData = generateModelData(lcaWrapper, params)

ax3 = plt.subplot2grid(shape=(2,2), loc=(1,0), colspan=1)
plotData(ax3, modelSimulationData, 'Model', '--')
plotData(ax3, experimentalData, 'Observed', '-')
ax3.set_title("No Predecisional Bias")




# NO FIXED UTILIY BIAS
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, constantInput, lossWeight, startingBias: \
    lca(attributeProbabilities, getChoiceAttributes(gain, lossWeight * loss),
        getStartingActivation(startingBias, threshold), decay, competition, constantInput,
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

params = [ 3.00517563,  4.31964694, 11.7902401,   0.50017349,  4.38374723, 14.58909752, 1.04120538, -0.17416005]
modelSimulationData = generateModelData(lcaWrapper, params)
ax4 = plt.subplot2grid(shape=(2,2), loc=(1,1), colspan=1)
plotData(ax4, modelSimulationData, 'Model', '--')
plotData(ax4, experimentalData, 'Observed', '-')
ax4.set_title("No Fixed Utility Bias")


plt.xticks((1, 2, 3, 4, 5))
plt.ylim(0, 1)

handles, labels = ax4.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', ncol=2)

fig.add_subplot(111, frameon=False).grid(False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel("Choice-factor Adjusted RT Bins")
plt.ylabel("P(accept)")


fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("fig/LCABehaviouralMarkers.png", bbox_inches='tight')
plt.close()





