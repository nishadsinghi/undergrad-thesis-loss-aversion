import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "..", "..", "src"))

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib

from LCA import PrepareRecurrentWeights, RunLCASimulation, GetValueInputZeroReference
from LUT import prepareStdNormalLUT, SampleFromLUT



QUANTILES = [0.1, 0.3, 0.5, 0.7, 0.9]
MARKERS = ['o', 's', 'D', 'v', '^']
COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

sns.set(font_scale=1.75)
fig = plt.figure(figsize=(12, 12))
fig.suptitle("Quantile Probability Functions - Dataset 2")
# matplotlib.rcParams.update({'font.size': 14})

def plotQPF(ax, dataWithRatios, plotType, linestyle, label):
    sortedDataWithRatios = dataWithRatios[dataWithRatios[:,-1].argsort()]
    split = np.array_split(sortedDataWithRatios, 8, axis=0)

    for quantile, marker, color in zip(QUANTILES, MARKERS, COLORS):
        def computeQuantiles(data):
            reactionTimes = data[:, 1].flatten()
            reactionTimeForQuantile = np.quantile(reactionTimes, quantile)
            return reactionTimeForQuantile

        def computeP(data):
            choices = data[:, 0].flatten()
            P_mean = np.mean(choices)
            return P_mean

        toPlotX = [computeP(_) for _ in split]
        toPlotY = [computeQuantiles(_) for _ in split]

        if plotType == 'plot':
            ax.plot(toPlotX, toPlotY, color=color, linestyle=linestyle, label=label)
        elif plotType == 'scatter':
            ax.scatter(toPlotX, toPlotY, marker=marker, color=color)

        ax.set_ylabel("RT Quantiles (s)")#, fontsize=16)
        ax.set_xlabel("P(accept)")#, fontsize=16)
        ax.set_ylim(0, 3.5)
        ax.set_xlim(0, 1)


data = np.genfromtxt("dataZeroSure_cleaned_250_10500_zScore3.csv", delimiter=',')[:, 1:]
allStakes = np.unique(data[:, -2:], axis=0)
computeRatio = lambda trial: -1*data[trial, 2]/data[trial, 3]
allRatios = [computeRatio(trial) for trial in range(np.shape(data)[0])]
dataWithRatios = np.hstack((data[:, 0:2], np.reshape(allRatios, (-1, 1))))

    # LCA
def filterFunction(tup):
    if tup[-1] != -1:
        return True
    else:
        return False

numSimulationsPerCondition = 150

def generateModelData(lcaWrapper, parameters):
    allModelData = np.zeros(3)
    for stakes in allStakes:
        gainValue, lossValue = stakes
        gainLossRatio = -1 * gainValue / lossValue
        print("Ratio: ", gainLossRatio)
        allSimulations = [lcaWrapper(gainValue, lossValue, *parameters) for _ in
                          range(numSimulationsPerCondition)]
        allValidResponseSimulations = list(filter(filterFunction, allSimulations))
        numValidResponses = len(allValidResponseSimulations)
        allActivations, allModelRTs, allModelResponses = zip(*allValidResponseSimulations)
        modelStakes = np.full((numValidResponses, 1), gainLossRatio)
        modelDataForStakes = np.hstack(
            (np.array(allModelResponses).reshape(-1, 1), np.array(allModelRTs).reshape(-1, 1), modelStakes))
        allModelData = np.vstack((allModelData, modelDataForStakes))

    allModelData = allModelData[1:, :]
    return allModelData


# set up look-up table (LUT)
LUTInterval = 0.0001
numAccumulators = 2
stdNormalLUT = prepareStdNormalLUT(LUTInterval)
sampleFromLUT = SampleFromLUT(stdNormalLUT)
sampleFromZeroMeanLUT = lambda stdDev: sampleFromLUT(0, stdDev, numAccumulators)

# set-up the LCA
identityUtilityFunction = lambda x: x
getValueInput = GetValueInputZeroReference(identityUtilityFunction)

maxTimeSteps = 750
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

lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, constantInput1, lossWeight, startingBias, constantInput2: lca(attributeProbabilities, getChoiceAttributes(gain, lossWeight * loss), getStartingActivation(startingBias, threshold), decay, competition, (constantInput1, constantInput2), noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

params = [ 1.50707942,  1.94389123, 33.4929678,   0.17683991, 10.739692,   20.74454472, 3.34318026, -0.14792952, 22.54369298]
modelSimulationData = generateModelData(lcaWrapper, params)

ax1 = plt.subplot2grid(shape=(2,2), loc=(0,0), colspan=1)
plotQPF(ax1, dataWithRatios, 'scatter', '-', 'Empirical data')
plotQPF(ax1, modelSimulationData, 'plot', '--', 'LCA')
ax1.set_title("Full Model")


# NO LOSS AVERSION
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, constantInput1, startingBias, constantInput2: \
    lca(attributeProbabilities, getChoiceAttributes(gain, loss), getStartingActivation(startingBias, threshold), decay,
        competition, (constantInput1, constantInput2),
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

params = [ 1.64080433,  1.66245883, 25.74527386,  0.06088753, 16.13667531, 43.0831163, -0.11187623, 41.54052658]

modelSimulationData = generateModelData(lcaWrapper, params)
ax2 = plt.subplot2grid(shape=(2,2), loc=(0,1), colspan=1)
plotQPF(ax2, dataWithRatios, 'scatter', '-', 'Empirical data')
plotQPF(ax2, modelSimulationData, 'plot', '--', 'LCA')
ax2.set_title("No Loss Aversion")


# NO STARTING BIAS
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, constantInput1, lossWeight, constantInput2: \
    lca(attributeProbabilities, getChoiceAttributes(gain, lossWeight * loss), getStartingActivation(0, threshold),
        decay, competition, (constantInput1, constantInput2),
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

params = [ 1.46429413,  2.93651752, 27.16307418,  0.18148277,  8.02916648, 14.79609182, 4.19878174, 17.05312185]
modelSimulationData = generateModelData(lcaWrapper, params)
ax3 = plt.subplot2grid(shape=(2,2), loc=(1,0), colspan=1)
plotQPF(ax3, dataWithRatios, 'scatter', '-', 'Empirical data')
plotQPF(ax3, modelSimulationData, 'plot', '--', 'LCA')
ax3.set_title("No Predecisional bias")


# NO FIXED UTILITY BIAS
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, constantInput, lossWeight, startingBias: \
    lca(attributeProbabilities, getChoiceAttributes(gain, lossWeight * loss),
        getStartingActivation(startingBias, threshold), decay, competition, constantInput,
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

params = [ 1.61874945,  3.87858521, 40.14551712,  0.1246042,  10.67892178, 20.03839262, 2.80531416, -0.07801095]
modelSimulationData = generateModelData(lcaWrapper, params)
ax4 = plt.subplot2grid(shape=(2,2), loc=(1,1), colspan=1)
plotQPF(ax4, dataWithRatios, 'scatter', '-', 'Empirical data')
plotQPF(ax4, modelSimulationData, 'plot', '--', 'LCA')
ax4.set_title("No Fixed Utility bias")



# plt.legend()
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("fig/QPF.png", bbox_inches='tight')
plt.close()
