import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "..", "..", "src"))

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib

from LCA import PrepareRecurrentWeights, RunLCASimulation, GetValueInputZeroReference#, getValueInput2Attributes2Choices
from LUT import prepareStdNormalLUT, SampleFromLUT



QUANTILES = [0.1, 0.3, 0.5, 0.7, 0.9]
MARKERS = ['o', 's', 'D', 'v', '^']
COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

matplotlib.rcParams.update({'font.size': 14})

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

        ax.set_ylabel("RT quantiles")#, fontsize=14)
        ax.set_xlabel("P(accept)")#, fontsize=14)
        ax.set_ylim(0.5, 3.5)


data = np.genfromtxt("data_preprocessed.csv", delimiter=',')[1:, 2:]
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

numSimulationsPerCondition = 100

def generateLCAData(lcaWrapper, parameters):
    allModelData = np.zeros(3)
    for stakes in allStakes:
        gainValue, lossValue = stakes
        gainLossRatio = -1 * gainValue / lossValue
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


# FULL LCA MODEL
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, startingBias, lossWeight, constantInput1, constantInput2: \
    lca(attributeProbabilities, getChoiceAttributes(gain, lossWeight * loss),
        getStartingActivation(startingBias, threshold), decay, competition, (constantInput1, constantInput2),
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))


params = [1.74549528e+00,  3.17903030e+00,  2.80430931e+01,  2.14425490e-01, 2.56822918e+01, -3.71411918e-02,  1.74424428e+00,  6.61264367e+01, 6.93722980e+01]
ax1 = plt.subplot2grid(shape=(2,2), loc=(0,0), colspan=1)
modelRecord = generateLCAData(lcaWrapper, params)
plotQPF(ax1, dataWithRatios, 'scatter', '-', 'Empirical data')
plotQPF(ax1, modelRecord, 'plot', '--', 'LCA')
ax1.set_title("Full LCA")


# NO LOSS AVERSION
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, startingBias, constantInput1, constantInput2: \
    lca(attributeProbabilities, getChoiceAttributes(gain, loss), getStartingActivation(startingBias, threshold), decay,
        competition, (constantInput1, constantInput2),
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

params = [2.61577469e+00, 4.30567645e+00, 5.11963733e+01, 4.24286552e-01, 2.00665808e+01, 5.50324662e-03, 5.74473121e+01, 5.57126277e+01]
ax2 = plt.subplot2grid(shape=(2,2), loc=(0,1), colspan=1)
modelRecord = generateLCAData(lcaWrapper, params)
plotQPF(ax2, dataWithRatios, 'scatter', '-', 'Empirical data')
plotQPF(ax2, modelRecord, 'plot', '--', 'LCA')
ax2.set_title("No Loss Aversion")



# NO PREDECISIONAL BIAS
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, lossWeight, constantInput1, constantInput2: \
    lca(attributeProbabilities, getChoiceAttributes(gain, lossWeight * loss), getStartingActivation(0, threshold),
        decay, competition, (constantInput1, constantInput2),
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

params = [1.72537805,   3.14166275,  29.70051179,   0.45390954, 18.93316093,   1.86035748,  49.49073978,  52.2559686]
ax3 = plt.subplot2grid(shape=(2,2), loc=(1,0), colspan=1)
modelRecord = generateLCAData(lcaWrapper, params)
plotQPF(ax3, dataWithRatios, 'scatter', '-', 'Empirical data')
plotQPF(ax3, modelRecord, 'plot', '--', 'LCA')
ax3.set_title("No Predecisional Bias")


# NO FIXED UTILIY BIAS
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, startingBias, lossWeight, constantInput: \
    lca(attributeProbabilities, getChoiceAttributes(gain, lossWeight * loss),
        getStartingActivation(startingBias, threshold), decay, competition, constantInput,
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

params = [2.74790253e+00,  4.01386458e+00,  2.43103345e+01,  5.78575468e-01, 1.20616881e+01, -3.26133146e-02,  1.37598125e+00,  4.26439994e+01]
ax4 = plt.subplot2grid(shape=(2,2), loc=(1,1), colspan=1)
modelRecord = generateLCAData(lcaWrapper, params)
plotQPF(ax4, dataWithRatios, 'scatter', '-', 'Empirical data')
plotQPF(ax4, modelRecord, 'plot', '--', 'LCA')
ax4.set_title("No Fixed Utility Bias")


# plt.legend()
plt.tight_layout()
plt.savefig("QPF.png", bbox_inches='tight')
plt.close()