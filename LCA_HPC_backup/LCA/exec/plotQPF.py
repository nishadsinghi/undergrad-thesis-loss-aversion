import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "..", "src"))

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from LCA import PrepareRecurrentWeights, RunLCASimulation, GetValueInputZeroReference#, getValueInput2Attributes2Choices
from LUT import prepareStdNormalLUT, SampleFromLUT



QUANTILES = [0.1, 0.3, 0.5, 0.7, 0.9]
MARKERS = ['o', 's', 'D', 'v', '^']
COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']



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

        ax.set_ylabel("RT quantiles", fontsize=16)
        ax.set_xlabel("P(accept)", fontsize=16)
        ax.set_ylim(0.5, 3.75)


# data = np.genfromtxt("../src/risk_data_cleaned.csv", delimiter=',')[1:, 1:]
data = np.genfromtxt("../dataZeroSure_cleaned.csv", delimiter=',')[1:, 1:]
allStakes = np.unique(data[:, -2:], axis=0)
computeRatio = lambda trial: -1*data[trial, 2]/data[trial, 3]
allRatios = [computeRatio(trial) for trial in range(np.shape(data)[0])]
dataWithRatios = np.hstack((data[:, 0:2], np.reshape(allRatios, (-1, 1))))

# plotQPF(dataWithRatios, 'scatter', '--', 'empirical data')


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

maxTimeSteps = 1000
deltaT = 0.02
prepareRecurrentWeights = PrepareRecurrentWeights(numAccumulators)
lca = RunLCASimulation(getValueInput, sampleFromZeroMeanLUT, prepareRecurrentWeights, maxTimeSteps, deltaT)


# LOSS AVERSION
attributeProbabilities = (0.5, 0.5)
startingActivation = (0, 0)
getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
getAllThresholds = lambda threshold: [threshold]*numAccumulators

lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, lossWeight, constantInput: \
    lca(attributeProbabilities, getChoiceAttributes(gain, lossWeight*loss), startingActivation, decay, competition, constantInput,
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

# params = [2.97553172,   4.76614439, 148.09895125,   0.19744745, 33.2914632 , 101.92218251,   2.17809482]
params = [3.70756359,  3.45232175, 22.60422262,  0.28049256,  7.20611797, 2.17565966, 31.180381]
lossAversionData = generateLCAData(lcaWrapper, params)
ax1 = plt.subplot2grid(shape=(1,1), loc=(0,0), colspan=1)
plotQPF(ax1, dataWithRatios, 'scatter', '-', 'Empirical data')
plotQPF(ax1, lossAversionData, 'plot', '--', 'LCA')
ax1.set_title("Unequal weights", fontsize=16)

plt.savefig('dataset2OnlyLossAversionQPF.png', bbox_inches='tight')
plt.close()
exit()


# LOSS ATTENTION
startingActivation = (0, 0)
getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
getAllThresholds = lambda threshold: [threshold]*numAccumulators

lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, constantInput, attribute1Prob: \
    lca((attribute1Prob, 1-attribute1Prob), getChoiceAttributes(gain, loss), startingActivation, decay, competition, constantInput,
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

# params = [1.55893187e+00, 2.59678066e+00, 1.36329544e+02, 1.25511453e-01, 4.71199432e+01, 1.11088120e+02, 3.57679855e-01]
params = [1.42469771,  2.3423496 , 17.30099185,  0.12683208,  7.85291344, 17.59723663,  0.26822441]
lossAttentionData = generateLCAData(lcaWrapper, params)
ax2 = plt.subplot2grid(shape=(1,3), loc=(0,1), colspan=1)
plotQPF(ax2, dataWithRatios, 'scatter', '-', 'Empirical data')
plotQPF(ax2, lossAttentionData, 'plot', '--', 'LCA')
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
constantOffsetData = generateLCAData(lcaWrapper, params)
ax3 = plt.subplot2grid(shape=(1,3), loc=(0,2), colspan=1)
plotQPF(ax3, dataWithRatios, 'scatter', '-', 'Empirical data')
plotQPF(ax3, constantOffsetData, 'plot', '--', 'LCA')
ax3.set_title("Constant Offset", fontsize=16)

plt.show()
exit()















# UNEQUAL THRESHOLDS
attributeProbabilities = (0.5, 0.5)
startingActivation = (0, 0)

getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
# getAllThresholds = lambda threshold: [threshold]*numAccumulators
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold1, threshold2, constantInput: \
    lca(attributeProbabilities, getChoiceAttributes(gain, loss), startingActivation, decay, competition, constantInput,
        noiseStdDev, nonDecisionTime, [threshold1, threshold2])

params = [2.9222192 ,  1.97753516, 17.59714645,  0.18920589,  7.08568937, 8.62839634, 28.39363504]
unequalThresholdsData = generateLCAData(lcaWrapper, params)
ax1 = plt.subplot2grid(shape=(1,2), loc=(0,0), colspan=1)
plotQPF(ax1, dataWithRatios, 'scatter', '-', 'Empirical data')
plotQPF(ax1, unequalThresholdsData, 'plot', '--', 'LCA')
ax1.set_title("LCA with unequal thresholds", fontsize=16)


# UNEQUAL STARTING POINTS
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

params = [2.06155217e+00,  3.18317118e+00,  2.08697307e+01,  1.27605889e-01, 1.04339815e+01, -1.95599890e-01,  3.01482276e+01]
unequalStartingData = generateLCAData(lcaWrapper, params)
ax2 = plt.subplot2grid(shape=(1,2), loc=(0,1), colspan=1)
plotQPF(ax2, dataWithRatios, 'scatter', '-', 'Empirical data')
plotQPF(ax2, unequalStartingData, 'plot', '--', 'LCA')
ax2.set_title("LCA with unequal starting point", fontsize=16)

plt.show()
