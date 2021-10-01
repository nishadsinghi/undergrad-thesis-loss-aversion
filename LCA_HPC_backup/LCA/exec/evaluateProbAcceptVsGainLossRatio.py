import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "..", "src"))

from LCA import PrepareRecurrentWeights, RunLCASimulation, GetValueInputZeroReference
from LUT import prepareStdNormalLUT, SampleFromLUT

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle
from sklearn.metrics import r2_score
import matplotlib


data = np.genfromtxt("../src/risk_data_cleaned.csv", delimiter=',')[1:, 1:]
# data = np.genfromtxt("../src/dataZeroSure_cleaned.csv", delimiter=',')[1:, 1:]
allStakes = np.unique(data[:, -2:], axis=0)

matplotlib.rcParams.update({'font.size': 22})


def filterFunction(tup):
    if tup[-1] != -1:
        return True
    else:
        return False


def plotData(ax, data, linestyle, label):
    # functions to help with data
    def selectConditionTrials(gain, loss):
        selectedData = data[data[:, 2] == gain][:]
        selectedData = selectedData[selectedData[:, 3] == loss][:]
        return selectedData

    def computePData(gain, loss):
        conditionTrialData = selectConditionTrials(gain, loss)
        responses = conditionTrialData[:, 0]
        return np.mean(responses)

    ActualRatioAndProbRecord = np.zeros(2)
    for stakes in allStakes:
        gainValue, lossValue = stakes
        gainLossRatio = -1 * gainValue / lossValue
        probAccept = computePData(gainValue, lossValue)
        ActualRatioAndProbRecord = np.vstack((ActualRatioAndProbRecord, (gainLossRatio, probAccept)))

    ActualRatioAndProbRecord = ActualRatioAndProbRecord[1:, :]
    df = pd.DataFrame(ActualRatioAndProbRecord)
    ratioCombinedDf = df.groupby(0).mean()
    ratioCombinedDf = ratioCombinedDf.rename(columns={1: label})
    ratioCombinedDf.plot(ax=ax, logx=True, linestyle=linestyle)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Gain/Loss")
    ax.set_ylabel("P(Accept Gamble)")


# simulation results
numSimulationsPerCondition = 100

def generateLCAData(lcaWrapper, params):
    modelRecord = np.zeros(4)
    for stakes in allStakes:
        gainValue, lossValue = stakes
        gainLossRatio = -1 * gainValue / lossValue
        allSimulations = [lcaWrapper(gainValue, lossValue, *params) for _ in
                          range(numSimulationsPerCondition)]
        allValidResponseSimulations = list(filter(filterFunction, allSimulations))
        _, allModelRTs, allModelResponses = zip(*allValidResponseSimulations)
        P_model = np.mean(allModelResponses)
        modelRecord = np.vstack((modelRecord, np.hstack((gainLossRatio, P_model, stakes))))

    modelRecord = modelRecord[1:, :]

    return modelRecord

def plotLCAData(ax, modelRecord, label):
    modelRecord = modelRecord[:, 0:2]
    dfSim = pd.DataFrame(modelRecord)
    ratioCombinedDfSim = dfSim.groupby(0).mean()
    ratioCombinedDfSim = ratioCombinedDfSim.rename(columns={1: label})
    ratioCombinedDfSim.plot(ax=ax, logx=True, linestyle='dashed')
    ax.set_xlabel("Gain/Loss")
    ax.set_ylabel("P(Accept Gamble)")
    # ax.set_title(label, fontsize=16)


def plotR2(ax, modelRecord, data):
    def selectConditionTrials(gain, loss):
        selectedData = data[data[:, 2] == gain][:]
        selectedData = selectedData[selectedData[:, 3] == loss][:]
        return selectedData

    def computePData(gain, loss):
        conditionTrialData = selectConditionTrials(gain, loss)
        responses = conditionTrialData[:, 0]
        return np.mean(responses)

    allPModel = []
    allPActual = []
    for stakes in allStakes:
        gain, loss = stakes
        modelTrialsForThisStakes = modelRecord[np.logical_and(modelRecord[:, 2] == gain, modelRecord[:, 3] == loss)]
        P_model = np.mean(modelTrialsForThisStakes[:, 1])
        allPModel.append(P_model)

        P_actual = computePData(gain, loss)
        allPActual.append(P_actual)

    print('number of stakes: ', len(allPActual), len(allPModel))
    r2Score = r2_score(allPModel, allPActual)

    ax.scatter(allPActual, allPModel)
    ax.plot((0, 1), (0, 1), linestyle='--', color='r')
    ax.annotate("R2 Score: %.2f" % r2Score, (0.25, 0.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Observed P(Accept)")
    ax.set_ylabel("model P(Accept)")



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


lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, startingBias, constantInput: \
    lca(attributeProbabilities, getChoiceAttributes(gain, loss), getStartingActivation(startingBias, threshold), decay, competition, constantInput,
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))


params = [1.77895768e+00,  3.20906638e+00,  9.58128213e+01,  1.96114120e-01, 3.40131479e+01, -3.41666518e-01,  7.71126655e+01]
# params = [2.06155217e+00,  3.18317118e+00,  2.08697307e+01,  1.27605889e-01, 1.04339815e+01, -1.95599890e-01,  3.01482276e+01]
ax1 = plt.subplot2grid(shape=(1,1), loc=(0,0), colspan=1)
plotData(ax1, data, '-', 'Actual data')
modelRecord = generateLCAData(lcaWrapper, params)
plotLCAData(ax1, modelRecord, 'LCA')

# ax2 = plt.subplot2grid(shape=(1,2), loc=(0,1), colspan=1)
# plotR2(ax2, modelRecord, data)
#
plt.savefig('choiceFitsDataset1.png', bbox_inches='tight')
plt.close()

plt.show()


exit()



# # LOSS ATTENTION
# startingActivation = (0, 0)
# getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
# getAllThresholds = lambda threshold: [threshold]*numAccumulators
#
# lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, constantInput, attribute1Prob: \
#     lca((attribute1Prob, 1-attribute1Prob), getChoiceAttributes(gain, loss), startingActivation, decay, competition, constantInput,
#         noiseStdDev, nonDecisionTime, getAllThresholds(threshold))
#
# params = [1.42469771,  2.3423496 , 17.30099185,  0.12683208,  7.85291344, 17.59723663,  0.26822441]
# ax4 = plt.subplot2grid(shape=(1,3), loc=(0,1), colspan=1)
# plotData(ax4, data, '-', 'Actual data')
# plotLCAData(ax4, lcaWrapper, params, 'Unequal attention')
#
#
# # CONSTANT OFFSET
# attributeProbabilities = (0.5, 0.5)
# startingActivation = (0, 0)
# getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
# getAllThresholds = lambda threshold: [threshold]*numAccumulators
#
# lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, constantInput1, constantInput2: \
#     lca(attributeProbabilities, getChoiceAttributes(gain, loss), startingActivation, decay, competition,
#         (constantInput1, constantInput2), noiseStdDev, nonDecisionTime, getAllThresholds(threshold))
#
# params = [9.55861851e-01,  2.39165361e+00,  2.25704739e+01,  1.37531578e-02, 1.45696354e+01, -8.90549815e-02,  3.26613632e+01]
# ax5 = plt.subplot2grid(shape=(1,3), loc=(0,2), colspan=1)
# plotData(ax5, data, '-', 'Actual data')
# plotLCAData(ax5, lcaWrapper, params, 'Unequal attention')
#
#

# # UNEUQAL THRESHOLDS
# attributeProbabilities = (0.5, 0.5)
# startingActivation = (0, 0)
#
# getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
# # getAllThresholds = lambda threshold: [threshold]*numAccumulators
# lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold1, threshold2, constantInput: \
#     lca(attributeProbabilities, getChoiceAttributes(gain, loss), startingActivation, decay, competition, constantInput,
#         noiseStdDev, nonDecisionTime, [threshold1, threshold2])
#
# params = [2.9222192 ,  1.97753516, 17.59714645,  0.18920589,  7.08568937, 8.62839634, 28.39363504]
# ax1 = plt.subplot2grid(shape=(2,6), loc=(0, 0), colspan=3)
# plotData(ax1, data, '-', 'Actual data')
# plotLCAData(ax1, lcaWrapper, params, 'Unequal thresholds')
#
# # STARTING BIAS
# def getStartingActivation(startingBias, threshold):
#     if startingBias < 0:
#         return [-1*startingBias*threshold, 0]
#     else:
#         return [0, startingBias*threshold]
#
# getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
# getAllThresholds = lambda threshold: [threshold]*numAccumulators
# lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, startingBias, constantInput: \
#     lca(attributeProbabilities, getChoiceAttributes(gain, loss), getStartingActivation(startingBias, threshold), decay, competition, constantInput,
#         noiseStdDev, nonDecisionTime, getAllThresholds(threshold))
#
# params = [2.06155217e+00,  3.18317118e+00,  2.08697307e+01,  1.27605889e-01, 1.04339815e+01, -1.95599890e-01,  3.01482276e+01]
# ax2 = plt.subplot2grid(shape=(2,6), loc=(0, 3), colspan=3)
# plotData(ax2, data, '-', 'Actual data')
# plotLCAData(ax2, lcaWrapper, params, 'Unequal starting points')

# # plt.show()
