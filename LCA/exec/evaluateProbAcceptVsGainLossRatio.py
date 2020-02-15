import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "..", "src"))

from LCA import PrepareRecurrentWeights, RunLCASimulation, GetValueInputZeroReference#, getValueInput2Attributes2Choices
from LUT import prepareStdNormalLUT, SampleFromLUT

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle

# functions to help with data
def selectConditionTrials(gain, loss):
    selectedData = data[data[:, 2] == gain][:]
    selectedData = selectedData[selectedData[:, 3] == loss][:]
    return selectedData

def computePData(gain, loss):
    conditionTrialData = selectConditionTrials(gain, loss)
    responses = conditionTrialData[:, 0]
    return np.mean(responses)

def filterFunction(tup):
    if tup[-1] != -1:
        return True
    else:
        return False

allGainValues = list(range(10, 110, 10))
allLossValues = list(range(-100, 0, 10))

# plot observed data
data = np.genfromtxt("../src/risk_data.csv", delimiter=',')[1:, 1:]

ActualRatioAndProbRecord = np.zeros(2)
for gainValue in allGainValues:
    for lossValue in allLossValues:
        gainLossRatio = -1 * gainValue / lossValue
        print("Ratio: ", gainLossRatio)
        probAccept = computePData(gainValue, lossValue)
        ActualRatioAndProbRecord = np.vstack((ActualRatioAndProbRecord, (gainLossRatio, probAccept)))

ActualRatioAndProbRecord = ActualRatioAndProbRecord[1:, :]
df = pd.DataFrame(ActualRatioAndProbRecord)
ratioCombinedDf = df.groupby(0).mean()
ratioCombinedDf = ratioCombinedDf.rename(columns={1: 'Actual Probability'})
ax = ratioCombinedDf.plot(logx=True)

# simulation results
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
deltaT = 0.01
prepareRecurrentWeights = PrepareRecurrentWeights(numAccumulators)
lca = RunLCASimulation(getValueInput, sampleFromZeroMeanLUT, prepareRecurrentWeights, maxTimeSteps, deltaT)
# lca = RunLCASimulation(getValueInput2Attributes2Choices, sampleFromZeroMeanLUT, prepareRecurrentWeights, maxTimeSteps, deltaT)

attributeProbabilities = (0.5, 0.5)
startingActivation = (0, 0)

getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
getAllThresholds = lambda threshold: [threshold]*numAccumulators
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold1, threshold2, constantInput: \
    lca(attributeProbabilities, getChoiceAttributes(gain, loss), startingActivation, decay, competition, constantInput,
        noiseStdDev, nonDecisionTime, [threshold1, threshold2])

for metropolisRun in range(1, 2):
    numSimulationsPerCondition = 200
    paramFile = open('../data/LCAZeroRefUnequalThresholds.pickle', 'rb')
    params = pickle.load(paramFile)
    params = (params[-1, :])#[0]
    print("PARAMS = ", params)

    modelRecord = np.zeros(2)
    for gainValue in allGainValues:
        for lossValue in allLossValues:
            gainLossRatio = -1 * gainValue / lossValue
            allSimulations = [lcaWrapper(gainValue, lossValue, *params) for _ in
                              range(numSimulationsPerCondition)]
            allValidResponseSimulations = list(filter(filterFunction, allSimulations))
            _, allModelRTs, allModelResponses = zip(*allValidResponseSimulations)
            P_model = np.mean(allModelResponses)
            modelRecord = np.vstack((modelRecord, (gainLossRatio, P_model)))

    modelRecord = modelRecord[1:, :]
    dfSim = pd.DataFrame(modelRecord)
    ratioCombinedDfSim = dfSim.groupby(0).mean()
    ratioCombinedDfSim = ratioCombinedDfSim.rename(columns={1: 'Simulation {}'.format(metropolisRun)})
    ratioCombinedDfSim.plot(ax=ax, logx=True, linestyle='dashed')


plt.xlabel("Gain/Loss")
plt.ylabel("P(Accept Gamble)")
# plt.title("2000 Swaps")
plt.show()
