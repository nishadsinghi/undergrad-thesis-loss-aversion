import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "..", "src"))

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle

from LCA import PrepareRecurrentWeights, RunLCASimulation, GetValueInput
from LUT import prepareStdNormalLUT, SampleFromLUT

data = np.genfromtxt("../src/risk_data.csv", delimiter=',')#[1:, 1:]
# data = data[np.where(data[:, 0] == 3)]
data = data[1:, 1:]

def selectConditionTrials(gain, loss):
    selectedData = data[data[:, 2] == gain][:]
    selectedData = selectedData[selectedData[:, 3] == loss][:]
    return selectedData

def computeRTData(gain, loss):
    conditionTrialData = selectConditionTrials(gain, loss)
    responses = conditionTrialData[:, 1]
    return np.mean(responses)

def filterFunction(tup):
    if tup[-1] != -1:
        return True
    else:
        return False

ratioAndRTRecord = np.zeros(2)

allGainValues = list(range(10, 110, 10))
allLossValues = list(range(-100, 0, 10))

for gainValue in allGainValues:
    for lossValue in allLossValues:
        gainLossRatio = -1 * gainValue / lossValue
        print("Ratio: ", gainLossRatio)
        meanRT = computeRTData(gainValue, lossValue)
        ratioAndRTRecord = np.vstack((ratioAndRTRecord, (gainLossRatio, meanRT)))

ratioAndRTRecord = ratioAndRTRecord[1:, :]
df = pd.DataFrame(ratioAndRTRecord)
ratioCombinedDf = df.groupby(0).mean()
ratioCombinedDf = ratioCombinedDf.rename(columns={1: 'Mean RT'})
ax = ratioCombinedDf.plot(logx=True)

                                                        # LCA
# set up look-up table (LUT)
LUTInterval = 0.0001
numAccumulators = 2
stdNormalLUT = prepareStdNormalLUT(LUTInterval)
sampleFromLUT = SampleFromLUT(stdNormalLUT)
sampleFromZeroMeanLUT = lambda stdDev: sampleFromLUT(0, stdDev, numAccumulators)

# set-up the LCA
identityUtilityFunction = lambda x: x
getValueInput = GetValueInput(identityUtilityFunction)

maxTimeSteps = 750
deltaT = 0.01
prepareRecurrentWeights = PrepareRecurrentWeights(numAccumulators)
lca = RunLCASimulation(getValueInput, sampleFromZeroMeanLUT, prepareRecurrentWeights, maxTimeSteps, deltaT)

attributeProbabilities = (0.5, 0.5)
startingActivation = (0, 0)

getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
getAllThresholds = lambda threshold: [threshold]*numAccumulators
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, constantInput: \
    lca(attributeProbabilities, getChoiceAttributes(gain, loss), startingActivation, decay, competition, constantInput,
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

numSimulationsPerCondition = 200
paramFile = open('../data/newAttempt.pickle', 'rb')
params = pickle.load(paramFile)
params = (params[30:31, :])[0]

print("PARAMS = ", params)

modelRecord = np.zeros(2)
for gainValue in allGainValues:
    for lossValue in allLossValues:
        gainLossRatio = -1 * gainValue / lossValue
        allSimulations = [lcaWrapper(gainValue, lossValue, *params) for _ in
                          range(numSimulationsPerCondition)]
        allValidResponseSimulations = list(filter(filterFunction, allSimulations))
        _, allModelRTs, allModelResponses = zip(*allValidResponseSimulations)
        RT_model = np.mean(allModelRTs)
        modelRecord = np.vstack((modelRecord, (gainLossRatio, RT_model)))

modelRecord = modelRecord[1:, :]
dfSim = pd.DataFrame(modelRecord)
ratioCombinedDfSim = dfSim.groupby(0).mean()
ratioCombinedDfSim = ratioCombinedDfSim.rename(columns={1: 'Simulation RT'})
ratioCombinedDfSim.plot(ax=ax, logx=True)

ax.set_xlabel("Gain/Loss")
ax.set_ylabel("Mean RT")

plt.show()

