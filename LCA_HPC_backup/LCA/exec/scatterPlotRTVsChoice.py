import numpy as np
from matplotlib import pyplot as plt

data = np.genfromtxt("../src/risk_data.csv", delimiter=',')
data = data[1:, 1:]

def selectConditionTrials(gain, loss):
    selectedData = data[data[:, 2] == gain][:]
    selectedData = selectedData[selectedData[:, 3] == loss][:]
    return selectedData


def computeRTData(gain, loss):
    conditionTrialData = selectConditionTrials(gain, loss)
    RTs = conditionTrialData[:, 1]
    return np.mean(RTs)


def computePData(gain, loss):
    conditionTrialData = selectConditionTrials(gain, loss)
    responses = conditionTrialData[:, 0]
    return np.mean(responses)


allGainValues = list(range(10, 110, 10))
allLossValues = list(range(-100, 0, 10))

allMeanRTs = []
allMeanP = []
allGainLossRatio = []

for gain in allGainValues:
    for loss in allLossValues:
        allGainLossRatio.append(-1*gain/loss)
        allMeanRTs.append(computeRTData(gain, loss))
        allMeanP.append(computePData(gain, loss))

sortedRatios, sortedP, sortedRTs = (list(t) for t in zip(*sorted(zip(allGainLossRatio, allMeanP, allMeanRTs))))

binnedP = np.array_split(sortedP, 10)
binP = [np.mean(binResponses) for binResponses in binnedP]

binnedRTs = np.array_split(sortedRTs, 5)
binRTs = [np.mean(binResponses) for binResponses in binnedRTs]

# plt.plot(binP, binRTs, marker='o')
plt.plot(binRTs, marker='o')
plt.show()

exit()

# plt.scatter(allMeanP, allMeanRTs)
plt.scatter(allMeanRTs, allMeanP)
plt.xscale('log')
plt.show()