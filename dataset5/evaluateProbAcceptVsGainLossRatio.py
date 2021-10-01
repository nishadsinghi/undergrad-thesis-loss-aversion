import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "..", "src"))


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle
from sklearn.metrics import r2_score
import matplotlib


data = np.genfromtxt("data_preprocessed.csv", delimiter=',')[1:, 2:]
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
    print(np.shape(ActualRatioAndProbRecord))
    sortedData = ActualRatioAndProbRecord[ActualRatioAndProbRecord[:, 0].argsort()]
    choices = sortedData[:, 1].flatten()
    choicesBinned = np.array_split(choices, 8, axis=0)
    meanChoices = [np.mean(_) for _ in choicesBinned]
    print(meanChoices)
    plt.plot(meanChoices, linestyle)
    # df = pd.DataFrame(ActualRatioAndProbRecord)
    # ratioCombinedDf = df.groupby(0).mean()
    # ratioCombinedDf = ratioCombinedDf.rename(columns={1: label})
    # ratioCombinedDf.plot(ax=ax, logx=True, linestyle=linestyle)
    # ax.set_ylim(0, 1)
    # ax.set_xlabel("Gain/Loss")
    # ax.set_ylabel("P(Accept Gamble)")


fullModelData = np.genfromtxt("simulatedData/fullModel.csv", delimiter=',')[:, 1:]
ax = plt.subplot2grid((1, 1), (0, 0))
plotData(ax, data, '-', 'Empirical Data')
plotData(ax, fullModelData, '--', 'Full Model Data')

# plt.legend()
# plt.show()


