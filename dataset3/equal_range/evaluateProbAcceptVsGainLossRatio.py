import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "..", "src"))

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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
    df = pd.DataFrame(ActualRatioAndProbRecord)
    ratioCombinedDf = df.groupby(0).mean()
    ratioCombinedDf = ratioCombinedDf.rename(columns={1: label})
    ratioCombinedDf.plot(ax=ax, logx=True, linestyle=linestyle)
    # ax.set_ylim(0, 1)
    ax.set_xlabel("Gain/Loss")
    ax.set_ylabel("P(Accept Gamble)")




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


ax = plt.subplot2grid(shape=(1,1), loc=(0,0), colspan=1)
plotData(ax, data, '-', 'experimental data')

noLossAversionData = np.genfromtxt("savedModels/no_loss_aversion/no_loss_aversion.csv", delimiter=',')
print(noLossAversionData[:10, :])
exit()


# ax2 = plt.subplot2grid(shape=(1,2), loc=(0,1), colspan=1)
# plotR2(ax2, modelRecord, data)
#
# plt.savefig('choiceFitsDataset1.png', bbox_inches='tight')
# plt.close()

plt.show()
