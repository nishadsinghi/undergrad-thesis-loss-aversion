import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "..", "src"))

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
import matplotlib


import seaborn as sns
sns.set(font_scale=1.5)
# matplotlib.rcParams.update({'font.size': 16})
fig = plt.figure(figsize=(10,10))
# fig.suptitle("Choice data fits - Dataset 1")



data = np.genfromtxt("data_preprocessed.csv", delimiter=',')[1:, 2:]
print(data[:5, :])
allStakes = np.unique(data[:, -2:], axis=0)



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
        P_model = np.mean(modelTrialsForThisStakes[:, 0])
        allPModel.append(P_model)

        P_actual = computePData(gain, loss)
        allPActual.append(P_actual)

    print('number of stakes: ', len(allPActual), len(allPModel))
    r2Score = r2_score(allPModel, allPActual)

    ax.scatter(allPActual, allPModel)
    ax.set_aspect(1)
    ax.plot((0, 1), (0, 1), linestyle='--', color='r')
    ax.annotate(r"$R^2 = %.2f$" % r2Score, (0.05, 0.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # ax.set_xlabel("Observed P(Accept)")
    # ax.set_ylabel("model P(Accept)")


ax = plt.subplot2grid(shape=(2,2), loc=(0,0), colspan=1)
# plotData(ax, data, '-', 'Experimental data')
fullModelData = np.genfromtxt("simulatedData/fullModel.csv", delimiter=',')[:, 1:]
print(fullModelData[:5, :])
# plotData(ax, fullModelData, '--', 'No Loss Aversion')
plotR2(ax, fullModelData, data)
ax.set_title('Full Model')

ax = plt.subplot2grid(shape=(2,2), loc=(0,1), colspan=1)
# plotData(ax, data, '-', 'Experimental data')
noLossAversionData = np.genfromtxt("simulatedData/noLossAversion.csv", delimiter=',')[:, 1:]
# plotData(ax, noLossAversionData, '--', 'No Loss Aversion')
plotR2(ax, noLossAversionData, data)
ax.set_title('No Loss Aversion')

ax = plt.subplot2grid(shape=(2,2), loc=(1,0), colspan=1)
# plotData(ax, data, '-', 'Experimental data')
noPredecisionalBiasData = np.genfromtxt("simulatedData/noPredecisionalBias.csv", delimiter=',')[:, 1:]
# plotData(ax, noPredecisionalBiasData, '--', 'No Predicisional bias')
plotR2(ax, noPredecisionalBiasData, data)
ax.set_title('No Predecisional Bias')

ax = plt.subplot2grid(shape=(2,2), loc=(1,1), colspan=1)
# plotData(ax, data, '-', 'Experimental data')
noAlphaData = np.genfromtxt("simulatedData/noAlpha.csv", delimiter=',')[:, 1:]
# plotData(ax, noAlphaData, '--', 'No Fixed Utility Bias')
plotR2(ax, noAlphaData, data)
ax.set_title('No Fixed Utility Bias')


fig.add_subplot(111, frameon=False).grid(False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel("Observed P(accept)")
plt.ylabel("Model P(accept)")


# ax2 = plt.subplot2grid(shape=(1,2), loc=(0,1), colspan=1)
# plotR2(ax2, modelRecord, data)
#
# plt.savefig('choiceFitsDataset1.png', bbox_inches='tight')
# plt.close()

plt.tight_layout()
plt.savefig("fig/choiceFitsR2.png", bbox_inches='tight')
plt.close()

