import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "..", "src"))

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
import matplotlib


data = np.genfromtxt("data_preprocessed.csv", delimiter=',')[1:, 1:]
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


# description: combines data of all participants, then for each pair of gain and loss, computes the quantiles and plots R2 on those values

# allQuantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
# def plotR2(ax, modelData, actualData):
#     allModelQuantiles = []
#     allActualQuantiles = []
#     for quantile in allQuantiles:
#         modelQuantiles = []
#         actualQuantiles = []
#         for stakes in allStakes:
#             gain, loss = stakes
#
#             modelTrialsForThisStakes = modelData[np.logical_and(modelData[:, 2] == gain, modelData[:, 3] == loss)]
#             modelQuantile = np.quantile(modelTrialsForThisStakes[:, 1].flatten(), quantile)
#             modelQuantiles.append(modelQuantile)
#
#             actualTrialsForThisStakes = actualData[np.logical_and(actualData[:, 2] == gain, actualData[:, 3] == loss)]
#             actualQuantile = np.quantile(actualTrialsForThisStakes[:, 1].flatten(), quantile)
#             actualQuantiles.append(actualQuantile)
#
#         allModelQuantiles += modelQuantiles
#         allActualQuantiles += actualQuantiles
#
#         ax.scatter(actualQuantiles, modelQuantiles)
#         ax.plot((0, 3.5), (0, 3.5), linestyle='--', color='r')
#         ax.set_xlim(0, 3.5)
#         ax.set_ylim(0, 3.5)
#         ax.set_xlabel("Observed RT Quantiles")
#         ax.set_ylabel("Model RT Quantiles")
#
#     r2Score = r2_score(allActualQuantiles, allModelQuantiles)
#     ax.annotate("R2 Score: %.2f" % r2Score, (0, 3.2))



QUANTILES = [0.1, 0.3, 0.5, 0.7, 0.9]
COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

def plotR2(ax, modelData, actualData):
    def extractPlottingPointsFromData(data):
        allParticipantIndex = np.unique(data[:, 0], axis=0)

        allParticipantData = []
        for participantIndex in allParticipantIndex:
            participantData = data[data[:, 0] == participantIndex]
            computeRatio = lambda trial: -1 * participantData[trial, 3] / participantData[trial, 4]
            allRatios = [computeRatio(trial) for trial in range(np.shape(participantData)[0])]
            dataWithRatios = np.hstack((participantData[:, 1:3], np.reshape(allRatios, (-1, 1))))
            sortedDataWithRatios = dataWithRatios[dataWithRatios[:,-1].argsort()]
            split = np.array_split(sortedDataWithRatios, 8, axis=0)
            # print(split[0][0])

            allQuantileDataForParticipant = np.zeros((8, 5))
            for row in range(8):
                for column in range(5):
                    splitBin = split[row]
                    quantile = QUANTILES[column]
                    reactionTimes = splitBin[:, 1].flatten()
                    allQuantileDataForParticipant[row, column] = np.quantile(reactionTimes, quantile)

            allParticipantData.append(allQuantileDataForParticipant)

        # exit()
        return np.array(allParticipantData)

    modelPlottingPoints = extractPlottingPointsFromData(modelData)
    actualPlottingPoints = extractPlottingPointsFromData(actualData)


    for i in range(5):
        modelPlottingPointsForThisQuantile = modelPlottingPoints[:, :, i].flatten()
        actualPlottingPointsForThisQuantile = actualPlottingPoints[:, :, i].flatten()

        uniqueModelPoints = np.unique(modelPlottingPointsForThisQuantile)

        ax.scatter(actualPlottingPointsForThisQuantile, modelPlottingPointsForThisQuantile, color='tab:blue')

    ax.plot((0, 10), (0, 10), '--')

    r2Score = r2_score(np.array(actualPlottingPoints).flatten(), np.array(modelPlottingPoints).flatten())
    ax.annotate("R2 Score: %.2f" % r2Score, (0, 9.6))

    ax.set_xlabel("Data RT quantiles")
    ax.set_xlabel("Model RT quantiles")




# allQuantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
# quantileColours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
# def plotR2(ax, modelData, actualData):
#     allModelQuantiles = []
#     allActualQuantiles = []
#     for quantile, quantileColor in zip(allQuantiles, quantileColours):
#         modelQuantiles = []
#         actualQuantiles = []
#         for stakes in allStakes:
#             gain, loss = stakes
#
#             modelTrialsForThisStakes = modelData[np.logical_and(modelData[:, 2] == gain, modelData[:, 3] == loss)]
#             modelQuantile = np.quantile(modelTrialsForThisStakes[:, 1].flatten(), quantile)
#             modelQuantiles.append(modelQuantile)
#
#             actualTrialsForThisStakes = actualData[np.logical_and(actualData[:, 2] == gain, actualData[:, 3] == loss)]
#             actualQuantile = np.quantile(actualTrialsForThisStakes[:, 1].flatten(), quantile)
#             actualQuantiles.append(actualQuantile)
#
#         allModelQuantiles += modelQuantiles
#         allActualQuantiles += actualQuantiles
#
#         ax.scatter(actualQuantiles, modelQuantiles)
#         ax.plot((0, 3.5), (0, 3.5), linestyle='--', color='r')
#         ax.set_xlim(0, 3.5)
#         ax.set_ylim(0, 3.5)
#         ax.set_xlabel("Observed RT Quantiles")
#         ax.set_ylabel("Model RT Quantiles")
#
#     r2Score = r2_score(allActualQuantiles, allModelQuantiles)
#     ax.annotate("R2 Score: %.2f" % r2Score, (0, 3.2))




ax = plt.subplot2grid(shape=(2,2), loc=(0,0), colspan=1)
# plotData(ax, data, '-', 'Experimental data')
fullModelData = np.genfromtxt("simulatedData/fullModel.csv", delimiter=',')
# plotData(ax, fullModelData, '--', 'No Loss Aversion')
# plotR2(ax, fullModelData, data)
ax.set_title('Full Model')

plotR2(ax, fullModelData, data)

ax = plt.subplot2grid(shape=(2,2), loc=(0,1), colspan=1)
# plotData(ax, data, '-', 'Experimental data')
noLossAversionData = np.genfromtxt("simulatedData/noLossAversion.csv", delimiter=',')
# plotData(ax, noLossAversionData, '--', 'No Loss Aversion')
plotR2(ax, noLossAversionData, data)
ax.set_title('No Loss Aversion')

ax = plt.subplot2grid(shape=(2,2), loc=(1,0), colspan=1)
# plotData(ax, data, '-', 'Experimental data')
noPredecisionalBiasData = np.genfromtxt("simulatedData/noPredecisionalBias.csv", delimiter=',')
# plotData(ax, noPredecisionalBiasData, '--', 'No Predicisional bias')
plotR2(ax, noPredecisionalBiasData, data)
ax.set_title('No Predecisional Bias')

ax = plt.subplot2grid(shape=(2,2), loc=(1,1), colspan=1)
# plotData(ax, data, '-', 'Experimental data')
noAlphaData = np.genfromtxt("simulatedData/noAlpha.csv", delimiter=',')
# plotData(ax, noAlphaData, '--', 'No Fixed Utility Bias')
plotR2(ax, noAlphaData, data)
ax.set_title('No Fixed Utility Bias')


# ax2 = plt.subplot2grid(shape=(1,2), loc=(0,1), colspan=1)
# plotR2(ax2, modelRecord, data)
#
# plt.savefig('choiceFitsDataset1.png', bbox_inches='tight')
# plt.close()

plt.show()
