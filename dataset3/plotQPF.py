import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "..", "src"))

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


QUANTILES = [0.1, 0.3, 0.5, 0.7, 0.9]
MARKERS = ['o', 's', 'D', 'v', '^']
COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']



def plotQPF(ax, data, plotType, linestyle, label):
    computeRatio = lambda trial: -1 * data[trial, 3] / data[trial, 4]
    allRatios = [computeRatio(trial) for trial in range(np.shape(data)[0])]
    dataWithRatios = np.hstack((data[:, 0:3], np.reshape(allRatios, (-1, 1))))

    allParticipantIndex = np.unique(dataWithRatios[:, 0])

    for quantile, marker, color in zip(QUANTILES, MARKERS, COLORS):
        allParticipantMeanAcceptanceRates = []
        allParticipantMeanRTQuantiles = []
        for participant in allParticipantIndex:
            participantData = dataWithRatios[dataWithRatios[:, 0] == participant]
            sortedDataWithRatios = participantData[participantData[:,-1].argsort()]

            split = np.array_split(sortedDataWithRatios, 8, axis=0)

            def computeQuantiles(data):
                reactionTimes = data[:, 2].flatten()
                reactionTimeForQuantile = np.quantile(reactionTimes, quantile)
                return reactionTimeForQuantile

            def computeP(data):
                choices = data[:, 1].flatten()
                P_mean = np.mean(choices)
                return P_mean

            allParticipantMeanAcceptanceRates.append([computeP(_) for _ in split])
            allParticipantMeanRTQuantiles.append([computeQuantiles(_) for _ in split])

        toPlotX = np.mean(allParticipantMeanAcceptanceRates, axis=0)

        toPlotY = np.mean(allParticipantMeanRTQuantiles, axis=0)

        if plotType == 'plot':
            ax.plot(toPlotX, toPlotY, color=color, linestyle=linestyle)
        elif plotType == 'scatter':
            ax.scatter(toPlotX, toPlotY, marker=marker, color=color)

    print(toPlotX)
    ax.set_ylabel("RT quantiles", fontsize=16)
    ax.set_xlabel("P(accept)", fontsize=16)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.75, 3)
    ax.set_title(label)











# data = np.genfromtxt("../src/risk_data_cleaned.csv", delimiter=',')[1:, 1:]
data = np.genfromtxt("data_preprocessed.csv", delimiter=',')[1:, 1:]


fullModelData = np.genfromtxt("simulatedData/fullModel.csv", delimiter=',')
ax = plt.subplot2grid(shape=(2,2), loc=(0,0), colspan=1)
plotQPF(ax, data, 'scatter', '--', 'Empirical data')
plotQPF(ax, fullModelData, 'plot', '--', 'Full Model Data')


noLossAversionData = np.genfromtxt("simulatedData/noLossAversion.csv", delimiter=',')
ax = plt.subplot2grid(shape=(2,2), loc=(0,1), colspan=1)
plotQPF(ax, data, 'scatter', '--', 'Empirical data')
plotQPF(ax, noLossAversionData, 'plot', '--', 'No Loss Aversion')


noPredecisionalBiasData = np.genfromtxt("simulatedData/noPredecisionalBias.csv", delimiter=',')
ax = plt.subplot2grid(shape=(2,2), loc=(1,0), colspan=1)
plotQPF(ax, data, 'scatter', '--', 'Empirical data')
plotQPF(ax, noPredecisionalBiasData, 'plot', '--', 'No Predecisional Bias')


noAlphaData = np.genfromtxt("simulatedData/noAlpha.csv", delimiter=',')
ax = plt.subplot2grid(shape=(2,2), loc=(1,1), colspan=1)
plotQPF(ax, data, 'scatter', '--', 'Empirical data')
plotQPF(ax, noAlphaData, 'plot', '--', 'No Fixed Utility Bias')




plt.show()

