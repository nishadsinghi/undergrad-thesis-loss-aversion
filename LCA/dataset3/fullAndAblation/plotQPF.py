import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "..", "..", "src"))

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib

QUANTILES = [0.1, 0.3, 0.5, 0.7, 0.9]
MARKERS = ['o', 's', 'D', 'v', '^']
COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

sns.set(font_scale=1.75)
fig = plt.figure(figsize=(12, 12))
# fig.suptitle("Quantile Probability Functions - Dataset 1")
# matplotlib.rcParams.update({'font.size': 14})

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

        # ax.set_ylabel("RT Quantiles (s)")#, fontsize=16)
        # ax.set_xlabel("P(accept)")#, fontsize=16)
        ax.set_ylim(0, 3.1)
        ax.set_xlim(0, 1)


data = np.genfromtxt("data_cleaned_250_10500_zScore3.csv", delimiter=',')[:, 1:]
allStakes = np.unique(data[:, -2:], axis=0)

def augmentDataWithRatios(data):
    computeRatio = lambda trial: -1*data[trial, 2]/data[trial, 3]
    allRatios = [computeRatio(trial) for trial in range(np.shape(data)[0])]
    dataWithRatios = np.hstack((data[:, 0:2], np.reshape(allRatios, (-1, 1))))

    return dataWithRatios

experimentalDataWithRatios = augmentDataWithRatios(data)

fullModelSimulatedData = np.genfromtxt("simulatedData/full.csv", delimiter=',')
fullModelDataWithRatios = augmentDataWithRatios(fullModelSimulatedData)
ax1 = plt.subplot2grid(shape=(2,2), loc=(0,0), colspan=1)
plotQPF(ax1, experimentalDataWithRatios, 'scatter', '-', 'Empirical data')
plotQPF(ax1, fullModelDataWithRatios, 'plot', '--', 'LCA')
ax1.set_title("Full Model")

noLossAversionSimulatedData = np.genfromtxt("simulatedData/noLossAversion.csv", delimiter=',')
noLossAversionDataWithRatios = augmentDataWithRatios(noLossAversionSimulatedData)
ax2 = plt.subplot2grid(shape=(2,2), loc=(0,1), colspan=1)
plotQPF(ax2, experimentalDataWithRatios, 'scatter', '-', 'Empirical data')
plotQPF(ax2, noLossAversionDataWithRatios, 'plot', '--', 'LCA')
ax2.set_title("No Loss Aversion")

noPredecisionalBiasSimulatedData = np.genfromtxt("simulatedData/noPredecisionalBias.csv", delimiter=',')
noPredecisionalBiasDataWithRatios = augmentDataWithRatios(noPredecisionalBiasSimulatedData)
ax3 = plt.subplot2grid(shape=(2,2), loc=(1,0), colspan=1)
plotQPF(ax3, experimentalDataWithRatios, 'scatter', '-', 'Empirical data')
plotQPF(ax3, noPredecisionalBiasDataWithRatios, 'plot', '--', 'LCA')
ax3.set_title("No Predecisional Bias")

noAlphaSimulatedData = np.genfromtxt("simulatedData/noFixedUtilityBias.csv", delimiter=',')
noAlphaDataWithRatios = augmentDataWithRatios(noAlphaSimulatedData)
ax4 = plt.subplot2grid(shape=(2,2), loc=(1,1), colspan=1)
plotQPF(ax4, experimentalDataWithRatios, 'scatter', '-', 'Empirical data')
plotQPF(ax4, noAlphaDataWithRatios, 'plot', '--', 'LCA')
ax4.set_title("No Fixed Utility Bias")

fig.add_subplot(111, frameon=False).grid(False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel("P(Accept)")
plt.ylabel("RT Quantiles (s)")

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("fig/5_LCA_QPF.png", bbox_inches='tight')
plt.close()
