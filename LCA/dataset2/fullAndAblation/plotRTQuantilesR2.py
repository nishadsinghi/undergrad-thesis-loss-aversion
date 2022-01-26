import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "..", "..", "src"))

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
import matplotlib

import seaborn as sns
sns.set(font_scale=2)

data = np.genfromtxt("dataZeroSure_cleaned_250_10500_zScore3.csv", delimiter=',')[:, 1:]
allStakes = np.unique(data[:, -2:], axis=0)

fig = plt.figure(figsize=(10,12))
# fig.suptitle("RT quantile fits - Dataset 1")



def filterFunction(tup):
    if tup[-1] != -1:
        return True
    else:
        return False

QUANTILES = [0.1, 0.3, 0.5, 0.7, 0.9]
def plotR2(ax, modelRecord, data):
    def selectConditionTrials(gain, loss):
        selectedData = data[data[:, 2] == gain][:]
        selectedData = selectedData[selectedData[:, 3] == loss][:]
        return selectedData

    allModelQuantiles = []
    allActualQuantiles = []
    for stakes in allStakes:
        gain, loss = stakes
        modelTrialsForThisStakes = modelRecord[np.logical_and(modelRecord[:, 2] == gain, modelRecord[:, 3] == loss)]
        modelRTs = modelTrialsForThisStakes[:, 1].flatten()
        modelQuantiles = [np.quantile(modelRTs, quantile) for quantile in QUANTILES]
        allModelQuantiles.append(modelQuantiles)

        actualTrialsForThisStakes = data[np.logical_and(data[:, 2] == gain, data[:, 3] == loss)]
        actualRTs = actualTrialsForThisStakes[:, 1].flatten()
        actualQuantiles = [np.quantile(actualRTs, quantile) for quantile in QUANTILES]
        allActualQuantiles.append(actualQuantiles)

    for i in range(len(QUANTILES)):
        sns.scatterplot(np.array(allActualQuantiles)[:, i].flatten(), np.array(allModelQuantiles)[:, i].flatten(), alpha=0.6)

    r2Score = r2_score(np.array(allActualQuantiles).flatten(), np.array(allModelQuantiles).flatten())

    # ax.scatter(np.array(allActualQuantiles).flatten(), np.array(allModelQuantiles).flatten())
    ax.plot((0, 5), (0, 5), linestyle='--', color='r')
    ax.annotate(r"$R^2 =$ %.2f" % r2Score, (0, 4.5))
    # ax.set_xlabel("Empirical RT Quantiles")
    # ax.set_ylabel("Model RT Quantiles")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_aspect(1)

fullModelSimulatedData = np.genfromtxt("simulatedData/full.csv", delimiter=',')
ax1 = plt.subplot2grid(shape=(2,2), loc=(0,0), colspan=1)
plotR2(ax1, fullModelSimulatedData, data)
ax1.set_title("Full Model")

NoLossAversionSimulatedData = np.genfromtxt("simulatedData/noLossAversion.csv", delimiter=',')
ax2 = plt.subplot2grid(shape=(2,2), loc=(0,1), colspan=1)
plotR2(ax2, fullModelSimulatedData, data)
ax2.set_title("No Loss Aversion")

NoPredecisionalBiasSimulatedData = np.genfromtxt("simulatedData/noPredecisionalBias.csv", delimiter=',')
ax3 = plt.subplot2grid(shape=(2,2), loc=(1,0), colspan=1)
plotR2(ax3, NoPredecisionalBiasSimulatedData, data)
ax3.set_title("No Predecisional Bias")

NoAlphaSimulatedData = np.genfromtxt("simulatedData/noFixedUtilityBias.csv", delimiter=',')
ax4 = plt.subplot2grid(shape=(2,2), loc=(1,1), colspan=1)
plotR2(ax4, NoAlphaSimulatedData, data)
ax4.set_title("No Fixed Utility Bias")

fig.add_subplot(111, frameon=False).grid(False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel("Empirical RT Quantiles (s)")
plt.ylabel("Model RT Quantiles (s)")


plt.tight_layout()
plt.savefig("fig/2_LCA_RTQuantileFitsR2.png", bbox_inches='tight')
plt.close()
