import numpy as np
from matplotlib import pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from scipy import stats
from sklearn.metrics import mean_absolute_error as MAE

NUMPARTICIPANTS = 39

import seaborn as sns
sns.set(font_scale=3)
# matplotlib.rcParams.update({'font.size': 16})
fig = plt.figure(figsize=(11,10))
# fig.suptitle("Choice data fits - Dataset 1")


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return h

def computeParticipantAllLogRTResidual(participantIndex, data):
    trialIndicesOfParticipant = np.asarray(np.where(data[:, 0] == participantIndex))[0]
    participantData = data[trialIndicesOfParticipant]
    participantAllRT = participantData[:, 2]
    participantAllLogRT = np.log(participantAllRT).reshape(-1, 1)
    participantAllGainLoss = participantData[:, 3:5]

    regressor = LinearRegression()
    regressor.fit(participantAllGainLoss, participantAllLogRT)

    participantAllPredictedLogRT = regressor.predict(participantAllGainLoss)
    participantAllLogRTResidual = participantAllLogRT - participantAllPredictedLogRT

    return np.ndarray.flatten(participantAllLogRTResidual)


def extractParticipantResponses(participantIndex, data):
    trialIndicesOfParticipant = np.asarray(np.where(data[:, 0] == participantIndex))[0]
    participantData = data[trialIndicesOfParticipant]
    participantResponses = participantData[:, 1]

    return participantResponses


def computeParticipantMeanPAcceptForBinnedRT(participantIndex, numBins, data):
    participantAllLogRTResidual = computeParticipantAllLogRTResidual(participantIndex, data)
    participantResponses = extractParticipantResponses(participantIndex, data)
    _, sortedResponses = (list(t) for t in zip(*sorted(zip(participantAllLogRTResidual.tolist(), participantResponses.tolist()))))
    binnedResponses = np.array_split(sortedResponses, numBins)
    binPAccept = [np.mean(binResponses) for binResponses in binnedResponses]

    return binPAccept

def wrapper(data):
    allParticipantMeanPAcceptForBinnedRT = np.array([computeParticipantMeanPAcceptForBinnedRT(participantIndex, 5, data) for participantIndex in range(1, NUMPARTICIPANTS+1)])
    meanPAcceptForBinnedRT = np.mean(allParticipantMeanPAcceptForBinnedRT, 0)

    return meanPAcceptForBinnedRT

actualData = np.genfromtxt("dataZeroSure.csv", delimiter=',')[1:, 1:]


def plot(ax, data, linestyle, label):
    # errorBars = [mean_confidence_interval(allParticipantMeanPAcceptForBinnedRT[:, i].flatten()) for i in range(5)]
    meanPAcceptForBinnedRT = wrapper(data)

    ax.plot((1, 2, 3, 4, 5), meanPAcceptForBinnedRT, marker='o', label=label, linestyle=linestyle)
    ax.set_xticks((1, 2, 3, 4, 5))
    # ax.set_ylabel("P(accept)")
    # ax.set_xlabel("Choice-factor adjusted RT bins")
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x')
    ax.tick_params(axis='y')
    ax.set_aspect(3)
    # ax.legend()


    actualDataPoints = wrapper(actualData)
    mae = MAE(actualDataPoints, meanPAcceptForBinnedRT)
    print(mae)
    if mae > 0.01:
        ax.annotate("MAE = %.3f" % mae, (1, 0.9))



fullModelSimulatedData = np.genfromtxt("simulatedData/full/full.csv", delimiter=',')[1:, 1:]
ax1 = plt.subplot2grid(shape=(1,1), loc=(0,0), colspan=1)
# plot(ax1, fullModelSimulatedData, '--', 'Predicted')
plot(ax1, actualData, '-', 'Observed')

plt.xlabel("Choice-factor Adjusted RT Bins")
plt.ylabel("P(accept)")

plt.tight_layout()
plt.savefig("fig/2_behavMarkerObservedOnly.png", bbox_inches='tight')
plt.close()

exit()


ax1.set_title("Full Model")


noBiasSimulatedData = np.genfromtxt("simulatedData/no_predecisional_bias/no_predecisional_bias.csv", delimiter=',')[1:, 1:]
ax2 = plt.subplot2grid(shape=(2,2), loc=(1,0), colspan=1)
plot(ax2, noBiasSimulatedData, '--', 'Predicted')
plot(ax2, actualData, '-', 'Observed')
ax2.set_title("No Predecisional Bias")


noLossAversionSimulatedData = np.genfromtxt("simulatedData/no_loss_aversion/no_loss_aversion.csv", delimiter=',')[1:, 1:]
ax3 = plt.subplot2grid(shape=(2,2), loc=(0,1), colspan=1)
plot(ax3, noLossAversionSimulatedData, '--', 'Predicted')
plot(ax3, actualData, '-', 'Observed')
ax3.set_title("No Loss Aversion")


noAlphaData = np.genfromtxt("simulatedData/no_alpha/no_alpha.csv", delimiter=',')[1:, 1:]
ax4 = plt.subplot2grid(shape=(2,2), loc=(1,1), colspan=1)
plot(ax4, noAlphaData, '--', 'Predicted')
plot(ax4, actualData, '-', 'Observed')
ax4.set_title("No Fixed Utility Bias")

handles, labels = ax4.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', ncol=2)

fig.add_subplot(111, frameon=False).grid(False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel("Choice-factor Adjusted RT Bins")
plt.ylabel("P(accept)")

# plt.show()

plt.tight_layout()
plt.savefig("fig/behavMarker.png", bbox_inches='tight')
plt.close()

