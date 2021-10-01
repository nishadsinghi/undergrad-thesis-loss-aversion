import numpy as np
from matplotlib import pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

data = np.genfromtxt("../src/risk_data.csv", delimiter=',')[1:, :]

def computeParticipantAllLogRTResidual(participantIndex):
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


def extractParticipantResponses(participantIndex):
    trialIndicesOfParticipant = np.asarray(np.where(data[:, 0] == participantIndex))[0]
    participantData = data[trialIndicesOfParticipant]
    participantResponses = participantData[:, 1]

    return participantResponses


def prepareParticipantDictPAcceptLogRTResidual(participantIndex):
    participantAllLogRTResidual = computeParticipantAllLogRTResidual(participantIndex)
    participantResponses = extractParticipantResponses(participantIndex)

    dictPAcceptLogRTResidual = {residual: pAccept for pAccept, residual in
                                zip(participantResponses, participantAllLogRTResidual)}

    return dictPAcceptLogRTResidual


combinedDictPAcceptLogRTResidual = defaultdict(list)
for participantIndex in range(1, 50):
    participantDict = prepareParticipantDictPAcceptLogRTResidual(participantIndex)
    for k, v in participantDict.items():
        combinedDictPAcceptLogRTResidual[k].append(v)

sortedDict = {k: v[0] for k, v in sorted(combinedDictPAcceptLogRTResidual.items(), key=lambda item: item[0])}
allPAccept = np.array(list(sortedDict.values()))
numTrials = np.size(allPAccept)

m1 = np.mean(allPAccept[0:int(numTrials/5)])
m2 = np.mean(allPAccept[int(numTrials/5):int(2/5*numTrials)])
m3 = np.mean(allPAccept[int(2/5*numTrials):int(3/5*numTrials)])
m4 = np.mean(allPAccept[int(3/5*numTrials):int(4/5*numTrials)])
m5 = np.mean(allPAccept[int(4/5*numTrials):numTrials])


# allParticipantResiduals = [computeParticipantMeanLogRTResidual(participantIndex) for participantIndex in range(1, 50)]
# allParticipantPAccept = [computeParticipantPAccept(participantIndex) for participantIndex in range(1, 50)]
#
# dicti = {pAccept: residual for pAccept, residual in zip(allParticipantPAccept, allParticipantResiduals)}



#
# m1 = np.mean(allPAccept[0:10])
# m2 = np.mean(allPAccept[10:20])
# m3 = np.mean(allPAccept[20:30])
# m4 = np.mean(allPAccept[30:40])
# m5 = np.mean(allPAccept[40:49])
#
plt.plot([m1, m2, m3, m4, m5], marker='o')
plt.ylim(0.15, 0.85)
plt.show()

