import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "..", "src"))

from LCA import PrepareRecurrentWeights, RunLCASimulation, GetValueInputZeroReference, getValueInput2Attributes2Choices

#
import numpy as np
# from matplotlib import pyplot as plt
# import pickle
#
# data = np.genfromtxt("../src/risk_data.csv", delimiter=',')[1:, :]
#
#
# def extractParticipantData(participantIndex):
#     trialIndicesOfParticipant = np.asarray(np.where(data[:, 0] == participantIndex))[0]
#     participantData = data[trialIndicesOfParticipant]
#     participantRT = participantData[:, 2]
#     participantGainLoss = participantData[:, -2:]
#     participantRatios = participantGainLoss[:, 0] / participantGainLoss[:, 1] * -1
#
#     participantDict = {ratio: RT for ratio, RT in zip(participantRatios, participantRT)}
#
#     return participantDict
#
#
# sortDictUsingKeys = lambda dictionary: {key: value for key, value in sorted(dictionary.items(), key=lambda item: item[0])}
# prepareParticipantSortedDict = lambda participantIndex: sortDictUsingKeys(extractParticipantData(participantIndex))
#
#
# def computeParticipantMeanRTForBinnedRatio(participantIndex, numBins):
#     participantSortedDict = prepareParticipantSortedDict(participantIndex)
#     allRT = np.array(list(participantSortedDict.values()))
#     binnedRT = np.array_split(allRT, numBins)
#     meanBinnedRT = [np.mean(RTBin) for RTBin in binnedRT]
#
#     return meanBinnedRT
#
#
# allParticipantMeanRTForBinnedRatio = np.array([computeParticipantMeanRTForBinnedRatio(participantIndex, 10) for
#                                                participantIndex in range(1, 50)])
# meanRTForBinnedRatio = np.mean(allParticipantMeanRTForBinnedRatio, 0)
# plt.plot(meanRTForBinnedRatio, marker='o')
# plt.xlabel("Mean RT (raw)")
# plt.ylabel("Gain/Loss ratio (binned)")
# plt.show()
