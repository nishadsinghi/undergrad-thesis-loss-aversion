import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, "..", "..", "src"))

import numpy as np
from matplotlib import pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import matplotlib
from scipy import stats
import seaborn as sns
from sklearn.metrics import mean_absolute_error as MAE

from LCA import PrepareRecurrentWeights, RunLCASimulation, GetValueInputZeroReference, getValueInput2Attributes2Choices
from LUT import prepareStdNormalLUT, SampleFromLUT

experimentalData = np.genfromtxt("data_cleaned_250_10500_zScore3.csv", delimiter=',')[:, 1:]
allStakes = np.unique(experimentalData[:, -2:], axis=0)

fig = plt.figure(figsize=(7, 7))
fig.suptitle("Choice vs. Choice-factor Adjusted RT - Dataset 3")

sns.set()


def computeBinPAccept(data):
    allLogRT = np.log(data[:, 1]).reshape(-1, 1)
    allGainLoss = data[:, 2:]
    regressor = LinearRegression()
    regressor.fit(allGainLoss, allLogRT)
    allPredictedLogRT = regressor.predict(allGainLoss)
    allLogRTResidual = allLogRT - allPredictedLogRT

    responses = data[:, 0]
    _, sortedResponses = (list(t) for t in zip(*sorted(zip(allLogRTResidual.tolist(), responses.tolist()))))

    binnedResponses = np.array_split(sortedResponses, 5)
    binPAccept = [np.mean(binResponses) for binResponses in binnedResponses]
    return binPAccept


def plotData(ax, data, label, linestyle):
    binPAccept = computeBinPAccept(data)
    plt.plot((1, 2, 3, 4, 5), binPAccept, marker='o', label=label, linestyle=linestyle)
    ax.set_xticks((1, 2, 3, 4, 5))
    ax.set_ylim(0, 1)

    actualBinPAccept = computeBinPAccept(experimentalData)
    mae = MAE(actualBinPAccept, binPAccept)
    print(mae)
    if mae > 0.01:
        ax.annotate("MAE = %.3f" % mae, (1, 0.9))


                                                    # SIMULATION
numSimulationsPerCondition = 2#150

def filterFunction(tup):
    if tup[-1] != -1:
        return True
    else:
        return False


def generateModelData(lcaWrapper, params):
    allModelData = np.zeros(4)
    for stakes in allStakes:
        gainValue, lossValue = stakes
        allSimulations = [lcaWrapper(gainValue, lossValue, *params) for _ in
                          range(numSimulationsPerCondition)]
        allValidResponseSimulations = list(filter(filterFunction, allSimulations))
        numValidResponses = len(allValidResponseSimulations)
        _, allModelRTs, allModelResponses = zip(*allValidResponseSimulations)

        allModelResponses = np.array(allModelResponses)
        allModelRTs = np.array(allModelRTs)
        modelStakes = np.hstack((np.full((numValidResponses, 1), gainValue), np.full((numValidResponses, 1), lossValue)))
        modelDataForStakes = np.hstack((np.array(allModelResponses).reshape(-1, 1), np.array(allModelRTs).reshape(-1, 1), modelStakes))
        allModelData = np.vstack((allModelData, modelDataForStakes))

    allModelData = allModelData[1:, :]
    return allModelData


LUTInterval = 0.0001
numAccumulators = 2
stdNormalLUT = prepareStdNormalLUT(LUTInterval)
sampleFromLUT = SampleFromLUT(stdNormalLUT)
sampleFromZeroMeanLUT = lambda stdDev: sampleFromLUT(0, stdDev, numAccumulators)

# set-up the LCA
identityUtilityFunction = lambda x: x
getValueInput = GetValueInputZeroReference(identityUtilityFunction)

maxTimeSteps = 1000
deltaT = 0.02
prepareRecurrentWeights = PrepareRecurrentWeights(numAccumulators)
lca = RunLCASimulation(getValueInput, sampleFromZeroMeanLUT, prepareRecurrentWeights, maxTimeSteps, deltaT)

# FULL LCA MODEL
def getStartingActivation(startingBias, threshold):
    if startingBias < 0:
        return [-1 * startingBias * threshold, 0]
    else:
        return [0, startingBias * threshold]


getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
getAllThresholds = lambda threshold: [threshold] * numAccumulators
attributeProbabilities = [0.5, 0.5]

lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, constantInput1, lossWeight, startingBias, constantInput2: \
        lca(attributeProbabilities, getChoiceAttributes(gain, lossWeight*loss), getStartingActivation(startingBias, threshold), decay, competition, (constantInput1, constantInput2),
            noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

params = [ 0.19083179,  1.97836936,  9.20583347,  0.22501344,  6.16374677,  5.44894383, 2.31502667, -0.40814467, 10.34846199]
modelSimulationData = generateModelData(lcaWrapper, params)
ax1 = plt.subplot2grid(shape=(2,2), loc=(0,0), colspan=1)
plotData(ax1, experimentalData, 'Actual data', '-')
plotData(ax1, modelSimulationData, 'LCA', '--')
ax1.set_title("Full LCA Model")



# NO LOSS AVERSION
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, constantInput1, startingBias, constantInput2: \
        lca(attributeProbabilities, getChoiceAttributes(gain, loss), getStartingActivation(startingBias, threshold), decay, competition, (constantInput1, constantInput2),
            noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

params = [ 0.36844384,  2.02498637,  6.38335664,  0.24684811,  3.99386005,  4.65329466, -0.3179889,  5.09714807]

modelSimulationData = generateModelData(lcaWrapper, params)
ax2 = plt.subplot2grid(shape=(2,2), loc=(0,1), colspan=1)
plotData(ax2, experimentalData, 'Actual data', '-')
plotData(ax2, modelSimulationData, 'LCA', '--')
ax2.set_title("No Loss Aversion")



# NO PREDECISIONAL BIAS
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, constantInput1, lossWeight, constantInput2: \
        lca(attributeProbabilities, getChoiceAttributes(gain, lossWeight*loss), getStartingActivation(0, threshold), decay, competition, (constantInput1, constantInput2),
            noiseStdDev, nonDecisionTime, getAllThresholds(threshold))
params = [ 0.76373263,  2.21259982, 11.46590887,  0.28965878,  5.450447,   10.01588434, 1.78277041,  9.61182545]

modelSimulationData = generateModelData(lcaWrapper, params)

ax3 = plt.subplot2grid(shape=(2,2), loc=(1,0), colspan=1)
plotData(ax3, experimentalData, 'Actual data', '-')
plotData(ax3, modelSimulationData, 'LCA', '--')
ax3.set_title("No Predecisional bias")




# NO FIXED UTILIY BIAS
lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, constantInput, lossWeight, startingBias: \
    lca(attributeProbabilities, getChoiceAttributes(gain, lossWeight * loss),
        getStartingActivation(startingBias, threshold), decay, competition, constantInput,
        noiseStdDev, nonDecisionTime, getAllThresholds(threshold))

params = [ 2.31385834,  3.95508608,  7.16409715,  0.37944811,  2.98087371,  8.62970996, 1.36309624, -0.21636609]

modelSimulationData = generateModelData(lcaWrapper, params)
ax4 = plt.subplot2grid(shape=(2,2), loc=(1,1), colspan=1)
plotData(ax4, experimentalData, 'Actual data', '-')
plotData(ax4, modelSimulationData, 'LCA', '--')
ax4.set_title("No Fixed Utility bias")


plt.xticks((1, 2, 3, 4, 5))
plt.ylim(0, 1)

plt.legend()

sns.set_style("whitegrid", {'axes.grid' : False})
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Choice-factor adjusted RT")
plt.ylabel("P(accept)")


fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("fig/LCABehaviouralMarkers.png", bbox_inches='tight')
plt.close()





