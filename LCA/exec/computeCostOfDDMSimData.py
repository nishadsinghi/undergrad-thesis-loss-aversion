import numpy as np
from scipy.stats import skew
from scipy.stats import chisquare

def selectConditionTrials(gain, loss, data):
    selectedData = data[data[:, 2] == gain][:]
    selectedData = selectedData[selectedData[:, 3] == loss][:]
    return selectedData

def computePData(gain, loss, data):
    conditionTrialData = selectConditionTrials(gain, loss, data)
    responses = conditionTrialData[:, 0]
    return np.mean(responses)

def computeRTStatsData(gain, loss, choice, data):
    conditionTrialData = selectConditionTrials(gain, loss, data)
    dataForThisChoice = conditionTrialData[conditionTrialData[:, 0] == choice]
    reactionTimes = dataForThisChoice[:, 1]
    return (np.mean(reactionTimes), np.std(reactionTimes), skew(reactionTimes))

# actual data
actual = np.genfromtxt("../src/risk_data.csv", delimiter=',')[1:, 1:]

# sim data
sim = np.genfromtxt("/Users/nishadsinghi/undergrad-project-loss-aversion/DDM/DDMSimDataAllClubbedOnlyStartingBias.csv", delimiter=',')[1:, 1:]

allGainValues = list(range(10, 110, 10))
allLossValues = list(range(-100, 0, 10))

totalCost = 0
quantiles = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
observedProportionsChoiceWise = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
quantileWeights = [2, 2, 1, 1, 0.5]
counter = 0
for gain in allGainValues:
    for loss in allLossValues:
        observedTrials = selectConditionTrials(gain, loss, actual)
        numObservedTrials = np.shape(observedTrials)[0]
        modelTrials = selectConditionTrials(gain, loss, sim)
        numModelTrials = np.shape(modelTrials)[0]
        for choice in range(2):

            observedTrialsForChoice = observedTrials[observedTrials[:, 0] == choice]
            observedRTsForChoice = observedTrialsForChoice[:, 1]
            numObservedRTsForChoice = np.size(observedRTsForChoice)
            observedPOfThisChoice = numObservedRTsForChoice/numObservedTrials

            if numObservedRTsForChoice < 5:
                continue

            quantilesBoundaries = np.quantile(observedRTsForChoice, quantiles)
            expectedFrequencies = np.histogram(observedRTsForChoice, bins=np.concatenate(([0], quantilesBoundaries, [100])))[0] / numObservedTrials

            if numObservedRTsForChoice == 5:
                expectedFrequencies = observedProportionsChoiceWise * observedPOfThisChoice

            modelTrialsForChoice = modelTrials[modelTrials[:, 0] == choice]
            modelRTsForChoice = modelTrialsForChoice[:, 1]
            modelFrequencies = np.histogram(modelRTsForChoice, bins=np.concatenate(([0], quantilesBoundaries, [100])))[0] /numModelTrials
            print(modelFrequencies)
            print(expectedFrequencies)
            print(np.sum(modelFrequencies), np.sum(expectedFrequencies))

            totalCost += chisquare(modelFrequencies, expectedFrequencies)[0]

            print("-------------")


print("total cost of DDM: ", totalCost)
exit()































t1 = 0
t2 = 0
t3 = 0
t4 = 0

for gain in allGainValues:
    for loss in allLossValues:
        for choice in range(2):
            t1 += ((computePData(gain, loss, actual) - computePData(gain, loss, sim)) ** 2)
            dataSD = computeRTStatsData(gain, loss, choice, actual)[1]
            if dataSD <= 0.1:
                t2 += ((computeRTStatsData(gain, loss, choice, actual)[0] - computeRTStatsData(gain, loss, choice, sim)[0]) ** 2)
            else:
                t2 += ((computeRTStatsData(gain, loss, choice, actual)[0] - computeRTStatsData(gain, loss, choice, sim)[0]) ** 2)# / (dataSD ** 2)
            t3 += ((computeRTStatsData(gain, loss, choice, actual)[1] - computeRTStatsData(gain, loss, choice, sim)[1]) ** 2)
            t4 += (computeRTStatsData(gain, loss, choice, actual)[2] - computeRTStatsData(gain, loss, choice, sim)[2])**2



print(t1, t2, t3, t4)
# print(250*t1 + t2 + 0.7*t3)


