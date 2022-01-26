import numpy as np
import pandas as pd
from scipy.stats import skew, chisquare


data = np.genfromtxt("dataZeroSure_cleaned_250_10500_zScore3.csv", delimiter=',')[:, 1:]
allStakes = np.unique(data[:, -2:], axis=0)

def selectConditionTrials(data, gain, loss):
    selectedData = data[data[:, 2] == gain][:]
    selectedData = selectedData[selectedData[:, 3] == loss][:]
    return selectedData

def computeChiSquared(allModelData):
    totalCost = 0
    quantiles = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    observedProportionsChoiceWise = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
    for stakes in allStakes:
        gain, loss = stakes
        observedTrials = selectConditionTrials(data, gain, loss)
        numObservedTrials = np.shape(observedTrials)[0]
        modelTrials = selectConditionTrials(allModelData, gain, loss)
        numModelTrials = np.shape(modelTrials)[0]
        for choice in range(2):
            observedTrialsForChoice = observedTrials[observedTrials[:, 0] == choice]
            observedRTsForChoice = observedTrialsForChoice[:, 1]
            numObservedRTsForChoice = np.size(observedRTsForChoice)
            observedPOfThisChoice = numObservedRTsForChoice / numObservedTrials

            if numObservedRTsForChoice < 5:
                continue

            quantilesBoundaries = np.quantile(observedRTsForChoice, quantiles)

            observedProportions = \
                np.histogram(observedRTsForChoice, bins=np.concatenate(([0], quantilesBoundaries, [100])))[
                    0] / numObservedTrials

            if numObservedRTsForChoice == 5:
                observedProportions = observedProportionsChoiceWise * observedPOfThisChoice

            observedFrequencies = numObservedTrials * observedProportions

            modelTrialsForChoice = modelTrials[modelTrials[:, 0] == choice]
            modelRTsForChoice = modelTrialsForChoice[:, 1]
            numModelRTsForChoice = np.size(modelRTsForChoice)
            modelProportions = \
            np.histogram(modelRTsForChoice, bins=np.concatenate(([0], quantilesBoundaries, [100])))[
                0] / numModelTrials

            modelFrequencies = numObservedTrials * modelProportions

            totalCost += chisquare(modelFrequencies, observedFrequencies)[0]
            if chisquare(modelFrequencies, observedFrequencies)[0] > 100000:
                print(modelFrequencies, observedFrequencies)
                exit()

    return totalCost


def computeAIC(allModelData, numParams):
    totalCost = 0
    quantiles = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    observedProportionsChoiceWise = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
    for stakes in allStakes:
        gain, loss = stakes
        observedTrials = selectConditionTrials(data, gain, loss)
        numObservedTrials = np.shape(observedTrials)[0]
        modelTrials = selectConditionTrials(allModelData, gain, loss)
        numModelTrials = np.shape(modelTrials)[0]
        for choice in range(2):
            observedTrialsForChoice = observedTrials[observedTrials[:, 0] == choice]
            observedRTsForChoice = observedTrialsForChoice[:, 1]
            numObservedRTsForChoice = np.size(observedRTsForChoice)
            observedPOfThisChoice = numObservedRTsForChoice / numObservedTrials

            if numObservedRTsForChoice < 5:
                continue

            quantilesBoundaries = np.quantile(observedRTsForChoice, quantiles)

            observedProportions = \
                np.histogram(observedRTsForChoice, bins=np.concatenate(([0], quantilesBoundaries, [100])))[
                    0] / numObservedTrials

            if numObservedRTsForChoice == 5:
                observedProportions = observedProportionsChoiceWise * observedPOfThisChoice

            modelTrialsForChoice = modelTrials[modelTrials[:, 0] == choice]
            modelRTsForChoice = modelTrialsForChoice[:, 1]
            numModelRTsForChoice = np.size(modelRTsForChoice)
            modelProportions = \
            np.histogram(modelRTsForChoice, bins=np.concatenate(([0], quantilesBoundaries, [100])))[
                0] / numModelTrials

            totalCost += -2 * numObservedTrials * np.sum(modelProportions * np.log(observedProportions))

    totalCost += 2 * numParams
    return totalCost

chiSquaredValues = {}
AICValues = {}

fullModelData = np.genfromtxt("simulatedData/onlyLossAversion.csv", delimiter=',')
fullModelChiSquared = computeChiSquared(fullModelData)
chiSquaredValues['onlyLossAversion'] = fullModelChiSquared
fullModelAIC = computeAIC(fullModelData, 9)
AICValues['onlyLossAversion'] = fullModelAIC

noLossAversionModelData = np.genfromtxt("simulatedData/onlyUnequalAttention.csv", delimiter=',')
noLossAversionChiSquared = computeChiSquared(noLossAversionModelData)
chiSquaredValues['onlyUnequalAttention'] = noLossAversionChiSquared
noLossAversionAIC = computeAIC(noLossAversionModelData, 8)
AICValues['onlyUnequalAttention'] = noLossAversionAIC


noPredecisionalBiasData = np.genfromtxt("simulatedData/onlyStartingBias.csv", delimiter=',')
noPredecisionalBiasChiSquared = computeChiSquared(noPredecisionalBiasData)
chiSquaredValues['onlyStartingBias'] = noPredecisionalBiasChiSquared
noPredecisionalBiasAIC = computeAIC(noPredecisionalBiasData, 8)
AICValues['onlyStartingBias'] = noPredecisionalBiasAIC


noFixedUtilityBiasData = np.genfromtxt("simulatedData/onlyUnequalThreshold.csv", delimiter=',')
noFixedUtilityBiasChiSquared = computeChiSquared(noFixedUtilityBiasData)
chiSquaredValues['onlyUnequalThreshold'] = noFixedUtilityBiasChiSquared
noFixedUtilityBiasAIC = computeAIC(noFixedUtilityBiasData, 8)
AICValues['onlyUnequalThreshold'] = noFixedUtilityBiasAIC


noFixedUtilityBiasData = np.genfromtxt("simulatedData/onlyFixedUtilityBias.csv", delimiter=',')
noFixedUtilityBiasChiSquared = computeChiSquared(noFixedUtilityBiasData)
chiSquaredValues['onlyFixedUtilityBias'] = noFixedUtilityBiasChiSquared
noFixedUtilityBiasAIC = computeAIC(noFixedUtilityBiasData, 8)
AICValues['onlyFixedUtilityBias'] = noFixedUtilityBiasAIC


df_ChiSquared = pd.DataFrame.from_dict(chiSquaredValues, orient='index')
df_AIC = pd.DataFrame.from_dict(AICValues, orient='index')

df = pd.concat([df_ChiSquared, df_AIC], axis=1)
df.columns = ['chiSquared', 'AIC']

df.to_csv("modelCosts.csv")