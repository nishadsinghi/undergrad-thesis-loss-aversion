from multiprocessing import Pool

def runMetropolis(chainIndex):
    import sys
    import os
    DIRNAME = os.path.dirname(__file__)
    sys.path.append(os.path.join(DIRNAME, "..", "..", "src"))

    from LCA import PrepareRecurrentWeights, RunLCASimulation, GetValueInputZeroReference, \
        getValueInput2Attributes2Choices
    from LUT import prepareStdNormalLUT, SampleFromLUT

    import numpy as np
    import time
    import pickle
    from scipy.stats import skew, truncnorm, chisquare

    # read observed data
    data = np.genfromtxt("data_preprocessed.csv", delimiter=',')[1:, 2:]
    allStakes = np.unique(data[:, -2:], axis=0)

    np.random.seed()

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

    def getStartingActivation(startingBias, threshold):
        if startingBias < 0:
            return [-1*startingBias*threshold, 0]
        else:
            return [0, startingBias*threshold]

    getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]])
    getAllThresholds = lambda threshold: [threshold]*numAccumulators
    attributeProbabilities = [0.5, 0.5]

    lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, startingBias, lossWeight, constantInput: \
        lca(attributeProbabilities, getChoiceAttributes(gain, lossWeight*loss), getStartingActivation(startingBias, threshold), decay, competition, constantInput,
            noiseStdDev, nonDecisionTime, getAllThresholds(threshold))


    def selectConditionTrials(data, gain, loss):
        selectedData = data[data[:, 2] == gain][:]
        selectedData = selectedData[selectedData[:, 3] == loss][:]
        return selectedData

    def computePData(data, gain, loss):
        conditionTrialData = selectConditionTrials(data, gain, loss)
        responses = conditionTrialData[:, 0]
        return np.mean(responses)

    def computeRTStatsData(data, gain, loss, choice):
        conditionTrialData = selectConditionTrials(data, gain, loss)
        dataForThisChoice = conditionTrialData[conditionTrialData[:, 0] == choice]
        if np.shape(dataForThisChoice)[0] == 0:
            return (0, 0, 0)
        reactionTimes = dataForThisChoice[:, 1]
        return (np.mean(reactionTimes), np.std(reactionTimes), skew(reactionTimes))

    def filterFunction(tup):
        if tup[-1] != -1:
            return True
        else:
            return False


    # define the cost function
    class CostFunction:
        def __init__(self, numSimulationsPerCondition, allStakes):
            self.numSimulationsPerCondition = numSimulationsPerCondition
            self.allStakes = allStakes

        def __call__(self, parameters):
            allModelData = np.zeros(4)
            for stakes in self.allStakes:
                gainValue, lossValue = stakes
                allSimulations = [lcaWrapper(gainValue, lossValue, *parameters) for _ in
                                  range(self.numSimulationsPerCondition)]
                allValidResponseSimulations = list(filter(filterFunction, allSimulations))
                numValidResponses = len(allValidResponseSimulations)
                if numValidResponses < self.numSimulationsPerCondition / 3:
                    return (-1, parameters[3])
                _, allModelRTs, allModelResponses = zip(*allValidResponseSimulations)
                modelStakes = np.hstack(
                    (np.full((numValidResponses, 1), gainValue), np.full((numValidResponses, 1), lossValue)))
                modelDataForStakes = np.hstack(
                    (np.array(allModelResponses).reshape(-1, 1), np.array(allModelRTs).reshape(-1, 1), modelStakes))
                allModelData = np.vstack((allModelData, modelDataForStakes))

            allModelData = allModelData[1:, :]

            actualDataMeanRT = np.mean(data[:, 1])
            simDataMeanRT = np.mean(allModelData[:, 1])
            delta = simDataMeanRT - actualDataMeanRT
            if delta > parameters[3]:
                delta = parameters[3]

            allModelData[:, 1] = allModelData[:, 1] - delta

            totalCost = 0
            quantiles = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
            observedProportionsChoiceWise = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
            for stakes in self.allStakes:
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

                    expectedFrequencies = \
                        np.histogram(observedRTsForChoice, bins=np.concatenate(([0], quantilesBoundaries, [100])))[
                            0] / numObservedTrials

                    if numObservedRTsForChoice == 5 or 0 in expectedFrequencies:
                        expectedFrequencies = observedProportionsChoiceWise * observedPOfThisChoice

                    modelTrialsForChoice = modelTrials[modelTrials[:, 0] == choice]
                    modelRTsForChoice = modelTrialsForChoice[:, 1]
                    modelFrequencies = \
                    np.histogram(modelRTsForChoice, bins=np.concatenate(([0], quantilesBoundaries, [100])))[
                        0] / numModelTrials

                    totalCost += chisquare(modelFrequencies, expectedFrequencies)[0]
            return (totalCost, parameters[3] - delta)


    costFunction = CostFunction(75, allStakes)
    bounds = ((0, 5), (0, 5), (0, 50), (0, 2), (0, 30), (-1, 1), (0, 7), (0, 50))   #decay, competition, noiseStdDev, nonDecisionTime, threshold, startingBias, lossWeight, constantInput
    rangeOfParams = np.diff((np.array(bounds))).flatten()
    initialParameterVariance = rangeOfParams/(1.5*numTotalChains)
    parameterVarianceMultiplier = np.full(np.shape(initialParameterVariance), 0.99)


    SAVE_DIRECTORY = 'savedModels/noFixedUtilityBias'.format(str(bounds))
    if not os.path.exists(SAVE_DIRECTORY):
        os.makedirs(SAVE_DIRECTORY)



    SAVEFILE = os.path.join(SAVE_DIRECTORY, '{}.pickle'.format(str(chainIndex)))

    def pickleRecord(record):
        saveFile = open(SAVEFILE, 'wb')
        pickle.dump(record, saveFile)
        saveFile.close()

    def generateProposalParams(currentParams, variances):
        np.random.seed()
        newParams = []
        for i in range(len(currentParams)):
            myclip_a = bounds[i][0]
            myclip_b = bounds[i][1]
            my_mean = currentParams[i]
            my_std = variances[i]
            if my_std == 0:
                newParam = my_mean
            else:
                a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
                newParam = truncnorm.rvs(a, b, loc=my_mean, scale=my_std)
            newParams.append(newParam)

        return newParams

    try:
        saveFile = open(SAVEFILE, "rb")
        record = pickle.load(saveFile)
        bestParameters = record[-1, :-1]
        minimumCost = record[-1, -1]
        # print("Best parameters: ", bestParameters, " Best cost: ", minimumCost)
        numIterationsPrevious = np.shape(record)[0]
        # ("NumIterations: ", numIterationsPrevious)
        parameterVarianceMultiplier = parameterVarianceMultiplier**(numIterationsPrevious//100)
        # print("Num Iterations: ", numIterationsPrevious, " Multiplier: ", parameterVarianceMultiplier)
        parameterVariance = np.multiply(initialParameterVariance, parameterVarianceMultiplier)
    except:
        numIterationsPrevious = 0
        counter = 0
        while True:
            print(counter)
            counter += 1
            # initialParameterValues = generateProposalParams(meanStartingPointForSearch, initialParameterVariance)
            low, high = list(zip(*bounds))
            initialParameterValues = np.random.RandomState().uniform(low, high)
            # print(initialParameterValues)
            initialCost, adjustedContTime = costFunction(initialParameterValues)
            if initialCost != -1:
                initialParameterValues[3] = adjustedContTime
                break

        print("Chain: ", chainIndex, "Initial value identified: ", initialParameterValues, " Cost: ", initialCost)

        bestParameters = initialParameterValues
        parameterVariance = np.asarray(initialParameterVariance)
        minimumCost = initialCost
        record = np.hstack((bestParameters, minimumCost))

        pickleRecord(record)

    minimizationIteration = numIterationsPrevious
    pickleFlag = False
    while True:
        minimizationIteration += 1
        startTime = time.time()
        # decrease the value of variance
        if minimizationIteration % 100 == 0 and minimizationIteration != 0:
            parameterVariance = parameterVarianceMultiplier * parameterVariance

        # propose new parameters
        newParameters = generateProposalParams(bestParameters, parameterVariance)

        # compute cost
        cost, adjustedNonDecisionTime = costFunction(newParameters)

        if cost != -1:
            # update parameters
            if cost < minimumCost:
                pickleFlag = True
                # print("updating parameters and cost")
                bestParameters = newParameters
                bestParameters[3] = adjustedNonDecisionTime
                minimumCost = cost
            else:
                pickleFlag = False

        # record best parameters
        iterationRecord = np.hstack((bestParameters, minimumCost))
        record = np.vstack((record, iterationRecord))

        if pickleFlag:
            pickleRecord(record)


        # time for iteration
        endTime = time.time()
        iterationTime = endTime - startTime
        # print("Iteration: {} Time: {}".format(minimizationIteration, iterationTime))
        # print("Proposed Value of params: ", newParameters)
        # print("Best Value of params: ", bestParameters)
        # print("Cost: ", cost)
        # print("Best cost: ", minimumCost)
        # print("-------------------------------")


        print('Chain: {}, iteration: {}, Best cost: {}'.format(chainIndex, minimizationIteration, minimumCost))


numTotalChains = 10

p = Pool(numTotalChains)
p.map(runMetropolis, range(numTotalChains))