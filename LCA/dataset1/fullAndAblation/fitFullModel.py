'''
Fits an LCA model with loss aversion + predecisional bias + fixed utility bias
'''

from multiprocessing import Pool    # using this library to run multiple chains in parallel

def runMetropolis(chainIndex):
    # this function contains all the code, and I had to make it a function so that the library could run it several
    # times in parallel

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
    data = np.genfromtxt("risk_data_cleaned.csv", delimiter=',')[:, 1:]
    allStakes = np.unique(data[:, -2:], axis=0)

    np.random.seed() # without this, all chains somehow sample the same values, making them identical to each other

    # set up the look-up table to sample from Gaussian distribution
    LUTInterval = 0.0001
    numAccumulators = 2
    stdNormalLUT = prepareStdNormalLUT(LUTInterval)
    sampleFromLUT = SampleFromLUT(stdNormalLUT)
    sampleFromZeroMeanLUT = lambda stdDev: sampleFromLUT(0, stdDev, numAccumulators)

    # set-up the LCA
    identityUtilityFunction = lambda x: x   # utility function with lambda = 1
    getValueInput = GetValueInputZeroReference(identityUtilityFunction)

    maxTimeSteps = 750
    deltaT = 0.02
    prepareRecurrentWeights = PrepareRecurrentWeights(numAccumulators)
    lca = RunLCASimulation(getValueInput, sampleFromZeroMeanLUT, prepareRecurrentWeights, maxTimeSteps, deltaT)
    # instantiate the RunLCASimulation class -- this is what you would use as the LCA function henceforth

    def getStartingActivation(startingBias, threshold):
        # This function is useful for models with pre-decisional bias
        # the starting bias can be a value between -1 and +1
        # negative values indicate that the starting value of the first accumulator > 0 and the starting value of the
        # second accumulator = 0. Starting value of the first accumulator is |starting bias| * threshold
        # positive values indicate that the starting value of the first accumulator = 0 and the starting value of the
        # second accumulator > 0. Starting value of the second accumulator is |starting bias| * threshold
        if startingBias < 0:
            return [-1*startingBias*threshold, 0]
        else:
            return [0, startingBias*threshold]

    getChoiceAttributes = lambda gain, loss: np.array([[0, 0], [gain, loss]]) # encodes the fact that there are two
    # options: rejection (both attributes, attribute1 (gain) = 0 and attribute2 (loss) = 0), and acceptance
    # (attribute1 = gain, attribute2 = loss)
    getAllThresholds = lambda threshold: [threshold]*numAccumulators # if you want a model in which both accumulators
    # have the same threshold
    attributeProbabilities = [0.5, 0.5] # equal attention to gain and loss at any time step

    lcaWrapper = lambda gain, loss, decay, competition, noiseStdDev, nonDecisionTime, threshold, constantInput1, lossWeight, startingBias, constantInput2: \
        lca(attributeProbabilities, getChoiceAttributes(gain, lossWeight*loss), getStartingActivation(startingBias, threshold), decay, competition, (constantInput1, constantInput2),
            noiseStdDev, nonDecisionTime, getAllThresholds(threshold)) # simply a wrapper to make calling the LCA
    # function a bit convenient. It takes in all the parameters for a trial and returns the response and reaction time.


    def selectConditionTrials(data, gain, loss): # reads all trials with a given set of potential gain and loss
        selectedData = data[data[:, 2] == gain][:]
        selectedData = selectedData[selectedData[:, 3] == loss][:]
        return selectedData

    def computePData(data, gain, loss): # computes the acceptance rate for a given set of potential gain and loss
        conditionTrialData = selectConditionTrials(data, gain, loss)
        responses = conditionTrialData[:, 0]
        return np.mean(responses)

    def computeRTStatsData(data, gain, loss, choice): # computes stats on RT data for a given set of gain, loss, choice
        # (I don't think this is being used anywhere, but leaving it in to not break anything)
        conditionTrialData = selectConditionTrials(data, gain, loss)
        dataForThisChoice = conditionTrialData[conditionTrialData[:, 0] == choice]
        if np.shape(dataForThisChoice)[0] == 0:
            return (0, 0, 0)
        reactionTimes = dataForThisChoice[:, 1]
        return (np.mean(reactionTimes), np.std(reactionTimes), skew(reactionTimes))

    def filterFunction(tup):
        # function used in the class CostFunction. Basically returns False if the last value in a tuple is = -1. This
        # is used to remove trials where the model returned response -1 (did not reach a decision).
        if tup[-1] != -1:
            return True
        else:
            return False


    # define the cost function
    class CostFunction:
        def __init__(self, numSimulationsPerCondition, allStakes):
            '''
            :param numSimulationsPerCondition: number of simulations to run per combination of potential gain, loss
            :param allStakes: all possible combinations of potential gain and loss
            '''
            self.numSimulationsPerCondition = numSimulationsPerCondition
            self.allStakes = allStakes

        def __call__(self, parameters):
            '''
            :param parameters: the set of parameters for which you are computing the cost
            :return:
            '''

            # first, we will simulate data from the LCA for the given set of parameters
            allModelData = np.zeros(4) # initialize all simulated data from the model
            for stakes in self.allStakes: # iterate over all possible gain, loss combinations
                gainValue, lossValue = stakes
                allSimulations = [lcaWrapper(gainValue, lossValue, *parameters) for _ in
                                  range(self.numSimulationsPerCondition)] # run LCA for #numSimulationsPerCondition
                # times for the given values of gain, loss and the parameters
                allValidResponseSimulations = list(filter(filterFunction, allSimulations)) # retain the trials with
                # response either 0 or 1 (not -1)
                numValidResponses = len(allValidResponseSimulations) # the number of valid responses from previous step
                if numValidResponses < self.numSimulationsPerCondition / 3: # if the number of invalid responses is
                    # more than 30%, just reject this set of parameters straight away
                    return (-1, parameters[3]) # cost = -1 is kind of an invalid value (to indicate that we don't want
                    # these parameters)
                # the next few lines basically take the simulated data and format it properly
                _, allModelRTs, allModelResponses = zip(*allValidResponseSimulations)
                modelStakes = np.hstack(
                    (np.full((numValidResponses, 1), gainValue), np.full((numValidResponses, 1), lossValue)))
                modelDataForStakes = np.hstack(
                    (np.array(allModelResponses).reshape(-1, 1), np.array(allModelRTs).reshape(-1, 1), modelStakes))
                allModelData = np.vstack((allModelData, modelDataForStakes))

            allModelData = allModelData[1:, :] # remove the first row because it is empty (used for initialization)

            actualDataMeanRT = np.mean(data[:, 1]) # RT mean of the experimental data
            simDataMeanRT = np.mean(allModelData[:, 1]) # RT mean of the simulated data
            delta = simDataMeanRT - actualDataMeanRT # difference of these two

            # the logic behind the next 4 lines is that the non-decision component of the reaction time doesn't affect
            # the shape of the RT distribution. During fitting, what we really want to fit is the shape of the RT
            # distribution. The distribution can be shifted to the left or right by changing this non-decision RT
            # component. So, for any given shape of the distribution, we can compute the value of this non-decision
            # component that brings the simulated distribution close to the experimentally observed distribution
            # (in particular, it brings the means of these distributions to the same point). This way, you don't have to
            # 'fit' this particular parameter, since you can just set it to the value that makes the mean of the
            # simulated distribution equal to the mean of the experimental distribution.
            if delta > parameters[3]:
                delta = parameters[3]

            allModelData[:, 1] = allModelData[:, 1] - delta

            # Now, we will use the simulated data to compute the chi^2 cost
            totalCost = 0
            quantiles = np.array([0.1, 0.3, 0.5, 0.7, 0.9]) # quantiles of the chi^2 function
            observedProportionsChoiceWise = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1]) # this is to cover some edge cases
            for stakes in self.allStakes: # loop over all combinations of possible gain and loss
                gain, loss = stakes
                observedTrials = selectConditionTrials(data, gain, loss)
                numObservedTrials = np.shape(observedTrials)[0]
                modelTrials = selectConditionTrials(allModelData, gain, loss)
                numModelTrials = np.shape(modelTrials)[0]
                for choice in range(2): # loop over choice = 0 (reject) and 1 (accept)
                    observedTrialsForChoice = observedTrials[observedTrials[:, 0] == choice]
                    observedRTsForChoice = observedTrialsForChoice[:, 1]
                    numObservedRTsForChoice = np.size(observedRTsForChoice)
                    observedPOfThisChoice = numObservedRTsForChoice / numObservedTrials

                    if numObservedRTsForChoice < 5: # less than 5 trials --> can't compute quantile boundaries
                        continue # skip this combination of gain, loss, choice

                    quantilesBoundaries = np.quantile(observedRTsForChoice, quantiles)

                    observedProportions = \
                        np.histogram(observedRTsForChoice, bins=np.concatenate(([0], quantilesBoundaries, [100])))[
                            0] / numObservedTrials # proportions of experimental RTs in all quantiles

                    if numObservedRTsForChoice == 5 or 0 in observedProportions: # some edge cases
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
            return (totalCost, parameters[3] - delta)


    costFunction = CostFunction(75, allStakes)
    bounds = ((0, 5), (0, 5), (0, 150), (0, 2), (0, 75), (0, 150), (0, 5), (-1, 1), (0, 150))
    # ranges for model parameters: decay, competition, noiseStdDev, nonDecisionTime, threshold, constantInput1,
    # lossWeight, startingBias, constantInput2
    rangeOfParams = np.diff((np.array(bounds))).flatten()
    initialParameterVariance = rangeOfParams/(1.5*numTotalChains)
    parameterVarianceMultiplier = np.full(np.shape(initialParameterVariance), 0.99)
    # multiplies (shrinks) the variance by a factor of 0.99 after every fixed number of iterations


    SAVE_DIRECTORY = 'savedModels/fullModel'.format(str(bounds))
    if not os.path.exists(SAVE_DIRECTORY):
        os.makedirs(SAVE_DIRECTORY)


    SAVEFILE = os.path.join(SAVE_DIRECTORY, '{}.pickle'.format(str(chainIndex)))

    def pickleRecord(record): # takes the record and stores it as a pickle file
        saveFile = open(SAVEFILE, 'wb')
        pickle.dump(record, saveFile)
        saveFile.close()

    # the Monte-Carlo Algorithm takes the current best set of parameters and samples another set of parameters from a
    # (clipped) Gaussian centered at the current best parameters. If the new parameters are better, they are updated.

    # This function generates a new set of parameters (proposal)
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

    try: # if a pickle file already exists, resume its progress
        saveFile = open(SAVEFILE, "rb")
        record = pickle.load(saveFile) # read its record
        bestParameters = record[-1, :-1] # get the best parameters from the record
        minimumCost = record[-1, -1] # get the corresponding best cost from the record
        # print("Best parameters: ", bestParameters, " Best cost: ", minimumCost)
        numIterationsPrevious = np.shape(record)[0] # get the number of iterations already run in the record (length of record)
        # ("NumIterations: ", numIterationsPrevious)
        parameterVarianceMultiplier = parameterVarianceMultiplier**(numIterationsPrevious//100) # compute the multiplier for the given number of iterations previously run
        # print("Num Iterations: ", numIterationsPrevious, " Multiplier: ", parameterVarianceMultiplier)
        parameterVariance = np.multiply(initialParameterVariance, parameterVarianceMultiplier)
    except: # a pickle file does not exist, start the fitting process from the first iteration
        numIterationsPrevious = 0
        # the next few lines generate a starting point for the fitting algorithm. They repeatedly sample points and
        # stop as soon as a set of parameters is found with cost != -1 (i.e., a valid set of parameters is found)
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

        # initialize some variables
        bestParameters = initialParameterValues
        parameterVariance = np.asarray(initialParameterVariance)
        minimumCost = initialCost
        record = np.hstack((bestParameters, minimumCost))

        pickleRecord(record)

    minimizationIteration = numIterationsPrevious
    pickleFlag = False

    # now the fitting process starts
    while True: # keep running the fitting process as long as you want (the cluster will stop it anyway after a point)
        minimizationIteration += 1 # update iteration
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

        print('Chain: {}, iteration: {}, Best cost: {}'.format(chainIndex, minimizationIteration, minimumCost))


numTotalChains = 10

p = Pool(numTotalChains)
p.map(runMetropolis, range(numTotalChains))