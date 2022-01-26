'''
Main implementation of the Leaky, Competing, Accumulator
'''

import numpy as np
import math


def getValueInput2Attributes2Choices(attributeProbabilities, choiceAttributes):
    '''
    A different kind of getValueInput function, which uses the value of the other option as a reference point
    (not used in the final version of the paper).
    '''
    selectedAttribute = np.random.RandomState().choice(2, 1, p=attributeProbabilities)
    rawValueInput = choiceAttributes[:, selectedAttribute].T
    processedValueInput = rawValueInput @ np.array([[1, -1], [-1, 1]])

    return processedValueInput


class UtilityFunction:
    '''
    Implementation of one particular utility function (not really used in the paper). Quite self-explanatory.
    '''
    def __init__(self, k, gamma):
        self.k = k
        self.gamma = gamma

    def __call__(self, x):
        if x >= 0:
            return math.log(1 + self.k * x)
        else:
            return -1 * self.gamma * math.log(1 - self.k * x)


class GetValueInputZeroReference:
    '''
    LCA is making a choice between several OPTIONS, and each OPTION has several ATTRIBUTES with respective ATTRIBUTE VALUES.
    At any given time step, the LCA decides to focus on just one attribute (based on its attributeProbability). This function
    takes all options with all their attribute values and probabilities of all attributes, and selects one attribute to focus on,
    then returns the value of that attribute for all options.
    '''
    def __init__(self, utilityFunction):
        '''
        :param utilityFunction: the utility function (for example: x, x > 0 and lambda*x, x < 0)
        '''
        self.utilityFunction = utilityFunction

    def __call__(self, attributeProbabilities, choiceAttributes):
        '''
        :param attributeProbabilities: probability the LCA will focus on a given attribute (for example, P(focus on gain) = P(focus on loss) = 0.5)
        :param choiceAttributes: value for every attribute for every option
        :return:
        '''
        numAttributes = np.size(attributeProbabilities)
        selectedAttribute = np.random.RandomState().choice(numAttributes, 1, p=attributeProbabilities)
        rawValueInput = choiceAttributes[:, selectedAttribute].T
        valueInput = np.vectorize(self.utilityFunction)(rawValueInput)

        return valueInput


class PrepareRecurrentWeights:
    '''
    Creates recurrent weights matrix used in LCA simulations
    size: dimension of the matrix (single value, since it is a square matrix)
    decay: decay parameter
    competition: competition parameter
    '''
    def __init__(self, size):
        self.size = size

    def __call__(self, decay, competition):
        diagonalMask = np.eye(self.size)
        offDiagonalMask = np.ones((self.size, self.size)) - np.eye(self.size)
        weightMatrix = -1 * (diagonalMask * decay + offDiagonalMask * competition)

        return weightMatrix


class RunLCASimulation:
    '''
    This is the core of the LCA.
    '''
    def __init__(self, getValueInput, getNoise, prepareRecurrentWeights, maxTimeSteps, deltaT):
        '''
        :param getValueInput: the function used to select a value for each option at a given time step based on attribute probabilities and attribute values (usually GetValueInputZeroReference)
        :param getNoise: function to sample (Gaussian) noise from (usually from LUT.py)
        :param prepareRecurrentWeights: function (passed to this class) used to create a matrix that multiply previous activations with decay/competition (usually PrepareRecurrentWeights)
        :param maxTimeSteps: the maximum number of time steps allowed for one trial (if no decision is reached in this period, the model returns -1)
        :param deltaT: delta-T used for simulations
        '''
        self.getValueInput = getValueInput
        self.getNoise = getNoise
        self.prepareRecurrentWeights = prepareRecurrentWeights
        self.maxTimeSteps = maxTimeSteps
        self.deltaT = deltaT

    def __call__(self, attributeProbabilities, choiceAttributes, startingActivation, decay, competition, constantInput,
                 noiseStdDev, nonDecisionTime, threshold):
        '''
        :param attributeProbabilities: probabilities of all attributes in the given trial
        :param choiceAttributes: values of all attributes of all options in the given trial
        :param startingActivation: starting activations of all accumulators
        :param decay: decay parameter
        :param competition: competition parameter
        :param constantInput: constant inputs (can be different for every accumulator)
        :param noiseStdDev: standard deviation of noise
        :param nonDecisionTime: non-decision component of the response time
        :param threshold: thresholds of all accumulators (can be different for every accumulator)
        :return:
        '''
        recurrentWeights = self.prepareRecurrentWeights(decay, competition)

        previousActivation = startingActivation
        allActivations = startingActivation
        reactionTime = 0
        response = -1

        for timeStep in range(self.maxTimeSteps):
            valueInput = self.getValueInput(attributeProbabilities, choiceAttributes) # one value for each option,
            # passed to next layers to compare different options (such as acceptance and rejection)
            noise = self.getNoise(noiseStdDev)
            currentActivation = previousActivation + self.deltaT * (previousActivation @ recurrentWeights + valueInput
                                                                    + noise + constantInput) # simply an implementation
            # of the LCA equation. Note that decay and competition are stored in the recurrentWeights matrix.
            currentActivation[currentActivation < 0] = 0 # clip negative activations to zero
            allActivations = np.vstack((allActivations, currentActivation)) # maintain a log of activations at all timesteps
            reactionTime += self.deltaT
            previousActivation = currentActivation
            if (currentActivation >= threshold).any():  # if a threshold is reached
                response = np.argwhere(currentActivation >= threshold)[0, 1]
                break

        reactionTime += nonDecisionTime

        return (allActivations, reactionTime, response)