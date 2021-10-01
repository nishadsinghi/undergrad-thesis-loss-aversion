import numpy as np
import math

def getValueInput2Attributes2Choices(attributeProbabilities, choiceAttributes): # this is only a temporary fix; need to write a general function
    selectedAttribute = np.random.choice(2, 1, p=attributeProbabilities)
    rawValueInput = choiceAttributes[:, selectedAttribute].T
    processedValueInput = rawValueInput @ np.array([[1, -1], [-1, 1]])

    return processedValueInput


class UtilityFunction:
    def __init__(self, k, gamma):
        self.k = k
        self.gamma = gamma

    def __call__(self, x):
        if x >= 0:
            return math.log(1 + self.k * x)
        else:
            return -1 * self.gamma * math.log(1 - self.k * x)


class GetValueInputZeroReference:
    def __init__(self, utilityFunction):
        self.utilityFunction = utilityFunction

    def __call__(self, attributeProbabilities, choiceAttributes):    # Need to add terms for d12, d23, so on
        numAttributes = np.size(attributeProbabilities)
        selectedAttribute = np.random.choice(numAttributes, 1, p=attributeProbabilities)
        rawValueInput = choiceAttributes[:, selectedAttribute].T
        valueInput = np.vectorize(self.utilityFunction)(rawValueInput)

        return valueInput


class PrepareRecurrentWeights:
    def __init__(self, size):
        self.size = size

    def __call__(self, decay, competition):
        diagonalMask = np.eye(self.size)
        offDiagonalMask = np.ones((self.size, self.size)) - np.eye(self.size)
        weightMatrix = -1 * (diagonalMask * decay + offDiagonalMask * competition)

        return weightMatrix


class RunLCASimulation:
    def __init__(self, getValueInput, getNoise, prepareRecurrentWeights, maxTimeSteps, deltaT):
        self.getValueInput = getValueInput
        self.getNoise = getNoise
        self.prepareRecurrentWeights = prepareRecurrentWeights
        self.maxTimeSteps = maxTimeSteps
        self.deltaT = deltaT

    def __call__(self, attributeProbabilities, choiceAttributes, startingActivation, decay, competition, constantInput,
                 noiseStdDev, nonDecisionTime, threshold):
        recurrentWeights = self.prepareRecurrentWeights(decay, competition)

        previousActivation = startingActivation
        allActivations = startingActivation
        reactionTime = 0
        response = -1

        for timeStep in range(self.maxTimeSteps):
            valueInput = self.getValueInput(attributeProbabilities, choiceAttributes)
            noise = self.getNoise(noiseStdDev)
            currentActivation = previousActivation + self.deltaT * (previousActivation @ recurrentWeights + valueInput
                                                                    + noise + constantInput)
            currentActivation[currentActivation < 0] = 0
            allActivations = np.vstack((allActivations, currentActivation))
            reactionTime += self.deltaT
            previousActivation = currentActivation
            if (currentActivation >= threshold).any():
                response = np.argwhere(currentActivation >= threshold)[0, 1]
                break

        reactionTime += nonDecisionTime

        # # this is a temporary fix - need to discuss this with advisors
        # if response == -1:
        #     print("No decision reached by LCA: choosing random response!")
        #     response = np.random.randint(0, np.size(startingActivation))
        #     print("RESPONSE:", response)

        return (allActivations, reactionTime, response)