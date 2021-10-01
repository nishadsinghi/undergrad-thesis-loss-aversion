from matplotlib import pyplot as plt
import numpy as np
import time


def trueFunction(x):
    y = x**3 + 4*(x**2) + x - 5
    return y


def polynomial(x, coefficients):
    y = 0
    for i in range(np.size(coefficients)):
        y += coefficients[i] * (x**i)

    return y


class Cost:
    def __init__(self, trueY):
        self.trueY = trueY

    def __call__(self, predictions):
        difference = np.asarray(predictions) - np.asarray(trueY)
        differenceSquared = np.square(difference)
        meanSqDiff = np.mean(differenceSquared)

        return meanSqDiff


# generate ground truth samples
numSamples = 10000
low = -5
high = 5
trueX = np.random.uniform(low=low, high=high, size=numSamples)
trueY = [trueFunction(x) for x in trueX]

# initial value of variance
variance = 1.5

# cost function
computeCost = Cost(trueY)

# initialize
coefficients = np.random.normal(loc=0, scale=3, size=4)
bestCoefficients = coefficients
initialPredictions = [polynomial(x, coefficients) for x in trueX]
initialCost = computeCost(initialPredictions)
minimumCost = initialCost

recordOfBestCoefficients = bestCoefficients
recordOfCost = [minimumCost]

numIterations = 50000
startTime = time.time()
for iteration in range(numIterations):
    # decrease variance
    if iteration % 100 == 0:
        variance = 0.99 * variance

    # propose new coefficients
    coefficients = np.random.normal(loc=bestCoefficients, scale=variance)

    # predict
    predictions = [polynomial(x, coefficients) for x in trueX]

    # compute cost
    cost = computeCost(predictions)
    recordOfCost.append(cost)

    # update
    if(cost < minimumCost):
        bestCoefficients = coefficients
        minimumCost = cost

    # record best coefficients
    recordOfBestCoefficients = np.vstack((recordOfBestCoefficients, bestCoefficients))

endTime = time.time()
executionTime = endTime - startTime
print("Time for execution of {} iterations = {}".format(numIterations, executionTime))

import pickle
saveFile = open("fitPolynomial.pickle", 'wb')
pickle.dump(recordOfBestCoefficients, saveFile)

allValuesOfA0 = recordOfBestCoefficients[:, 0]
allValuesOfA1 = recordOfBestCoefficients[:, 1]
allValuesOfA2 = recordOfBestCoefficients[:, 2]
allValuesOfA3 = recordOfBestCoefficients[:, 3]
plt.plot(allValuesOfA0)
plt.plot(allValuesOfA1)
plt.plot(allValuesOfA2)
plt.plot(allValuesOfA3)

print("Coefficients: ", coefficients)

plt.show()

