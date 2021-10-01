import pickle
import numpy as np
from os import listdir
from os.path import isfile, join

PATH_OF_DATA = '.'#"../data/varyCostFunction/furtherFineTuningCorona"
fileNames = [file for file in listdir(PATH_OF_DATA) if ".pickle" in file]

def returnNumIterations(fileName):
    filePath = join(PATH_OF_DATA, fileName)
    read = open(filePath, "rb")
    record = pickle.load(read)
    return np.shape(record)[0], record[-1]

minimumCost = np.inf
for fileName in fileNames:
    numIterations, bestFitParamsAndCost = returnNumIterations(fileName)
    cost = bestFitParamsAndCost[-1]
    if cost < minimumCost:
        minimumCost = cost
        bestFitFileName = fileName
        bestNumIterations = numIterations
        bestParams = bestFitParamsAndCost[:-1]

print("best cost = {} for number of iterations = {}".format(minimumCost, bestNumIterations))
print("best parameters: ", bestParams)