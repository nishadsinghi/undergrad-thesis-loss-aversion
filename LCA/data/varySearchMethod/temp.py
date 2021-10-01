import pickle
import numpy as np
from os import listdir
from os.path import isfile, join

PATH_OF_DATA = '.'
fileNames = [file for file in listdir(PATH_OF_DATA) if ".pickle" in file]

def returnNumIterations(fileName):
    filePath = join(PATH_OF_DATA, fileName)
    read = open(filePath, "rb")
    record = pickle.load(read)
    return np.shape(record)[0], record[-1]

[print(fileName, returnNumIterations(fileName)) for fileName in fileNames]
