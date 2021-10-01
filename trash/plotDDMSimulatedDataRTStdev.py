import numpy as np
from matplotlib import pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

data = np.genfromtxt("../src/risk_data.csv", delimiter=',')[1:, :]

allGainValues = list(range(10, 110, 10))
allLossValues = list(range(-100, 0, 10))

allStdevs = []

for participantIndex in range(1, 50):
    participantData = data[data[:, 0] == participantIndex][:]
    print(participantData)
    for gain in allGainValues:
        for loss in allLossValues:
            dataForThisGain = participantData[participantData[:, -2] == gain][:]
            dataForThisGamble = dataForThisGain[dataForThisGain[:, -1] == loss][:]
            RTs = dataForThisGamble[:, 2]
            allStdevs.append(np.std(RTs))

plt.hist(allStdevs)
plt.xlabel("Standard Dev. of RTs for a particular gamble and a particular participant")
plt.ylabel("Frequency")
plt.show()
