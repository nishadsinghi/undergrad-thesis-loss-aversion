import pickle
from matplotlib import pyplot as plt
import numpy as np

allFinalParams = np.zeros(6)

for trial in range(1, 11):
    readFile = open("BTPpickles/{}.pickle".format(trial), 'rb')
    record = pickle.load(readFile)
    finalParams = record[-1, :]

    allFinalParams = np.vstack((allFinalParams, finalParams))

allFinalParams = (allFinalParams[1:, :]).T

subPlotTitles = ["decay", "competition", "noiseStdDev", "nonDecisionTime", "threshold", "constantInput"]

fig = plt.figure()
for subplotIndex in range(1, 7):
    axForDraw = fig.add_subplot(2, 3, subplotIndex)
    axForDraw.hist(allFinalParams[subplotIndex-1, :])
    axForDraw.set_title(subPlotTitles[subplotIndex-1])
    axForDraw.set_xlabel("Value")
    axForDraw.set_ylabel("Frequency")

plt.show()