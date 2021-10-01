import pickle
from matplotlib import pyplot as plt

readFile = open("BTPpickles/6.pickle", 'rb')
record = pickle.load(readFile)

print("Final Values: ", record[-1, :])

allValuesOfA0 = record[:, 0]
allValuesOfA1 = record[:, 1]
allValuesOfA2 = record[:, 2]
allValuesOfA3 = record[:, 3]
allValuesOfA4 = record[:, 4]


ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
ax1.set_title("Decay")
ax1.set_xlabel("iteration")
ax2 = plt.subplot2grid((2,6), (0,2), colspan=2)
ax2.set_title("Competition")
ax2.set_xlabel("iteration")
ax3 = plt.subplot2grid((2,6), (0,4), colspan=2)
ax3.set_title("Noise Std Dev")
ax3.set_xlabel("iteration")
ax4 = plt.subplot2grid((2,6), (1,1), colspan=2)
ax4.set_title("Non-decision time")
ax4.set_xlabel("iteration")
ax5 = plt.subplot2grid((2,6), (1,3), colspan=2)
ax5.set_title("Decision threshold")
ax5.set_xlabel("iteration")

ax1.plot(allValuesOfA0)
ax2.plot(allValuesOfA1)
ax3.plot(allValuesOfA2)
ax4.plot(allValuesOfA3)
ax5.plot(allValuesOfA4)

plt.show()