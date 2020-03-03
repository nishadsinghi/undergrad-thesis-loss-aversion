from psychopy import core, visual, gui, data, event  # import some libraries from PsychoPy
import itertools
import random
from psychopy.tools.filetools import fromFile, toFile
import time
try:  # try to get a previous parameters file

# present a dialogue to change params

# make a text file to save data

############################################################################################

# create a window

# randomly decide which key corresponds to accept 
responseCorrespondingToUp = random.randrange(2)
responseCorrespondingToDown = (responseCorrespondingToUp + 1) % 2

if responseCorrespondingToUp == 1:
    instructionMessage = "Press UP to ACCEPT the gamble. Press DOWN to REJECT the gamble. "
elif responseCorrespondingToUp == 0:
    instructionMessage = "Press DOWN to ACCEPT the gamble. Press UP to REJECT the gamble. "

wrapWidth = 35
# print text for a gamble on screen
def printGambleValues(combination):
    gamble = combination[0]
    orientation = combination[1]
    
    instructions = visual.TextStim(win, pos=[0, 10], text=instructionMessage, height=0.8, wrapWidth=wrapWidth)
    
    if orientation == 'increaseLeft':
        leftValue = '+' + str(gamble[0])
        rightValue = str(gamble[1])
    elif orientation == 'increaseRight':
        rightValue = '+' + str(gamble[0])
        leftValue = str(gamble[1])
        
    height = 4.5
    message2 = visual.TextStim(win, pos=[10, 0],text=rightValue, height=height)
    instructions.draw()
    message1.draw()
    message2.draw()
    
# display feedback
feedbackDuration = 0.5
def displayFeedback(response):
    if response == 1:    
        feedback = visual.Circle(win=win, units="pix", radius=15, fillColor=[0, 0, 0], lineColor=[0, 1, 0])
        feedback.draw()
    elif response == 0:
        feedbackLine1 = visual.Line(win=win,units="pix",lineColor=[1, 0, 0])
        feedbackLine1.start = [-15, -15]
        feedbackLine1.end = [15, 15]
        
        feedbackLine2 = visual.Line(win=win,units="pix",lineColor=[1, 0, 0])
        feedbackLine2.start = [-15, 15]
        feedbackLine2.end = [15, -15]
        
        feedbackLine1.draw()
        feedbackLine2.draw()
    
# display fixation
def displayFixation(fixationDuration):
    fixation = visual.GratingStim(win, color=1, colorSpace='rgb',
                              tex=None, mask='cross', size=1)
    fixation.draw()
    win.flip()
    core.wait(fixationDuration)
        
# generate all combinations
allIncreaseValues = [1, 5, 10, 15, 20, 25]
allDecreaseValues = [-1, -5, -10, -15, -20, -25]
allCombinations = list(itertools.product(allIncreaseValues, allDecreaseValues))
allCombinationsAlternatePresentLocations = list(itertools.product(allCombinations, ('increaseLeft', 'increaseRight')))
random.shuffle(allCombinationsAlternatePresentLocations)

# Welcome message
welcomeMessage = visual.TextStim(win, pos=[0, 0], text="Welcome! You will now be playing a sequence of gambles. " + instructionMessage + "Press any key when you are ready to begin!", height=1.5, wrapWidth=wrapWidth)
welcomeMessage.draw()
win.flip()

event.waitKeys()

# present gambles
fixationDuration = 1
interBlockBreakDuration = 15

numBlocks = 2
numGambles = len(allCombinationsAlternatePresentLocations)
blockLength = numGambles // numBlocks
print("Length of block = ", blockLength)
numTrials = 0
for combination in allCombinationsAlternatePresentLocations:
    if numTrials % blockLength == 0 and numTrials != 0:
        print("Displaying block at numTrial", numTrials)
        displayFixation(interBlockBreakDuration)    
    
    # display stimulus
    printGambleValues(combination)
    win.flip()
    startTime = time.time()
    
    # get response
    thisResp=None
                endTime = time.time()
                endTime = time.time()
    reactionTime = endTime - startTime
    
    printGambleValues(combination)
    displayFeedback(thisResp)
    win.flip()
    core.wait(feedbackDuration)
    
    # write to file
    gamble = combination[0]
    dataFile.write('%i,%i,%i, %.3f\n' %(gamble[0], gamble[1], thisResp, reactionTime))
    
    # fixation
    displayFixation(fixationDuration)
    
    numTrials += 1
    
endMessage = visual.TextStim(win, pos=[0, 0], text="The experiment has come to an end. Thanks for participating!", height=1.5, wrapWidth=wrapWidth)
endMessage.draw()
win.flip()
core.wait(5)
