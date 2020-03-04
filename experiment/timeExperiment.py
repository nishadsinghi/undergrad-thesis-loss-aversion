from psychopy import core, visual, gui, data, event  # import some libraries from PsychoPy
import itertools
import random
from psychopy.tools.filetools import fromFile, toFile
import time

try:  # try to get a previous parameters file
    expInfo = fromFile('lastParams.pickle')
except:  # if not there then use a default set
    expInfo = {'observer':'default'}
expInfo['dateStr'] = data.getDateStr()  # add the current time

# present a dialogue to change params
dlg = gui.DlgFromDict(expInfo, title='Time Gambles', fixed=['dateStr'])
if dlg.OK:
    toFile('lastParams.pickle', expInfo)  # save params to file for next time
else:
    core.quit()  # the user hit cancel so exit

# make a text file to save data
fileName = expInfo['observer'] + expInfo['dateStr']
dataFile = open(fileName+'.csv', 'w')  # a simple text file with 'comma-separated-values'
dataFile.write('increaseValue,decreaseValue,response, RT\n')

############################################################################################

# create a window
win = visual.Window([800,600], monitor="testMonitor", units="deg", fullscr=True)

# randomly decide which key corresponds to accept 
responseCorrespondingToUp = random.randrange(2)
responseCorrespondingToDown = (responseCorrespondingToUp + 1) % 2

if responseCorrespondingToUp == 1:
    instructionMessage = "Press UP to ACCEPT. Press DOWN to REJECT. "
elif responseCorrespondingToUp == 0:
    instructionMessage = "Press DOWN to ACCEPT. Press UP to REJECT. "

wrapWidth = 35
# print text for a gamble on screen
def printGambleValues(combination):
    gamble = combination[0]
    orientation = combination[1]
    
    instructions = visual.TextStim(win, pos=[0, 10], text=instructionMessage, height=0.8, wrapWidth=wrapWidth)
    
    if orientation == 'increaseLeft':
        leftValue = 'M ' + str(gamble[0])
        rightValue = 'L ' + str(gamble[1])
    elif orientation == 'increaseRight':
        leftValue = 'L ' + str(gamble[1])
        rightValue = 'M ' + str(gamble[0])
        
    height = 4.5
    message1 = visual.TextStim(win, pos=[-10, 0],text=leftValue, height=height)
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
allDecreaseValues = [1, 5, 10, 15, 20, 25]
allCombinations = list(itertools.product(allIncreaseValues, allDecreaseValues))
allCombinationsAlternatePresentLocations = list(itertools.product(allCombinations, ('increaseLeft', 'increaseRight')))
random.shuffle(allCombinationsAlternatePresentLocations)

# Welcome message
welcomeMessage = visual.TextStim(win, pos=[0, 0], text="Welcome! You will now be making a sequence of decisions. " + instructionMessage + "Press any key when you are ready to begin!", height=1.5, wrapWidth=wrapWidth)
welcomeMessage.draw()
win.flip()

event.waitKeys()

# present gambles
fixationDuration = 1
interBlockBreakDuration = 20

numBlocks = 2
numGambles = len(allCombinationsAlternatePresentLocations)
blockLength = numGambles // numBlocks
print("Length of block = ", blockLength)
numTrials = 0
for combination in allCombinationsAlternatePresentLocations:
    if numTrials % blockLength == 0 and numTrials != 0:
        print("Displaying block at numTrial", numTrials)
        m = visual.TextStim(text="Break", win=win)
        m.draw()
        win.flip()
        core.wait(interBlockBreakDuration)
#        displayFixation(interBlockBreakDuration)    
    
    # display stimulus
    printGambleValues(combination)
    win.flip()
    startTime = time.time()
    
    # get response
    thisResp=None
    while thisResp==None:
        allKeys=event.waitKeys()
        for thisKey in allKeys:
            if thisKey=='up':
                thisResp = responseCorrespondingToUp  # accept
                endTime = time.time()
            elif thisKey=='down':
                thisResp = responseCorrespondingToDown  # reject
                endTime = time.time()
            elif thisKey in ['q', 'escape']:                core.quit()
        event.clearEvents()  # clear events
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

