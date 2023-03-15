import pygame.midi
from time import sleep
import random
import sounddevice as sd
from scipy.io.wavfile import write
import json
from tkinter import filedialog
import os

print("starting")

#get the path of the folder the data will be stored in
global dataPath
dataPath = str(filedialog.askdirectory(title="What folder will the data be saved to?"))

def hasFile(path, filename):
    list = os.listdir(path)
    return (filename in list)

#if there's no index.json file, create one
#index.json will store the names of all the sound files and the notes they contain
if not hasFile(dataPath, "index.json"):
    with open(dataPath + "/index.json", "x") as outfile:
        outfile.write('{}')

#index file dictates which files contain which notes
index = open(dataPath + "/index.json")

#allData will store all the information ever collected
global allData
allData = json.load(index)

#the ID of the instrument played
instrument = 5

#number of notes in each recording
dataLen = 1
extra = 1
recDuration = dataLen + extra

#bitrate
fs = 44100

#pygame/midi stuff
pygame.init()
pygame.midi.init()
defaultOut = pygame.midi.get_default_output_id()
player = pygame.midi.Output(defaultOut)


#play a sound with pygame midi
def playSound(note):
    player.set_instrument(instrument)
    player.note_on(note, 127)
    sleep(.5)
    player.note_off(note, 127)


#note range example: [45, 72] or A2 - C5
#may be changable in future updates

while True:

    #thisData holds the notes of the current recording and is emptied each cycle
    thisData = []
    theseNotes = []

    #pick random notes to play and add them to thisData
    for i in range(0, dataLen):
        min = 45
        max = 75
        note = random.randrange(min, max)
        print(note)
        thisData.append(note)
        playSound(note)
        sleep(0.5)
        theseNotes.append(note)

    print("sing!")

    #record using soundDevice
    recording = sd.rec(int(recDuration * fs), samplerate=fs, channels=2)
    sd.wait()

    name = dataPath + '/note' + str(len(allData)+1) + ".wav"


    print("press c to continue, press s to save and quit, press q to quit, press d to delete the most recent recording,")
    nextInput = input()

    if nextInput == "q":
        print("quitting...")
        exit()

    if (nextInput != "d") and (nextInput != "x"):
        #save the recording and data
        write(name, fs, recording)
        recordingName = "note" + str(len(allData)+1) + ".wav"
        allData.update({recordingName: thisData[0]})
        print(allData)
    if nextInput != "c":
        if nextInput == "s":
            #save the new data
            print("saving...")
            with open(dataPath + "/index.json", "w") as f:
                json.dump(allData, f)
            exit()
        elif nextInput == "d":
            print("deleting...")