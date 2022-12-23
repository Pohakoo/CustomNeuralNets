#rEaL pRoGrAmMiNg LaNgUaGeS uSe SeMiCoLoNs
print("starting...")
import function
import json
from tkinter import filedialog

try:
    with open('config.json') as f:
        config = json.load(f)
        print(config)


    print("getting training data folder...")
    dataFolder = str(filedialog.askdirectory(title="What folder is the training data in?")) # a folder full of audio or image data. Each one must be in the index file.
    print("getting index json file...")
    labelsIndex = str(filedialog.askopenfilename(title="index json file")) # example: {"file0.wav": 9, "file1.wav": 2, ...} for labels (the label must be a number), OR {"infile0.wav": "outfile0.wav", "infile1.wav": "outfile1.wav", ...} for generation
    print("getting output folder...")
    outputFolder = str(filedialog.askdirectory(title="What folder should the model be saved to?")) # this is where the model will be saved so you can use it later.

    epochs = config['epochs'] # for a small network, these usually go about 1-2 per second
    inputType = config['inputType'] # supports .wav and all image formats. If you want to use mp3, convert them to a wav first.
    hiddenLayers = config['hiddenLayers'] # the hidden layers of the network in between the input and output layers. You should be able to figure out what these values should be if you have a basic conceptual knowledge of neural network structure.
    sameLength = config['sameLength'] # also can be true if they're similar but not exactly the same in length
    generate = config['generate'] # If you aren't making a network that labels or identifies data, and are making one that generates new data based on old data (audio cleanup, for example), set this to True.
    possibleLabels = config['possibleLabels'] # Number of output nodes/number of possible labels. If generate=True, set this to 0.
    device = config['device'] # The device uesd for training. Runs best on a GPU. If you don't have one, use GPU:0 to use your default CPU. To use a GPU or a different CPU, right click on the windows icon, click task manager, click more details (if that button exists), then click preformance. On the left-hand side, your usable GPUs and their IDs will be on the bottom of the list, named something like GPU 1. Set this variable to "1" or whatever ID you want to use.
    optimizer = config['optimizer'] # advanced option. Doesn't really matter with simple networks, but you can choose between "adam", "SGD" (Stochastic gradient descent), "RMSprop" (Root Mean Squared Propagation), "Adagrad" (Adaptive Gradient Algorithm), and "Adadelta".

    if generate.lower() == 'true':
        generate = True
    elif generate.lower() == 'false':
        generate = False
    else:
        print("config formatting error")
        exit()
    if sameLength.lower() == 'true':
        sameLength = True
    elif sameLength.lower() == 'false':
        sameLength = False
    else:
        print("config formatting error")
        exit()
except:
    print("error with config file!")
    exit()

# for more specific/clearer info, refer to the README.md file in github.
function.train(epochs=epochs, dataFolder=dataFolder, outputFolder=outputFolder, labelsIndex=labelsIndex, inputType=inputType, hiddenLayers=hiddenLayers, sameLength=sameLength, generate=generate, possibleLabels=possibleLabels, device=device, optimizer=optimizer)
