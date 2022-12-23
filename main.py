#rEaL pRoGrAmMiNg LaNgUaGeS uSe SeMiCoLoNs
print("starting...")
import function

dataFolder = "B:/documents/DeepShift/dataGather/trainingData" # a folder full of audio or image data. Each one must be in the index file.
labelsIndex = "B:/documents/DeepShift/dataGather/trainingData/index.json" # example: {"file0.wav": 9, "file1.wav": 2, ...} for labels (the label must be a number), OR {"infile0.wav": "outfile0.wav", "infile1.wav": "outfile1.wav", ...} for generation
epochs = 100 # for a labeling network, these usually go about 1-2 per second
inputType = 'wav' # supports .wav and all image formats. If you want to use mp3, convert them to a wav first.
hiddenLayers=[128, 64, 64]
sameLength=True # also can be true if they're similar but not exactly the same in length
generate=False # If you aren't making a network that labels or identifies data, and are making one that generates new data based on old data (audio cleanup, for example), set this to True.
possibleLabels=40 # Number of output nodes/number of possible labels. If generate=True, set this to 0.
device='GPU:1' # The device uesd for training. Runs best on a GPU. If you don't have one, use GPU:0 to use your default CPU. To use a GPU or a different CPU, right click on the windows icon, click task manager, click more details (if that button exists, if not don't), then click preformance. On the left-hand side, your usable GPUs and their IDs will be on the bottom of the list, named something like GPU 1. Set this variable to "GPU:1" or whatever ID you want to use.
outputFolder = 'B:/documents/DeepShift/model' # this is where the model will be saved so you can use it later.

function.train(dataFolder=dataFolder, outputFolder=outputFolder, labelsIndex=labelsIndex, inputType=inputType, hiddenLayers=hiddenLayers, sameLength=sameLength, generate=generate, possibleLabels=possibleLabels, device=device)
