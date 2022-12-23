#rEaL pRoGrAmMiNg LaNgUaGeS uSe SeMiCoLoNs
print("starting...")
import function

dataFolder = "B:/documents/DeepShift/dataGather/trainingData"
labelsIndex = "B:/documents/DeepShift/dataGather/trainingData/index.json"
epochs = 100
inputType = 'wav'
hiddenLayers=[128, 64, 64]
sameLength=True #also can be true if they're similar but not exactly the same in length
generate=False
possibleLabels=40
device='GPU:1'
outputFolder = 'B:/documents/DeepShift/model'

function.train(dataFolder=dataFolder, outputFolder=outputFolder, labelsIndex=labelsIndex, inputType=inputType, hiddenLayers=hiddenLayers, sameLength=sameLength, generate=generate, possibleLabels=possibleLabels, device=device)
