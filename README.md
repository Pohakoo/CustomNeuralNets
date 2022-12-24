<h2 align="center">
This project makes it easy to train a custom neural network using your own training data.<br>
</h4>
<p align="center">
If you have a basic conceptial knowledge of how neural networks work, and have some training data and labels, this project is great for training a network without having to write the code yourself which is super hard.
</p>

<br><br>
**If you aren't familliar with neural networks,** I'd like to strongly reccommend watching these three videos in order to gain a working knowledge of their concept and structure:<br>
CGP Grey's HOW MACHINES LEARN: https://youtu.be/R9OHn5ZF4Uo<br>
3Blue1Brown's BUT WHAT IS A NEURAL NETWORK? Chapters 1 and 2: https://youtu.be/aircAruvnKk https://youtu.be/IHZwWFHWa-w (chapter 2 is optional)

If you need help, feel free to join my [Discord server](https://dsc.gg/Pohakoo), create a ticket, and ping me with your question (I'm the owner, @Pohakoo#0001).

**What you need:** You need a dataset (a bunch of images or audio clips in a folder) and labels. This script supports neural netwoks of two types, **identification and generation.** The identification type trains a network to label, identify, or categorize an image or audio clip. The generation type trains a network to take an image or audio clip as input, and then output another image or audio clip. This type is still in beta. If you want some sample datasets, I have some fun ones that can be downloaded [here](https://drive.google.com/file/d/15K4D35rS53anJg68fpmE48BaByK8ojI7/view?usp=share_link).

**If you want to train an identification network,** you need about 40+ peices of data. If the data is images, they all should be the same/similar size. If the data is audio, they should all be the same/similar length. If the data is very different in size or length, there's an option that supports this, but it's not reccommended. You also need an index.json file that labels each peice of data. It should be formatted like this: `{"file0.wav" : 9, "file1.wav" : 2...}`. The files do not have to have any specific name, but they each need to have a label. For example, if your neural network recognizes song genre, the index might look like this: `{"rock1.wav" : 3, "jazz1.wav" : 1, "pop1.wav" : 2, "rock2.wav" : 3...}` in this case, jazz is 1, pop is 2, and rock is 3. The labels have to be integers if you're doing this type of network.

**If you want to train a generation network,** you need about 150+ peices of data. If the data is images, they all should be the same/similar size. If the data is audio, they should all be the same/similar length. If the data is very different in size or length, there's an option that supports this, but it's not reccommended. You also need an index.json file that labels each peice of data. It should be formatted like this: `{"input0.png" : "output0.png", "input1.png" : "output1.png", "input2.png" : "output2.png"...}` The network will be trained to convert the inputs to the outputs. This network might do something like noise reduction in an image or video, meaning the inputs would be noisy images and the outputs would be those same images but without noise. Right now, this script only works if you're using the same input and output datatype, but that will be fixed soon.

In the future, I'll create some scripts that make it easy to create these datasets and also automatically generate index.json files.

**How to train:** start by downloading all of the files in the repository. In your interpreter, install the nessecary packages by running 
`pip install -r requirements.txt`. Then, open the config.json file. 

The following is a description of what each setting in the config does and what you should set them to. However, if you have a specific set of training data and you know how you want to use it, you can use this [scratch project](https://scratch.mit.edu/projects/780826281/) I made (because why not lol) that's basically a Buzzfeed quiz and it basically gives you suggestions on what the variables should be based on your answers to a few questions.<br>

**epochs:** the number iterations your network goes through. You need to have more than 50 of these. It depends on what you're doing, but for an identification network, 100-500 is utually sufficient. You'll need more for a generation network though, and it will probably take longer per iteration.<br>
**inputType:** the file extension of your training data. For audio, WAV is best. For images, PNG is best. I also support other image formats. I will soon update this so text is supported, among other things.<br>
**hiddenLayers:** the hidden layers of the network in between the input and output layers. You should be able to figure out what these values should be, if not, use the scratch project.<br>
**sameLength:** I wouldn't reccommend ever setting this to False. If all your input data is the same or similar lengths, set this to True. If it varies wildly, set it to False. Results are not gurenteed.<br>
**generate:** If you're using an identification network, set this to False. If you're using a generation network, set it to True.<br>
**possibleLabels:** the number of different labels that could be assigned to a peice of data. The label numbers must range from zero to this number. For example, if an audio clip could be labeled rock, pop, or jazz, this value would be 2. (0, 1, 2)<br>
**device:** The device uesd for training. These run best on a GPU. If you don't have one, use the default option. To use a GPU or a different CPU, right click on the windows icon, click task manager, click more details (if that button exists), then click preformance. On the left-hand side, your usable GPUs and their IDs will be on the bottom of the list, named something like GPU 1. Set this variable to "GPU:1" or whatever ID you want to use.<br>
**Optimizer:** advanced option. Doesn't really matter with simple networks, but you can choose between "adam", "sgd" (Stochastic gradient descent), "rmsprop" (Root Mean Squared Propagation), "adagrad" (Adaptive Gradient Algorithm), and "adadelta". If you want to learn more, Google it.

Once all these variables are set, run main.py. It will ask you to tell it where your input data is, where your index.json file is, and where you want your network to be saved. Then it will run. It will keep you updated in the console. Once it's finished, you will find a trained_network.h5 file in your output destination. Time to use your network. 

**How to predict:** run the predict.py script. It will ask you to select your trained_model.h5 file and the input data file. It will then reutrn a prediction and a confidence value, along with the confidence values for all the other output nodes.
<br>
<h3>
Feature support
</h3>
<h4>
✅ = Working and tested <br>🚧 =  in beta/iffy <br>❌ = in development
</h4>

* ✅ train a NN to sort audio<br>
* ✅ train a NN to sort images<br>
* ❌ train a NN to sort text<br>
* ✅ Customizable hidden layers<br>
* ✅ Customize the number of epochs<br>
* ✅ Customize optimizer<br>
* 🚧 Contains example training data > [download here](https://drive.google.com/file/d/15K4D35rS53anJg68fpmE48BaByK8ojI7/view?usp=share_link)<br>
* 🚧 Train a NN that inputs audio and outputs audio<br>
* 🚧 Train a NN that inputs images and outputs images<br>
* 🚧 Train a NN outputs text<br>
* ✅ Includes a script that predicts input using trained NN<br>
* ❌ Includes a script that allows you to label images in a folder<br>
* ❌ Includes a script that allows you to label audio in a folder<br>
