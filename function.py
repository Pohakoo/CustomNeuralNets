import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import json
import soundfile as sf
import imageio

#if the Neural Network takes in audio/images and outputs probability of each label, use generate=False and the index.json file is formatted like this: {"file0.ex": 3, "file2.ex": 1, ...}
#If the Neural Network takes in audio/images and outputs audio/images, use generate=True and the index.json is formatted like this: {"infile0.ex": "outfile0.ex", "infile1.ex": "outfile1.ex"}
#It's fine if the file names are different

def train(dataFolder, outputFolder, labelsIndex, epochs=0, inputType='wav', hiddenLayers=[], sameLength=True, generate=False, possibleLabels=0, device='GPU:0', npyoutputfolder=None, npyfilesfolder=None, optimizer="adam"):

    if hiddenLayers == []:
        if not generate:
            hiddenLayers = [128, 64, 64]
        else:
            hiddenLayers = [2048, 2048, 2048]
    if epochs == 0:
        if not generate:
            epochs = 100
        else:
            epochs = 250

    def wav_to_npy(dataFolder, labelsIndex, generate):
        # Load the index data from the json file
        with open(labelsIndex, 'r') as f:
            index_data = json.load(f)

        # Initialize empty lists
        data = []
        labels = []

        # Iterate over the index data
        for item in index_data:
            # Load the audio data from the .WAV file
            print(dataFolder + '/' + item)
            audio, sample_rate = sf.read(dataFolder + '/' + item)

            # Convert the audio data to a NumPy array
            audio = np.reshape(audio, (audio.shape[0]*2))

            # Add the audio data and label to the lists
            data.append(audio)
            if generate == False:
                labels.append(index_data[item])
            else:
                audio, sample_rate = sf.read(dataFolder + '/' + index_data[item])
                audio = np.reshape(audio, (audio.shape[0] * 2))
                labels.append(audio)

        # Convert the lists to NumPy arrays
        data = np.array(data)
        labels = np.array(labels)

        return data, labels

    def img_to_npy(dataFolder, labelsIndex, generate):
        # Load the index data from the json file
        with open(labelsIndex, 'r') as f:
            index_data = json.load(f)

        # Initialize empty lists to store the data and labels
        data = []
        labels = []

        # Iterate over the index data
        for item in index_data:
            # Load the image data from the .WAV file
            print(dataFolder + '/' + item)
            image = imageio.imread(dataFolder + '/' + item)

            # Convert the image data to a NumPy array
            image = image.flatten()

            # Add the image data and label to the lists
            data.append(image)
            if not generate:
                labels.append(index_data[item])
            else:
                audio, sample_rate = sf.read(dataFolder + '/' + index_data[item])
                audio = np.reshape(audio, (audio.shape[0] * 2))
                labels.append(audio)

        # Convert the lists to NumPy arrays
        data = np.array(data)
        labels = np.array(labels)

        return data, labels

    def data_to_npy(inputType, dataFolder, labelsIndex, generate):
        if inputType.lower() == "wav":
            data, labels = wav_to_npy(dataFolder, labelsIndex, generate)
        if inputType.lower() == "png" or inputType.lower() == "jpg" or inputType.lower() == "jpeg" or inputType.lower() == "bmp" or inputType.lower() == "tiff" or inputType.lower() == "gif" or inputType.lower() == "pdf" or inputType.lower() == "svg":
            data, labels = img_to_npy(dataFolder, labelsIndex, generate)

        return data, labels

    if npyfilesfolder == None:
        data, labels = data_to_npy(inputType=inputType, dataFolder=dataFolder, labelsIndex=labelsIndex, generate=generate)
    else:
        data = np.load(npyfilesfolder + "/data.npy")
        labels = np.load(npyfilesfolder + "/labels.npy")

    if npyoutputfolder != None:
        np.save(npyoutputfolder + '/data.npy', data)
        np.save(npyoutputfolder + '/labels.npy', labels)

    def pad_or_truncate_to_average(audio_data, sameLength):
        # Calculate the target length of the audio clips
        if sameLength:
            target_length = np.mean([data.shape[0] for data in audio_data])
        else:
            target_length = np.max([data.shape[0] for data in audio_data])

        # Round the target length to the nearest integer
        target_length = int(round(target_length))

        # Get the shape of the audio data
        shape = audio_data.shape

        # Create an empty array with the same shape as the audio data
        padded_data = np.zeros(shape)

        # Iterate over the audio data
        for i, data in enumerate(audio_data):
            # Truncate the data if it is too long
            if data.shape[0] > target_length:
                padded_data[i] = data[:target_length]
            # Pad the data if it is too short
            elif data.shape[0] < target_length:
                padded_data[i, :data.shape[0]] = data
            # Leave the data unchanged if it is the right length
            else:
                padded_data[i] = data

        return padded_data, target_length

    # Preprocess the audio data
    preprocessed_data, inputLength = pad_or_truncate_to_average(data, sameLength)
    if possibleLabels == 0:
        if not generate:
            possibleLabels = 10
        else:
            possibleLabels = inputLength

    # Split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(preprocessed_data, labels)

    # Define the model architecture
    with tf.device('/device:GPU:'+str(device)):
        model = tf.keras.Sequential()

    for i in range(0, len(hiddenLayers)-1):
        model.add(tf.keras.layers.Dense(hiddenLayers[i], activation='relu'))
    model.add(tf.keras.layers.Dense(possibleLabels, activation='softmax'))

    # Compile the model
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model and store the training history
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))

    # Extract the accuracy and loss values from the history object
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Plot the accuracy and loss values
    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    # Save the trained model to a file
    model.save(outputFolder + '/trained_model.h5')
