print("starting...")
import numpy as np
import soundfile as sf
import tensorflow as tf
import imageio
import json
from PIL import Image
from tkinter import filedialog

noteNum = 10
with open('config.json') as f:
    config = json.load(f)
# Load the trained model from the .h5 file
modelFile = str(filedialog.askopenfilename(title="Please select your trained_model.h5"))
model = tf.keras.models.load_model(modelFile)

# Load the audio data from the .WAV file
inputFile = str(filedialog.askopenfilename(title="Please select your input file for the prediction"))

def predict_audio_identification():
    audio, sample_rate = sf.read(inputFile)

    # Convert the audio data to a NumPy array
    audio = np.array(audio)

    # Reshape the audio data to match the model's input shape
    audio = audio.reshape(1, -1)

    # Use the model to predict the category of the audio
    return model.predict(audio)

def predict_image_identification():
    image = imageio.imread(inputFile)
    # Flatten the numpy array
    image = image.reshape(1, -1)
    # Use the model to predict the category of the image
    return model.predict(image)

def predict_image_generate():
    inimage = imageio.imread(inputFile)
    # Flatten the numpy array
    inimage = inimage.reshape(1, -1)

    outFolder = str(filedialog.askdirectory(title="Please select the folder for the output image to be saved to"))

    prediction = model.predict(inimage)

    image = Image.open(inputFile)

    width, height = image.size

    outImage = prediction.reshape((width, height))

    outImage = Image.fromarray(outImage)

    outImage.save(outFolder + "/result.png")

    return "Saved image to " + outFolder + "/result.png"

if config['generate'].lower() == 'false':
    if config['inputType'].lower() == "wav":
        prediction = predict_audio_identification()
    if config['inputType'].lower() == "png" or config['inputType'].lower() == "jpg" or config['inputType'].lower() == "jpeg" or config['inputType'].lower() == "bmp" or config['inputType'].lower() == "tiff" or config['inputType'].lower() == "gif" or config['inputType'].lower() == "pdf" or config['inputType'].lower() == "svg":
        prediction = predict_image_identification()

    # Print the prediction
    list = np.ndarray.tolist(prediction[0])
    most_probable = list.index(max(list))
    print("The note is most likely output is " + str(most_probable) + ". The confidence is " + str(round(max(list)) * 100) + "%.")

elif config['generate'].lower() == 'true':

    print(predict_image_generate())


