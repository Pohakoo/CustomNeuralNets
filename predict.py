import numpy as np
import soundfile as sf
import tensorflow as tf
from tkinter import filedialog

noteNum = 10

# Load the trained model from the .h5 file
modelFile = str(filedialog.askopenfilename(title="Please select your trained_model.h5"))
model = tf.keras.models.load_model(modelFile)

# Load the audio data from the .WAV file
inputFile = str(filedialog.askopenfilename(title="Please select your input file for the prediction"))
audio, sample_rate = sf.read(inputFile)

# Convert the audio data to a NumPy array
audio = np.array(audio)

# Reshape the audio data to match the model's input shape
audio = audio.reshape(1, -1)

# Use the model to predict the category of the audio
prediction = model.predict(audio)

# Print the prediction
list = np.ndarray.tolist(prediction[0])
most_probable = list.index(max(list))
print("The note is most likely note " + str(most_probable) + ". The confidence is " + str(round(max(list))) + ".")
print("output nodes: " + str(list))
