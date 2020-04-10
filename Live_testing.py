import os
import pyaudio
import wave
import soundfile
import numpy as np
import librosa
import pandas as pd
import keras
from sklearn.preprocessing import LabelEncoder
from keras.models import model_from_json
import pyaudio
import wave
import os
import threading
import time
import tkinter.messagebox
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from ttkthemes import themed_tk as tk
from mutagen.mp3 import MP3
from pygame import mixer

from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

def extract_feature(file_name, **kwargs):

    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        if X.ndim >= 2:
            X = np.mean(X, 1)
        sample_rate = sound_file.samplerate
        result = np.array([])
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
            result = np.hstack((result, tonnetz))
    return result

################################################recording audio##########################################################
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "/home/ria/ML project/output.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []


for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()


feature_test=extract_feature(WAVE_OUTPUT_FILENAME, mfcc=True, chroma=True, mel=True)
livedf2 = feature_test
livedf2= pd.DataFrame(data=livedf2)
livedf2 = livedf2.stack().to_frame().T

input_test= np.expand_dims(livedf2, axis=2)
y = ["neutral",
    "calm",
    "happy",
    "sad",
    "angry"]
lb = LabelEncoder()
y_final = np_utils.to_categorical(lb.fit_transform(y))

json_file = open('/home/ria/ML project/Saved model/model2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/home/ria/ML project/Saved model/Emotion_Voice_Detection_Model2.h5")
print("Loaded model from disk")

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


preds = loaded_model.predict(input_test,
                         batch_size=32,
                         verbose=1)

preds1=preds.argmax(axis=1)

abc = preds1.astype(int).flatten()
predictions = (lb.inverse_transform((abc)))

preddf = pd.DataFrame({'predictedvalues': predictions})
print("Looks like you are :")
print(preddf)

s = preddf['predictedvalues']

import csv
import pandas as pd
import numpy as np

# csv file name
filename = "/home/ria/ML project/songs.csv"

# initializing the titles and rows list
fields = []
rows = []
songs = []
songs1=[]
if s[0] == 'happy':
    emotion ='Happy'
elif s[0] == 'sad':
    emotion ='Sad'
elif s[0] == 'neutral':
    emotion ='Neutral'
elif s[0] == 'calm':
    emotion ='Calm'
elif s[0] == 'angry':
    emotion ='Angry'
else:
    print("no emotion")
final=[]
ind_pos = [1,2,4]

# reading csv file
with open(filename, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)

    # extracting each data row one by one
    for row in csvreader:
        rows.append(row)

    for i in range(csvreader.line_num-1):
        words = rows[i][3].split(",")
        for word in words:
            if word.strip() == emotion:
                songs.append(rows[i])
        # get total number of rows
    pd.set_option('display.max_rows',500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    for j in range(len(songs)):
        final.append([songs[j][i] for i in ind_pos])
    pd_dataframe = pd.DataFrame(final, columns=['Artist', 'Song', 'Link'])
    print(" The songs that match your mood are : ")
    print(pd_dataframe)

####################################if you are angry or sad, the below songs will be suggested to make your mood better###################################

    emo_subset = ['Angry','Sad']
    if emotion in emo_subset:
        for i in range(csvreader.line_num-1):
            words = rows[i][3].split(",")
            for word in words:
                if word.strip() == 'Happy':
                    songs.append(rows[i])
            # get total number of rows
        pd.set_option('display.max_rows',500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        for j in range(len(songs1)):
            final.append([songs1[j][i] for i in ind_pos])
        pd_dataframe1 = pd.DataFrame(final, columns=['Artist', 'Song', 'Link'])
        print(" The songs that you should listen to : ")
        print(pd_dataframe1)
    else:
        print("The songs you should listen to are listed above")
