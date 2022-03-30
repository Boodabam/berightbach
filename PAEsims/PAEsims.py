# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa as lb
import librosa.display
import os
from IPython.display import Audio
from sklearn.preprocessing import scale
import timeit

path = os.path.dirname(os.path.realpath(__file__))

start = timeit.default_timer()

signal, sr = lb.load(path+"\\test2.wav", sr=22050, offset=30,duration=10)
nb_points = signal.size

mel = librosa.feature.melspectrogram(signal)
spectrum = np.abs(lb.stft(signal))
mfccs = lb.feature.mfcc(signal, sr = sr, n_mfcc =127)[1:]

stop = timeit.default_timer()

print('Time: ', stop - start)  


Audio(signal, rate=sr)

fig_signal = plt.figure(figsize=(120,150))
ax = fig_signal.add_subplot(5,1,1)
mesh_time = np.linspace(0,nb_points/sr,nb_points)
ax.plot(mesh_time, signal)
ax.set_xlim(0,nb_points/sr)

ax = fig_signal.add_subplot(5,1,2)
img = librosa.display.specshow(lb.amplitude_to_db(spectrum,ref=np.max),y_axis='log', x_axis='time', ax=ax)
ax.set_title('Power spectrogram')

ax = fig_signal.add_subplot(5,1,3)
img = librosa.display.specshow(lb.amplitude_to_db(mel,ref=np.max),y_axis='log', x_axis='time', ax=ax)
ax.set_title('Power spectrogram')

fig_signal.colorbar(img, ax=ax, format="%+2.0f dB")
ax = fig_signal.add_subplot(5,1,4)
img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
fig_signal.colorbar(img, ax=ax)

pre_resample = np.transpose(mfccs)
mfccs = np.transpose(pre_resample[::1])
resampled = scale(mfccs, axis=1)

ax = fig_signal.add_subplot(5,1,5)
img = librosa.display.specshow(resampled, x_axis='time', ax=ax)
fig_signal.colorbar(img, ax=ax)
ax.legend()