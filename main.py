import numpy as np
import matplotlib.pyplot as plt
import sklearn.neural_network as sknn
import librosa as lb
import os

path = os.getcwd()+"\\minidataset\\"

"""
classe principale de préprocessing audio

"""
def preprocessiong(file_names, hi_cut=100, lo_cut=15000,  ):
    
    for file_name in file_names:
        loaded = lb.load(path+"\\"+file_name)
        signal = loaded[0]
        nb_data = signal.size
        sr = loaded[1] #sample rate
        length = nb_data/sr
        





