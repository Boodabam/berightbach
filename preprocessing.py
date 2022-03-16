import numpy as np
import matplotlib.pyplot as plt

import librosa as lb
import os, json

import pandas as pnd
from sklearn.utils import shuffle
import seaborn as sn
import math as mt

from collections import Counter

import os

path = os.path.dirname(os.path.realpath(__file__))

currentFile = open(path+"\maestro-v3.0.0.csv", encoding="utf8")
maestro = pnd.read_csv(currentFile)

composer_b = pnd.unique(maestro['canonical_composer'])
#print(composer_b)
#print(len(composer_b))

composer = [x for x in composer_b if '/' not in x]
composer_d = [x for x in composer_b if '/' in x]
#print(composer)
#print(len(composer))
#print(composer_d)
#print(len(composer_d))

duration=[]
for c in composer:
    count =0
    for index, current in maestro.iterrows():
        if current['canonical_composer']==c:
            count = count + current['duration']
    duration.append(count)
    
#print(duration)

plt.figure()
plt.bar(composer, duration)
plt.title("durée totale des morceaux")
plt.xticks(rotation=90)
plt.xlabel("compositeur")
plt.ylabel("durée")
plt.show()

index = composer.index('Modest Mussorgsky')
#print(index)
#print(composer[index], duration[index])

composer_f=[]
for i in range (len(composer)):
    if duration[i]>duration[index]:
        composer_f.append(composer[i])

#print(composer_f)

def list_compo(path, threshold):
    '''
    

    Parameters
    ----------
    path : string
        Chemin du fichier csv
    threshold : float
        Seuil limite à partir duquel est sélectionné le compositeur

    Returns
    -------
    composer : array
        Liste des compositeurs qui remplissent les critères

    '''
    currentFile = open(path, encoding="utf8")
    File = pnd.read_csv(currentFile)
    composer = pnd.unique(File['canonical_composer'])
    composer_f = []
    for c in composer:
        count =0
        for index, current in File.iterrows():
            if current['canonical_composer']==c:
                count = count + current['duration']
        if count > threshold:
            composer_f.append(c)
    return composer_f


def new_dataset(path, threshold):
    '''
    

    Parameters
    ----------
    path : string
        Chemin du fichier csv
    threshold : float
        Seuil limite à partir duquel est sélectionné le compositeur

    Returns
    -------
    new_tab : pnd array
        Tableau pandas contenant uniquement les compositeurs correspondants aux critères
        associés aux morceaux et leur durée

    '''
    currentFile = open(path, encoding="utf8")
    File = pnd.read_csv(currentFile)
    tab = []
    composer = list_compo(path, threshold)
    for index, current in File.iterrows():
        if '/' not in current['canonical_composer']:
            if current['canonical_composer'] in composer:
                tab.append((current['audio_filename'],current['canonical_composer'],current['duration']))
    new_tab = pnd.DataFrame(data = tab, columns=('audio_filename','canonical_composer','duration'))
    return new_tab

tab_f = new_dataset(path+"\maestro-v3.0.0.csv", duration[index])
print(pnd.unique(tab_f['canonical_composer']))
print(tab_f['duration'].min())


def resampling(datas, sr=22050):
    '''
    rééchantillonage des morceaux

    Parameters
    ----------
    datas : pnd array
        tableau en sortie de new_dataset
    sr : int, optional
        La fréquence d'échantillonage voulue. The default is 22050.

    Returns
    -------
    X : numpy ndarray (3D)
        tableau des données temporelles resamplées
    Y : numpy array[string]
        labels
    length : numpy array[int]
        longueur des donées en échantillons
    '''
    return X, Y, length

def sample_cut(X, Y, length, cut):
    '''
    Découpe des morceaux en blocs de mêmes tailles

    Parameters
    ----------
    X : numpy ndarray (3D)
        Les données temporelles des morceaux entiers
    Y : numpy array[string]
        labels
    length : numpy array[int]
        nombre d'échantillons par morceau
    cut : int
        échantillon de coupe

    Returns
    -------
    X_blocs : numpy ndarray (3D)
        données reformatées
    Y_blocs : numpy array[int]
        labels reformatés

    '''
    return X_blocs, Y
    

def audio_preprocessing(X, nb_mfcc, mfcc_sample_rate):
    '''
    Transformation mfc et rééchantillonage
    
    Parameters
    ----------
    X : numpy ndarray (3D)
        données temporelles
    nb_mfcc : int
        nombre de mfcc
    mfcc_sample_rate : int
        pas de rééchantillonage après transformation mfc

    Returns
    -------
    mfcc : numpy ndarray (3D)
        mfcc rééchantillonnés

    '''
    return mfcc

def write_json(X, Y, json_name):
    '''
    Effectue le mapping de labels et écrit les données dans un json

    Parameters
    ----------
    X : numpy ndarray (3D)
        mfcc resamplées
    Y : numpy array[string]
        labels
    json_name : string
        nom du fichier json

    Returns
    -------
    None.

    '''
    