import numpy as np
import matplotlib.pyplot as plt

import librosa as lb
import os, json

import pandas as pnd
from sklearn.utils import shuffle
from sklearn.preprocessing import scale
import seaborn as sn
import math as mt

from collections import Counter

import os
sr = 22050
path = os.path.dirname(os.path.realpath(__file__))

currentFile = open(path+"\maestro-v3.0.0.csv", encoding="utf8")
maestro = pnd.read_csv(currentFile)

def composer(maestro):
    '''
    Renvoie un tableau avec tous les compositeurs du fichier csv
    '''
    composer_b = pnd.unique(maestro['canonical_composer'])
    return composer_b


def tri_composer(composer_b):
    '''
    Renvoie un tableau contenant les compositeurs simples
    Renvoie un tableau contenant les compositeurs doubles
    '''
    composer = [x for x in composer_b if '/' not in x]
    composer_d = [x for x in composer_b if '/' in x]
    return composer, composer_d


def duree(composer):
    '''
    Renvoie un tableau contenant la durée totale des morceaux pour chaque compositeur
    donné en paramètre
    '''
    duration=[]
    for c in composer:
        count =0
        for index, current in maestro.iterrows():
            if current['canonical_composer']==c:
                count = count + current['duration']
        duration.append(count)
    return duration
    

def affichage(composer, duration):
    '''
    Affiche un graphe avec la durée totale des morceaux de chaque compositeur
    '''
    plt.figure()
    plt.bar(composer, duration)
    plt.title("durée totale des morceaux")
    plt.xticks(rotation=90)
    plt.xlabel("compositeur")
    plt.ylabel("durée")
    plt.show()


def list_compo(path, threshold):
    '''
    Liste des compositeurs remplissant les critères de sélection

    Parameters
    ----------
    path : string
        Chemin du fichier csv
    threshold : float
        Seuil limite à partir duquel est sélectionné le compositeur

    Returns
    -------
    composer : array
        Liste des compositeurs

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
    Création du tableau pandas avec les morceaux des compositeurs correspondant aux critères de sélection

    Parameters
    ----------
    path : string
        Chemin du fichier csv
    threshold : float
        Seuil limite à partir duquel est sélectionné le compositeur

    Returns
    -------
    new_tab : pnd array
        Tableau pandas contenant les morceaux, leur compositeur et leur durée

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

tab_f = new_dataset(path+"\maestro-v3.0.0.csv", 12000)
print(pnd.unique(tab_f['canonical_composer']))
print(tab_f['duration'].min())


def resampling(datas, cut=30.0):
    '''
    rééchantillonage des morceaux et découpe en blocs de mêmes tailles

    Parameters
    ----------
    datas : pnd array
        tableau en sortie de new_dataset
    sr : int, optional
        La fréquence d'échantillonage voulue. The default is 22050.
    cut : float
        Temps de coupe en seconde. The default is 30.0

    Returns
    -------
    X : numpy ndarray (2D)
        tableau des données temporelles resamplées et reformatées
    Y : numpy array[string]
        labels
    '''
    X = np.array()
    Y = np.array()
    for index, current in datas.iterrows():
        count = current['duration']
        i=0.0
        while i+cut < count :
            t, s = lb.load(current['audio_filename'],sr = sr, offset=i,duration=cut)
            X.append(t)
            Y.append(current['canonical_composer'])
            i = i+cut
    return X, Y
    

def audio_preprocessing(X, nb_mfcc, mfcc_resample=1):
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
    mfcc = np.ndarray(X.len)
    for x in X:
        cepstrum = lb.feature.mfcc(y=x,n_mfcc=nb_mfcc)[:1]
        if mfcc_resample != 1:
            cepstrum = np.transpose(cepstrum)
            cepstrum = np.transpose(cepstrum[::mfcc_resample])
        mfcc[x] = scale(cepstrum, axis=1)
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
    
