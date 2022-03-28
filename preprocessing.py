import numpy as np
import matplotlib.pyplot as plt

import librosa as lb
import os, json

import pandas as pnd
from sklearn.utils import shuffle
from sklearn.preprocessing import scale
import seaborn as sn
import math as mt


''' Déclaration des variables globales '''

sr = 22050
path = os.path.dirname(os.path.realpath(__file__))
path_csv = path + "\\maestro-v3.0.0\\maestro-v3.0.0.csv"
datas_csv = open(path_csv, encoding="utf8")
maestro = pnd.read_csv(datas_csv)


def composer(maestro):
    '''
    Renvoie un tableau avec tous les compositeurs du fichier csv
    
    Parameters
    ----------
    maestro : tableau pandas
        tableau contenant les données du fichier csv
    
    Returns
    -------
    composer_b : numpy array[string]
        tableau contenant tous les compositeurs du dataset
    '''
    composer_b = pnd.unique(maestro['canonical_composer'])
    return composer_b


def tri_composer(composer_b):
    '''
    Tri les compositeurs entre les morceaux ayant un unique compositeur et les morceaux contenant deux compositeurs
    
    Parameters
    ----------
    composer_b : numpy array[string]
        Liste des compositeurs à trier
    
    Returns
    -------
    composer : numpy array[string]
        tableau contenant les compositeurs simples
    composer_d :numpy array[string]
        tableau contenant les compositeurs doubles
        
    '''
    composer = [x for x in composer_b if '/' not in x]
    composer_d = [x for x in composer_b if '/' in x]
    return composer, composer_d


def duree(composer):
    '''
    Renvoie un tableau contenant la durée totale des morceaux pour chaque compositeur
    donné en paramètre
    
    Parameters
    ----------
    composer : numpy array[string]
        Tableau contenant le nom des compositeurs
    
    Returns
    -------
    duration : numpy array
        Tableau contenant la durée totale pour chaque compositeur
    
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
    
    Parameters
    ----------
    composer : numpy array[string]
        Tableau des noms de compositeurs
    duration : numpy array
        Tableau avec la durée totale des morceaux pour chaque compositeur

    Returns
    -------
    None.
        
    '''
    plt.figure()
    plt.bar(composer, duration)
    plt.title("durée totale des morceaux")
    plt.xticks(rotation=90)
    plt.xlabel("compositeur")
    plt.ylabel("durée")
    plt.show()


def threshold(composer, duration, nb):
    '''
    Renvoie le palier pour avoir le nombre de compositeurs nécessaires

    Parameters
    ----------
    composer : numpy array[string]
        Tableau contenant les compositeurs simple
    duration : numpy array
        Tableau contenant la durée totale des morceaux par compositeur
    nb : int
        Nombre de compositeurs souhaités dans le dataset 

    Returns
    -------
    threshold : float
        Seuil pour avoir le nombre de compositeurs souhaités

    '''
    t = len(composer)
    if nb>t :
        print("Erreur dans le choix de la taille, trop de compositeurs demandés")
    else :
        dico = {}
        for i in range(t):
            dico[composer[i]]=duration[i]
        sortedDict = sorted(dico.items(), key=lambda x: x[1])
        print(sortedDict)
        m = t-nb
        threshold = sortedDict[m-1,1]
        return threshold


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
        if count >= threshold:
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

#tab_f = new_dataset(path+"\\maestro-v3.0.0\\maestro-v3.0.0.csv", 12000)
#print(pnd.unique(tab_f['canonical_composer']))
#print(tab_f['duration'].min())


def resampling(audio_filename,curent_duree, cut=30.0):
    '''
    rééchantillonage d'un morceau et découpe en blocs de même taille

    Parameters
    ----------
    audio_filename : string
        chemin du fichier audio à ouvrir
    curent_duree : float
        durée en seconde du morceau
    cut : float
        Temps de coupe en seconde. The default is 30.0

    Returns
    -------
    X : numpy ndarray (2D)
        tableau des données temporelles resamplées et reformatées
    '''
    X = np.array()
    count = curent_duree
    i=0.0
    while i+cut < count :
        t, s = lb.load(audio_filename,sr = sr, offset=i,duration=cut)
        # Regarder composition de t, tableau 1D suffisant normalement
        print(t[0])
        X.append(t[0])
        i = i+cut
    return X
    

# Réécrire la fonction pour return un tableau 2D
def audio_preprocessing(X, nb_mfcc, mfcc_resample=1):
    '''
    Transformation mfc et rééchantillonage
    
    Parameters
    ----------
    X : numpy ndarray (1D)
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
    cepstrum = lb.feature.mfcc(y=X,n_mfcc=nb_mfcc)[:1]
    if mfcc_resample != 1:
        cepstrum = np.transpose(cepstrum)
        cepstrum = np.transpose(cepstrum[::mfcc_resample])
    mfcc = scale(cepstrum, axis=1)
    return mfcc


def write_json(X, composer, json_name):
    '''
    Effectue le mapping de labels et écrit les données dans un json

    Parameters
    ----------
    X : numpy array(3D)
        mfcc resamplées
    composer : string
        label
    json_name : string
        nom du fichier json

    Returns
    -------
    None.

    '''
    # Récupère les données présentes dans le fichier json
    with open(path+json_name) as js:
        data_dict = json.load(js)

    # Définition du nouveau tableau de labels
    dk = list(data_dict.keys())
    morceau = np.empty(len(X[0]))
    for i in range(len(X[0])):
        morceau[i]=dk.index(composer)
    # Trouver comment écrire à l'intérieur du fichier json
    #with open(path+json_name, "w") as js:
    #    json.dump(dictio, js, indent=2)


def pipeline():
    # récupère les compositeurs
    compo = composer(maestro)
    # récupère les compositeurs simples
    composer, double = tri_composer(compo)
    # récupère les durées associées
    duration = duree(composer)
    # Définir nb
    # retourne le seuil 
    threshold(composer, duration, nb)
    datas = new_dataset(path_csv, threshold)
    
    # Initialisation du dictionnaire dans le json
    dicto = {}
    dictio["mapping"] = composer
    dictio["labels"] = []
    dictio["mfcc"] = []
    
    # Définir json_name
    # Création du fichier jason
    # Ecriture du dictionnaire
    with open(path+json_name, "w") as js:
        json.dump(dictio, js, indent=2)
    
    for index, current in datas.iterrows():
        # Resampling
        x1 = resampling(current['audio_filename'], current['duration'], cut=30.0)
        # Mfcc, nb de mfcc ? 
        x1 = audio_preprocessing(x1, '''nb_mfcc''', mfcc_resample=1)
        # Ecriture dans le fichier json
        write_json(x1,current['canonical_composer'],json_name)
