import numpy as np
import matplotlib.pyplot as plt

import librosa as lb
import os, json
from json import JSONEncoder

import pandas as pnd
from sklearn.utils import shuffle
from sklearn.preprocessing import scale
import seaborn as sn
import math as mt
import timeit
import warnings


warnings.filterwarnings("ignore")
# Préparation pour l'écriture des ndarray dans le fichier json
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


''' Déclaration des variables globales '''

sr = 22050
path = os.path.dirname(os.path.realpath(__file__))

path_datas = path +"\\maestro-v3.0.0\\"
path_csv = path + "\\maestro-v3.0.0\\maestro-v3.0.0.csv"

path_minidata = path+"\\minidataset\\"
path_minicsv = path_minidata+"minimaestro.csv"

            
def all_composer(maestro):
    '''
    Renvoie un tableau avec tous les compositeurs du fichier csv
    
    Parameters
    ----------
    maestro : tableau pandas
        tableau contenant les données du fichier csv
    
    Returns
    -------
    call_compo : numpy array[string]
        tableau contenant tous les compositeurs du dataset
    '''
    all_compo = pnd.unique(maestro['canonical_composer'])
    return all_compo


def tri_composer(compo):
    '''
    Tri les compositeurs entre les morceaux ayant un unique compositeur et les morceaux contenant deux compositeurs
    
    Parameters
    ----------
    compo : numpy array[string]
        Liste des compositeurs à trier
    
    Returns
    -------
    composer_s : numpy array[string]
        tableau contenant les compositeurs simples
    composer_d :numpy array[string]
        tableau contenant les compositeurs doubles
        
    '''
    # Compositeur simple
    composer_s = [x for x in compo if '/' not in x]
    # Compositeur double
    composer_d = [x for x in compo if '/' in x]
    return composer_s, composer_d


def duree(composer, datas):
    '''
    Renvoie un tableau contenant la durée totale des morceaux pour chaque compositeur
    donné en paramètre
    
    Parameters
    ----------
    composer : numpy array[string]
        Tableau contenant le nom des compositeurs
    datas : tableau pandas
        Tableau pandas des données
    
    Returns
    -------
    duration : numpy array
        Tableau contenant la durée totale pour chaque compositeur
    
    '''
    duration=[]
    for c in composer:
        count =0
        for index, current in datas.iterrows():
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


def list_compo(composer, duration ,nb):
    '''
    Construit la liste des compositeurs avec le nombre désiré

    Parameters
    ----------
    composer : nd array
        Tableau contenant les compositeurs du tableau pandas
    duration : nd array
        Tableau contenant la durée totale des morceaux associées à un compositeur
    nb : int
        Nombre de compositeurs désirés pour les tests

    Returns
    -------
    final_compo : nd array
        Tableau contenant le nombre de compositeurs désiré possédant le plus de données

    '''
    t = len(composer)
    if nb>t :
        print("Erreur dans le choix de la taille, trop de compositeurs demandés")
    else :
        dico = {}
        for i in range(t):
            dico[composer[i]]=duration[i]
        sortedDict = np.array(sorted(dico.items(), key=lambda x: x[1]))
        
        final_compo = []
        
        for i in range(np.size(sortedDict,axis=0)):
            if i>= np.size(sortedDict,axis=0)-nb:
                final_compo.append(sortedDict[i][0])
        return final_compo


def new_dataset(path, nb, reduce = False):
    '''
    Création du tableau pandas avec les morceaux du nombre de compositeur désiré

    Parameters
    ----------
    path : string
        Chemin du fichier csv
    nb : int
        Nombre de compositeurs souhaités
    reduce : boolean
        Indique si on réduit le nombre de morceaux dans le data set. The default is False

    Returns
    -------
    new_tab : pnd array
        Tableau pandas contenant les morceaux, leur compositeur et leur durée

    '''
    # Récupère le fichier csv et on le met dans un tableau pandas
    currentFile = open(path, encoding="utf8")
    File = pnd.read_csv(currentFile)
    # Déclaration du nouveau tableau
    tab = []
    
    # Récupère tout les compositeurs
    all_compo = all_composer(File)
    # Fait le tri dans les compositeurs
    compo_simple, compo_double = tri_composer(all_compo)
    # Récupère les durées
    duration = duree(compo_simple, File)
    # Récupère les compositeurs à ajouter dans notre tableau
    composer = list_compo(compo_simple,duration,nb)
    
    if reduce == True :
        # Récupère la durée minimale pour les nouveaux compositeurs
        new_duration = duree(composer, File)
        mind = min(new_duration)
        # Déclaration du tableau de compte
        count = {}
        for i in range(len(composer)):
            count[composer[i]]=0
    
    # Remplissage du tableau
    for index, current in File.iterrows():
        
        # On vérifie que le morceau ne possède qu'un seul compositeur
        if '/' not in current['canonical_composer']:
           
            # On vérifie qu'il est dans notre liste de compositeur à ajouter
            if current['canonical_composer'] in composer:
                if reduce == False :
                    # On ajoute les morceaux au nouveau tableau
                    tab.append((current['audio_filename'],current['canonical_composer'],current['duration']))
                    
                if reduce == True : 
                    # On ajoute uniquement les morceaux tant qu'on ne dépasse pas le seuil minimal
                    if count[current['canonical_composer']] < mind:
                        tab.append((current['audio_filename'],current['canonical_composer'],current['duration']))
                        count[current['canonical_composer']] = count[current['canonical_composer']] + current['duration']
    
    # On met le nouveau tableau dans un tableau pandas
    new_tab = pnd.DataFrame(data = tab, columns=('audio_filename','canonical_composer','duration'))
    return new_tab


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
        Temps de coupe en seconde. The default is 10.0

    Returns
    -------
    X : numpy ndarray (2D)
        tableau des données temporelles resamplées et reformatées
    '''
    audio_filename = audio_filename.replace('/','\\')
    X = np.array([])
    count = curent_duree
    i=0.0
    while i+cut < count :
        t, s = lb.load(audio_filename,sr = sr, offset=i,duration=cut)
        if (len(X)==0):
                X = [t]
        else :
            X = np.append(X,[t], axis=0)
    
        i = i+cut
    return X
    

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
    mfcc : numpy ndarray (2D)
        mfcc rééchantillonnés

    '''
    mfcc = lb.feature.mfcc(y=X,n_mfcc=127)[127-nb_mfcc:]
    if mfcc_resample != 1:
        mfcc = np.transpose(mfcc)
        mfcc = np.transpose(mfcc[::mfcc_resample])
    mfcc = scale(mfcc, axis=1)
    return mfcc


def pipeline(nb,json_name,path_csv=path_csv,path_datas=path_datas, reduce = False,nb_mfcc=107, mfcc_resample=1):
    '''

    Parameters
    ----------s
    nb : INT
        Nombre de compositeurs désirés
    json_name : string
        nom du fichier json à écrire

    Returns
    -------
    None.

    '''
    # Création du tableau pandas avec le nombre de compositeurs souhaités
    datas = new_dataset(path_csv, nb, reduce = reduce)
    # Récupère la liste des compositeurs
    final_compo = all_composer(datas)
    # Initialisation du dictionnaire à écrire dans le fichier json
    dictio = {}
    dictio["mapping"] = final_compo
    dictio["labels"] = []
    dictio["mfcc"] = []
    
    nb_morceaux = np.size(datas, axis=0)
    for index, current in datas.iterrows():
        start_timer = timeit.default_timer()
        # Resampling
        x1 = resampling(path_datas+current['audio_filename'], current['duration'], cut=30.0)
        # Définition du nouveau tableau de labels, associe pour chaque bout découpé son compositeur via son indice dans le dictionnaire
        morceau = np.ones(np.size(x1,axis=0),dtype=int)*int(np.where(final_compo == current['canonical_composer'])[0])
        # on ajoute les nouveaux indices au dictionnaire
        dictio["labels"] = np.append(dictio["labels"],morceau)
        for i in range(np.size(x1,axis=0)):
            # Préprocessing de chaque nouvelle coupe par morceau
            x2 = audio_preprocessing(x1[i], nb_mfcc,mfcc_resample=mfcc_resample)
            # Remplissage du dictionnaire
            if (len(dictio["mfcc"])==0):
                dictio["mfcc"] = [x2]
            else :
                dictio["mfcc"] = np.append(dictio["mfcc"],[x2], axis=0)
        
        stop_timer = timeit.default_timer()
        print(index +1,'/', nb_morceaux, "  time:",stop_timer-start_timer)
        i= i+1
    
    print("preprocessing ended. Start writing")
    # Ecriture du dictionnaire dans le fichier json    
    with open(path+json_name, "w",encoding="utf8") as jsf:
        json.dump(dictio, jsf, cls=NumpyArrayEncoder, indent=2)
