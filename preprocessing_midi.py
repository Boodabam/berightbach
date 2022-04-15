import numpy as np
import matplotlib.pyplot as plt

import mido as md
#import librosa as lb
import os, json
from json import JSONEncoder

import pandas as pnd
from sklearn.utils import shuffle
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize
#import crepe
import seaborn as sn
import math as mt
import timeit
import warnings
import descriptors

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
                    tab.append((current['midi_filename'],current['canonical_composer'],current['duration']))
                    
                if reduce == True : 
                    # On ajoute uniquement les morceaux tant qu'on ne dépasse pas le seuil minimal
                    if count[current['canonical_composer']] < mind:
                        tab.append((current['midi_filename'],current['canonical_composer'],current['duration']))
                        count[current['canonical_composer']] = count[current['canonical_composer']] + current['duration']
    
    # On met le nouveau tableau dans un tableau pandas
    new_tab = pnd.DataFrame(data = tab, columns=('midi_filename','canonical_composer','duration'))
    return new_tab

def preprocessing_midi_roll(midi_filename):
    midi_filename_ = midi_filename.replace('/','\\')
    mid = md.MidiFile(midi_filename_)
    file_length = mid.length
    notes_roll = np.zeros((round(file_length+1)*10,88))
    notes_vel = np.zeros((round(file_length+1)*10,88))
    notes_len = np.zeros((round(file_length+1)*10,88))
    notes_bin = np.zeros((round(file_length+1)*10,88))
    t=0
    dico = {}
    for msg in mid:
        t = t+msg.time
        if msg.type == 'note_on':
            t_self = round(t,1)
            key = str(msg.note)
            if msg.velocity == 0:
                for i in range(int(dico[key][0]*10),int(t_self*10)):
                    notes_roll[i, msg.note-21] = dico[key][1]
                    notes_len[int(dico[key][0]*10),msg.note-21] = 2*np.arctan(int(t_self*10)-int(dico[key][0]*10))/np.pi
                #dico.pop(key,None)
            else:
                notes_vel[int(t_self*10), msg.note-21] = msg.velocity
                notes_bin[int(t_self*10), msg.note-21] = 1
                dico[key] = (t_self, msg.velocity)
    
    notes_roll=normalize(notes_roll,norm='max')
    notes_vel=normalize(notes_vel,norm='max')
    return notes_roll, file_length, notes_vel, notes_len, notes_bin

def cut_pieces(activation, length, cut=30):
    n_pieces = int(length//cut)
    activations_array = np.empty((n_pieces,cut*10,activation.shape[1]))
    for i in range(n_pieces):
        activations_array[i]=activation[i:i+cut*10]
    return activations_array

def midi_get_features(notes_roll, file_length, notes_vel, notes_len, notes_bin):
    features = np.array([])
    features = np.append(features,file_length/2700)
    features = np.append(features,descriptors.note_rate(notes_bin))
    features = np.append(features,descriptors.value_rate(notes_bin, notes_vel))
    features = np.append(features,descriptors.value_rate(notes_bin, notes_len))
    features = np.append(features,descriptors.notes_center(notes_bin))
    features = np.append(features,descriptors.notes_center(notes_roll))
    features = np.append(features,descriptors.moves_center(notes_bin))
    features = np.append(features,descriptors.moves_center(notes_roll))
    features = np.append(features,descriptors.silence_rate(notes_roll))
    features = np.append(features,descriptors.repetition_rate(notes_bin))
    features = np.append(features,descriptors.autocorr(notes_roll))
    features = scale(features)
    return features

def pipeline_feat(nb,json_name="\\preprocessed_data.json",path_csv=path_csv,path_datas=path_datas,reduce = False, equalize=False):
    datas = new_dataset(path_csv, nb, reduce = reduce)
    # Récupère la liste des compositeurs
    final_compo = all_composer(datas)
    # Initialisation du dictionnaire à écrire dans le fichier json
    dictio = {}
    dictio["mapping"] = final_compo
    dictio["labels"] = []
    dictio["features"] = []
    
    nb_morceaux = np.size(datas, axis=0)
    for index, current in datas.iterrows():
        start_timer = timeit.default_timer()
        
        notes_roll, file_length, notes_vel, notes_len, notes_bin = preprocessing_midi_roll(path_datas+current['midi_filename'])
        features = midi_get_features(notes_roll, file_length, notes_vel, notes_len, notes_bin)
        
        dictio["labels"] = np.append(dictio["labels"],int(np.where(final_compo == current['canonical_composer'])[0]))
        if (len(dictio["features"])==0):
            dictio["features"] = [features]
        else :
            dictio["features"] = np.append(dictio["features"],[features], axis=0)
    
        stop_timer = timeit.default_timer()
        print(index +1,'/', nb_morceaux, "  time:",stop_timer-start_timer)
    
    print("preprocessing ended. Start writing")
    # Ecriture du dictionnaire dans le fichier json    
    with open(path+json_name, "w",encoding="utf8") as jsf:
        json.dump(dictio, jsf, cls=NumpyArrayEncoder, indent=2)
    
    
    
    
    
def pipeline_roll(nb,json_name="\\preprocessed_data.json",path_csv=path_csv,path_datas=path_datas,reduce = False):
    # Création du tableau pandas avec le nombre de compositeurs souhaités
    datas = new_dataset(path_csv, nb, reduce = reduce)
    # Récupère la liste des compositeurs
    final_compo = all_composer(datas)
    # Initialisation du dictionnaire à écrire dans le fichier json
    dictio = {}
    dictio["mapping"] = final_compo
    dictio["labels"] = []
    dictio["features"] = []
    
    nb_morceaux = np.size(datas, axis=0)
    for index, current in datas.iterrows():
        start_timer = timeit.default_timer()
        # Resampling
        matrix, l = preprocessing_midi_roll(path_datas+current['midi_filename'])[:2]
        x1 = cut_pieces(matrix, l , cut=30)
        # Définition du nouveau tableau de labels, associe pour chaque bout découpé son compositeur via son indice dans le dictionnaire
        morceau = np.ones(np.size(x1,axis=0),dtype=int)*int(np.where(final_compo == current['canonical_composer'])[0])
        # on ajoute les nouveaux indices au dictionnaire
        dictio["labels"] = np.append(dictio["labels"],morceau)
        for i in range(np.size(x1,axis=0)):
            if (len(dictio["features"])==0):
                dictio["features"] = [x1[i]]
            else :
                dictio["features"] = np.append(dictio["features"],[x1[i]], axis=0)
        
        stop_timer = timeit.default_timer()
        print(index +1,'/', nb_morceaux, "  time:",stop_timer-start_timer)
    
    print("preprocessing ended. Start writing")
    # Ecriture du dictionnaire dans le fichier json    
    with open(path+json_name, "w",encoding="utf8") as jsf:
        json.dump(dictio, jsf, cls=NumpyArrayEncoder, indent=2)
        
if __name__ == "__main__":
    pipeline_feat(6)
    #pipeline_roll(6,reduce=True)