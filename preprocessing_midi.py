import numpy as np
import matplotlib.pyplot as plt

import mido as md
#import librosa as lb
import os, json
from json import JSONEncoder

import pandas as pnd
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import seaborn as sn
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
    """
    Transforme un fichier midi en midi maps pour préprocessing

    Parameters
    ----------
    midi_filename : string
        nom du fichier midi

    Returns
    -------
    notes_roll : numpy array (2D)
        midi map
    file_length : float
        longueur du morceau en secondes
    notes_vel : numpy array (2D)
        midi map des attaques avec information sur la vélocité des notes
    notes_len : numpy array (2D)
        midi map des attaques avec information sur la longueur des notes
    notes_bin : numpy array (2D)
        midi map des attaques avec information binaire noteOn

    """
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
    """
    Découpe une midi map en morceaux de cut secondes

    Parameters
    ----------
    activation : numpy array (2D)
        midi map
    length : float
        longueur du morceau en secondes
    cut : TYPE, optional
        longueur des morceaux découpés The default is 30.

    Returns
    -------
    activations_array : numpy array (3D)
        tableau des midi-maps découpées

    """
    n_pieces = int(length//cut)
    activations_array = np.empty((n_pieces,cut*10,activation.shape[1]))
    for i in range(n_pieces):
        activations_array[i]=activation[i:i+cut*10]
    return activations_array

def midi_get_features(notes_roll, file_length, notes_vel, notes_len, notes_bin):
    """
    transforme les midi maps en features. les features sont données dans le fichier descriptors.py

    Parameters
    ----------
    notes_roll : numpy array (2D)
        midi map
    file_length : float
        longueur du morceau en secondes
    notes_vel : numpy array (2D)
        midi map des attaques avec information sur la vélocité des notes
    notes_len : numpy array (2D)
        midi map des attaques avec information sur la longueur des notes
    notes_bin : numpy array (2D)
        midi map des attaques avec information binaire noteOn

    Returns
    -------
    features : numpy array (1D)
        tableau des features

    """
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
    """
    Pipeline de préprocessing midi pour la stratégie de l'utilisation de la méthode des features codées à la main'

    Parameters
    ----------
    nb : INT
        Nombre de compositeurs désirés
    json_name : string
        nom du fichier json à écrire The default is "\\preprocessed_data.json".
    path_csv : string, optional
        The default is path_csv.
    path_datas : string, optional
        The default is path_datas.
    reduce : bool, optional
        Inefficace pour l'intant, à implémenter. The default is False.
    equalize : bool, optional
        Si True, on découpe des sections sur lesquelles on travaille, si False on travaille sur les morceaux entiers.
        The default is False.

    Returns
    -------
    dictio: dict
        le dictionnaire contenant le mapping, les labels et les features

    """
    datas = new_dataset(path_csv, nb, reduce = reduce)
    # Récupère la liste des compositeurs
    final_compo = all_composer(datas)
    # Initialisation du dictionnaire à écrire dans le fichier json
    dictio = {}
    dictio["mapping"] = final_compo
    dictio["labels"] = []
    dictio["features"] = []
    
    nb_morceaux = np.size(datas, axis=0)
    #pour tous les morceaux
    for index, current in datas.iterrows():
        start_timer = timeit.default_timer()
        #créer les midi maps
        notes_roll, file_length, notes_vel, notes_len, notes_bin = preprocessing_midi_roll(path_datas+current['midi_filename'])
        #si on souhaite découper les morceaux en extraits
        if equalize:
            cut = 30 #temps de découpe en secondes
            #On découpe toutes les midi map
            notes_roll = cut_pieces(notes_roll,file_length, cut = cut)
            notes_vel = cut_pieces(notes_vel,file_length, cut = cut)
            notes_len = cut_pieces(notes_len,file_length, cut = cut)
            notes_bin = cut_pieces(notes_bin,file_length, cut = cut)
            morceau = np.ones(np.size(notes_roll,axis=0),dtype=int)*int(np.where(final_compo == current['canonical_composer'])[0])
            # on ajoute les nouveaux indices au dictionnaire
            dictio["labels"] = np.append(dictio["labels"],morceau)
            for i in range(np.size(notes_roll,axis=0)):
                #pour chaque extrait, on calcule les features et on les ajoute au dictionnaire
                features = midi_get_features(notes_roll[i], file_length, notes_vel[i], notes_len[i], notes_bin[i])
                if (len(dictio["features"])==0):
                    dictio["features"] = [features]
                else :
                    dictio["features"] = np.append(dictio["features"],[features], axis=0)
        #si on souhaite garder les morceaux entiers
        else:
            #calcul des features
            features = midi_get_features(notes_roll, file_length, notes_vel, notes_len, notes_bin)
            # on ajoute les nouveaux indices au dictionnaire
            dictio["labels"] = np.append(dictio["labels"],int(np.where(final_compo == current['canonical_composer'])[0]))
            #on ajoute les features ajoute au dictionnaire
            if (len(dictio["features"])==0):
                dictio["features"] = [features]
            else :
                dictio["features"] = np.append(dictio["features"],[features], axis=0)
        
        #print le temps de calcul de chaque morceau
        stop_timer = timeit.default_timer()
        print(index +1,'/', nb_morceaux, "  time:",stop_timer-start_timer)
    
    print("preprocessing ended. Start writing")
    # Ecriture du dictionnaire dans le fichier json    
    with open(path+json_name, "w",encoding="utf8") as jsf:
        json.dump(dictio, jsf, cls=NumpyArrayEncoder, indent=2)

    return dictio
    
    
    
    
def pipeline_roll(nb,json_name="\\preprocessed_data.json",path_csv=path_csv,path_datas=path_datas,reduce = False):
    """
    Pipeline de préprocessing midi pour la stratégie de la convolution sur les features map'

    Parameters
    ----------
    nb : INT
        Nombre de compositeurs désirés
    json_name : string
        nom du fichier json à écrire The default is "\\preprocessed_data.json".
    path_csv : string, optional
        The default is path_csv.
    path_datas : string, optional
        The default is path_datas.
    reduce : bool, optional
        Inefficace pour l'intant, à implémenter. The default is False.
    Returns
    -------
    None.

    """
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
        x1 = cut_pieces(matrix, l , cut=60)
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

def visualize(dico):
    """
    Visualisation: PCA et correlation

    Parameters
    ----------
    dico : dict
        dictionnaire des données préprocessées

    Returns
    -------
    None.

    """
    #PCA
    colors = ["navy","turquoise","darkorange","green","red","brown"]
    X = dico["features"]
    pca = PCA(n_components=2)
    pca.fit(X)
    X = pca.transform(X)
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    for i, name in enumerate(dico["mapping"]):
        x_0 = np.array([])
        x_1 = np.array([])
        for j, x in enumerate(X):
            if int(dico["labels"][j])==i: 
                x_0 = np.append(x_0,x[0])
                x_1 = np.append(x_1,x[1])
        ax.scatter(x_0,x_1,c=colors[i],label=name)
    ax.set_title("Espace des features après PCA")
    ax.legend()
    
    #Correlation Matrix
    feat = ['length','note_rate_mean','note_rate_std','vel_mean','vel_std','len_mean','len_std','bary_mean_bin','bary_std_bin', 'entr_bin','bary_mean_roll','bary_std_roll','entr_roll','move_bary_std_bin','harm_amp_std_bin','bary_speed_mean_bin','bary_speed_amp_bin','speed_harm_amp_mean_bin','speed_harm_amp_std_bin','entr_var_mean_bin','entr_var_std_bin','move_bary_std_roll','harm_amp_std_roll','bary_speed_mean_roll','bary_speed_amp_roll','speed_harm_amp_mean_roll','speed_harm_amp_std_roll','entr_var_mean_roll','entr_var_std_roll','silence','repetition','autocorr']
    fig = plt.figure(figsize=(20,20))
    
    ax = fig.add_subplot(111)
    feat_dic = {}
    X = dico["features"]
    for i, name in enumerate(feat):
        feat_dic[name] = []
        for j, x in enumerate(X):
            feat_dic[name] = np.append(feat_dic[name],x[i])
    df = pnd.DataFrame(feat_dic,columns=feat)
    corrMatrix = df.corr()
    sn.heatmap(corrMatrix, annot=True)
     
    
if __name__ == "__main__":
    d = pipeline_feat(6, equalize=True)
    visualize(d)
    #pipeline_roll(6,reduce=True)