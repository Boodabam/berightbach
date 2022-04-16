import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

def note_rate(notes_bin):
    """
    Moyenne et écart-type normalisés du nombre de notes par secondes

    Parameters
    ----------
    notes_bin : numpy array (2D)
        midi map binaire

    Returns
    -------
        moyenne, écrat-type
    """
    M = notes_bin
    note_per_sec = np.empty(len(M)//10)
    for i in range(len(M)//10):
        note_per_sec[i]=M[i*10:i*10+10].sum()
    return np.mean(note_per_sec), np.std(note_per_sec)

def value_rate(notes_bin,notes_val):
    """
    Moyenne et écart-type normalisés des vélocité/longueur sur tout le morceau

    Parameters
    ----------
    notes_bin : numpy array (2D)
        midi map binaire
    notes_bin : numpy array (2D)
        midi map en vélocité OU en longueur

    Returns
    -------
    Moyenne, écart-type

    """
    mean = np.sum(notes_val)/np.sum(notes_bin)
    std = np.sqrt(np.sum((notes_val-mean)**2)/np.sum(notes_bin))
    return mean, std

def notes_center(notes_bin):
    """
    moyenne, écart-type et entropie normalisées de la distribution du barycentre des notes en hauteur

    Parameters
    ----------
    notes_bin : numpy array (2D)
        midi map binaire OU midi roll

    Returns
    -------
    moyenne, écart-type, entropie

    """
    nb_notes =notes_bin.sum()
    entro_max = np.log2(88)
    if nb_notes == 0:
        return 0.5, 0.25, 1
    else:
        sum_bin = np.sum(notes_bin,axis=0)/nb_notes
        mean_note = np.dot(sum_bin,np.arange(88))/88
        std = np.sqrt(np.sum((sum_bin-mean_note)**2)/nb_notes)
        entr = entropy(sum_bin)/entro_max
        return mean_note, std, entr

def moves_center(notes_bin):
    """
    Etude dynamique de notes_center par secondes. Evaluation au cours du temps de:
        écart-type du barycentre harmonique
        écart type de l'amplitude harmonique
        Vitesse moyenne de déplacement du barycentre
        Amplitude moyenne de la vitesse de déplacement du barycentre
        Vitesse moyenne de changement de l'amplitude harmonique
        Amplitude moyenne de la vitesse de changement de l'amplitude harmonique
        Ecart-type de l'entropie de la distribution des notes comme mesure de la tonalité
        Variation moyenne d'entropie
        Ecart-type de la variation d'entropie'
        Toutes ces valeurs sont normalisées

    Parameters
    ----------
    notes_bin : numpy array (2D)
        midi map binaire OU midi roll

    Returns
    -------
    cf. description

    """
    M = notes_bin
    tab_mean = np.empty(len(M)//10)
    tab_std = np.empty(len(M)//10)
    tab_entr = np.empty(len(M)//10)
    for i in range(len(M)//10):
        tab_mean[i], tab_std[i], tab_entr[i] = notes_center(M[i*10:i*10+10])
    d_tab_mean = np.array([(tab_mean[i]-tab_mean[i+1]+1)/2 for i in range(len(M)//10-1)])
    d_tab_std = np.array([(tab_std[i]-tab_std[i+1]+1)/2 for i in range(len(M)//10-1)])
    d_tab_entr = np.array([(tab_entr[i]-tab_entr[i+1]+1)/2 for i in range(len(M)//10-1)])
    return np.std(tab_mean), np.std(tab_std), np.mean(d_tab_mean), np.std(d_tab_mean), np.mean(d_tab_std), np.std(d_tab_std), np.std(tab_entr), np.mean(d_tab_entr), np.std(d_tab_entr)
    

def silence_rate(notes_roll):
    """
    Taux de silence normalisé

    Parameters
    ----------
    notes_roll : numpy array (2D)
        midi map

    Returns
    -------
    taux de silence normalisé

    """
    s=0
    for i in notes_roll:
        if not np.any(i):
            s=s+1
    return s/len(notes_roll)

def repetition_rate(notes_bin):
    """
    Taux de répétition des notes normalisé

    Parameters
    ----------
    notes_bin : numpy array (2D)
        midi map binaire.

    Returns
    -------
    Taux de répétition des notes normalisé

    """
    nb_notes =notes_bin.sum()
    rep = 0
    for i, note_slice in enumerate(np.transpose(notes_bin)):
        for j, note_cell in enumerate(note_slice):
            if note_cell and np.any(note_slice[j+1:j+15]):
                rep = rep+np.sum(note_slice[j+1:j+15])
                j=j+15
    return rep/nb_notes

def autocorr(notes_roll):
    """
    Autocorrelation comme mesure des répétitions de motif. Normalisé sur le max du dataset

    Parameters
    ----------
    notes_roll : numpy array (2D)
        midi map

    Returns
    -------
    autocorrelation
    """
    M = notes_roll
    M_corr = np.empty((len(M)//10,len(M)//10))
    for i in range(len(M)//10):
        for j in range(len(M)//10):
            M_corr[i,j]=np.ndarray.max(np.correlate(np.ravel(M[i*10:i*10+10]),np.ravel(M[j*10:j*10+10])))
    return np.ndarray.max(M_corr)/5.66