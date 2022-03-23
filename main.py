import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import tensorflow.keras as kr
import librosa as lb
import os, json

path = os.getcwd()+"//preprocessed_data.json"
test_size = 0.3
topo_network = (128, 128, 128)
used_package = "tensorflow"
learning_rate = 0.001
batch_size = 32
nb_epoch = 50

"""
Import des data à partir d'un json
Format du json:
    le tableau (3D) des mfcc préprocessées de tout le dataset (je fais le split ici finalement) avec l'étiquette features
    le tableau (1D) des classes avec l'étiquette labels
    Le nombre de classes avec l'étiquette nb_classes
Evidemment à changer si l'oscarito devient un hyperparamètre à optimiser  
"""
def load_data(path=path):
    data = json.load(open(path, "r"))
    X = np.array(data["features"])
    Y = np.array(data["labels"])
    nb_classes = data["nb_classes"]
    return X, Y, nb_classes

"""
fonction principale
import, split des données (que test set et train_set pour l'instant'), 
j'ai implémenté deux solveurs différents, à voir lequel se débrouille le mieux
"""

if __name__ == "__main__":
    #importer les données
    X, Y, nb_classes = load_data()
    #splitter le dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
    
    #solveur: On utilise SGD (Adam a l'air plus performant mais je sais pas trop comment il marche)
    if used_package == "tensorflow":
        #avec tensorflow
        network = kr.Sequential([
            #convolution
            kr.layers.Conv2D(2,(3,3)),
            #pooling
            kr.layers.MaxPooling2D(pool_size=(2,2)),
            #entrée
            kr.layers.Flatten(input_shape=(X.shape[1], X.shape[2])),
            #couches intermédiaires
            kr.layers.Dense(topo_network[0], activation='relu'),
            kr.layers.Dense(topo_network[1],activation='relu'),
            kr.layers.Dense(topo_network[2],activation='relu'),
            #Sortie
            kr.layers.Dense(nb_classes, activation='softmax')
        ])
        optimizer = kr.optimizers.SGD(learning_rate=learning_rate)
        
        network.compile(optimizer=optimizer, loss="MeanSquaredError", metrics=["Accuracy"])
        #un résumé du réseau
        network.summary()
        #entrainement
        #si on veut ajouter une cross-validation, splitter à nouveau et remplir le champ validation_data
        #source https://keras.io/api/models/model_training_apis/
        network.fit(X_train,Y_train, batch_size=batch_size, epochs=nb_epoch )
        
    elif used_package == "MLPClassifier":
        #avec scikit
        #si on veut ajouter une cross-validation, mettre early_stopping sur true et renseigner validation_fraction et n_iter_no_change
        #source https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
        network = MLPClassifier(topo_network, activation='relu',solver='sgd', max_iter=nb_epoch, batch_size=batch_size, learning_rate='adaptive', learning_rate_init=learning_rate, early_stopping=False)
        #entrainement
        network.fit(X_train,Y_train)






















