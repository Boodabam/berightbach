import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow import reshape
from seaborn import heatmap
from pandas import DataFrame
import os, json


path = os.getcwd() + "\\preprocessed_data.json"
test_size = 0.2
learning_rate = 0.0005
batch_size = 32
nb_epoch = 150


def load_data(path=path):
    """
    Load les données préprocessées et leurs labels à parti d'un json

    Parameters
    ----------
    path : string, optional
        path du fichier json contenant les données préprocessées. The default is path.

    Returns
    -------
    X : numpy ndarray (3D)
        Données des mfcc.
    Y : numpy array (1D)
        classes mappées.
    nb_classes : int
        nombre de classes.

    """
    print("loading data")
    data = json.load(open(path, "r"))
    X = np.asarray(data["features"])
    Y_float = np.asarray(data["labels"])
    mapping = np.asarray(data["mapping"])
    nb_classes = len(np.asarray(data['mapping']))
    Y = np.zeros((len(Y_float),nb_classes))
    for i, val in enumerate(Y_float):
        Y[i,int(val)]=1
    print("Data loaded")
    return X, Y, mapping, nb_classes

def plot_acuracy(hist):
    """
    trace accuracy et loss pour les ensemble de test et de validation

    Parameters
    ----------
    hist : keras hist
        l'historique d'entrainement keras.

    Returns
    -------
    None.

    """
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(2,1,1)
    ax.plot(hist.history['accuracy'])
    ax.plot(hist.history['val_accuracy'])
    ax.set_title('model accuracy')
    ax.set_ylabel('accuracy')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'validation'], loc='upper left')
    ax = fig.add_subplot(2,1,2)
    ax.plot(hist.history['loss'])
    ax.plot(hist.history['val_loss'])
    ax.set_title('model loss')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'validation'], loc='upper left')

if __name__ == "__main__":
    # importer les données

    X, Y, mapping, nb_classes = load_data()
    
    #X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2],1))
    # splitter le dataset
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
    
    #On utilise SGD
    network = Sequential()
            # entrée
    network.add(Input(shape=(X.shape[1])))
    """
            # convolution
    network.add(Conv2D(8, (3, 3)))
    
    network.add(BatchNormalization(axis=-1))
            # pooling
    network.add(MaxPooling2D(pool_size=(2, 2)))
            # convolution
    network.add(Conv2D(16, (3, 3)))
    
    network.add(BatchNormalization(axis=-1))
            # pooling
    network.add(MaxPooling2D(pool_size=(2, 1)))
            # convolution
    network.add(Conv2D(32, (3, 3)))
    
    network.add(BatchNormalization(axis=-1))
            # pooling
    network.add(MaxPooling2D(pool_size=(2, 1)))
            # convolution
    network.add(Conv2D(64, (3, 3)))
    
    network.add(BatchNormalization(axis=-1))
            # pooling
    network.add(MaxPooling2D(pool_size=(2, 1)))
             # convolution
    network.add(Conv2D(128, (3, 3)))
    
    network.add(BatchNormalization(axis=-1))
            # pooling
    network.add(MaxPooling2D(pool_size=(2, 1)))
            #applatir
    network.add(Flatten())
    
    network.add(Dropout(0.3))
    """
            # couches intermédiaires
    network.add(Dense(128, activation='relu'))
    network.add(Dense(128, activation='relu'))
    network.add(Dense(128, activation='relu'))
    network.add(Dense(128, activation='relu'))
    network.add(Dense(128, activation='relu'))

            # Sortie
    network.add(Dense(nb_classes, activation='softmax'))

    optimizer = Adam(learning_rate=learning_rate)

    network.compile(optimizer=optimizer, loss="MeanSquaredError", metrics=["accuracy"])
        # un résumé du réseau
    network.summary()
        # entrainement
    
    
    
    hist = network.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, validation_split=0.2)
    
    plot_acuracy(hist)
    
    results = network.evaluate(X_test, Y_test, batch_size=batch_size)

    print("test loss, test acc:", results)