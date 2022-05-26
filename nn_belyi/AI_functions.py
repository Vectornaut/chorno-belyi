# Functions related to setting up the network.

# INTERESTING: Try to train a network that recognizes the (0,1)-dessign from a (0, inf)-dessign 

###############################################################################################
# Utilities for creating and fine-tuning neural networks in Keras.


import os, sys, scipy.io, scipy.linalg, time, random, pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #don't display warnings; only errors

import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt

# Keras imports
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras import optimizers

from sklearn.metrics import confusion_matrix

##################################################
# Local imports

from dessin_data import TrainingOrbit

##################################################
# Training functions.

def train_belyi(data,
                PCAk = 23,
                BatchSize = 6,
                EpochNum = 10,
                StepSizeMLP = 1e-5,
                StepSizeCNN = 1e-5,
                Balancing = False):
    
    raw_train_x, raw_train_y = data
    train_x = parse_dessin_pairs(raw_train_x)
    train_y = parse_labels(raw_train_y)
    
    bs, ep = BatchSize, EpochNum

    # ** SUPERVISED: MULTILAYER PERCEPTRON
    print("\nSTEP: Training Filter 1 (MLP using X,Y)... ")
    
    hlsizes, numiters, act  = (10, 10, 10), 100, "relu"
    NN = MLPClassifier(hlsizes, StepSizeMLP, act, train_x.shape[1])

    NN.fit(train_x, train_y, batch_size=bs, epochs=ep, verbose=1) # Main MLP-Training.
    print("        ...done.")

    return NN



###############################################################################################
# Classifier constructors.

def MLPClassifier(hlsizes, ss, act, insz):
    model = Sequential()
    model.add(Dense(hlsizes[0], input_dim=insz, kernel_initializer="uniform", activation = act))
    
    for i in range(len(hlsizes)-1):
        model.add(Dense(hlsizes[i+1], kernel_initializer="uniform", activation=act))
    model.add(Dense(1, kernel_initializer="uniform", activation='sigmoid'))

    #sgd = optimizers.SGD(lr=ss, momentum=0.9, nesterov=True)
    opti = optimizers.Adam()
    model.compile(loss='binary_crossentropy',
                  optimizer=opti,
                  metrics=['accuracy'])
    return model


###############################################################################################
# Data handling

def parse_dessin_pairs(train_x, network_type = "MLP"):
    """
    Format the data so that it can be fed into a neural network of the given type.

    Each datapoint of train_x consists of a pair of dessins from the same passport.
    """
    formatted_data = np.empty((0, 6))
    if network_type == "MLP":
       for x, y in train_x:
           a = np.array([[1,2,3,1,2,3]])
           formatted_data = np.append(formatted_data, a, axis=0)
    else:
        raise NotImplementedError

    return formatted_data

def parse_labels(train_y):
    return np.array(train_y)
    
