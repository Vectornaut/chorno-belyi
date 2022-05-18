# Functions related to setting up the network.

# INTERESTING: Try to train a network that recognizes the (0,1)-dessign from a (0, inf)-dessign 

###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
# Utilities for creating and fine-tuning neural networks in Keras.



import os, sys, scipy.io, scipy.linalg, time, random, pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #don't display warnings; only errors
import numpy as np, tensorflow as tf, matplotlib.pylab as plt
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from numpy import genfromtxt
#from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from time import time, asctime
import pickle as pk

import tensorflow as tf
from keras.models import Sequential,load_model
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical
from keras import optimizers

##################################################
# Local imports

from dessin_data import TrainingOrbit


def train_belyi(data,
                PCAk = 23,
                BatchSize = 6,
                EpochNum = 10,
                StepSizeMLP = 1e-5,
                StepSizeCNN = 1e-5,
                Balancing = False):
    
    train_x, train_y = data
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

