#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 08:33:06 2019

@author: max
"""


from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K
from sklearn.model_selection import GridSearchCV
import numpy as np
#(x_train, y_train), (x_test, y_test) = mnist.load_data()





num_classes = 10
# input image dimensions
img_rows, img_cols = 28, 28




def format_data(x_train, x_test,y_train_in,y_test_in):
    
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    
    return(x_train,x_test, y_train_in, y_test_in)







def simple_model(dense_layer_sizes, filters, kernel_size, pool_size):
    '''Creates model comprised of 2 convolutional layers followed by dense layers
    dense_layer_sizes: List of layer sizes.
        This list has one number for each layer
    filters: Number of convolutional filters in each convolutional layer
    kernel_size: Convolutional kernel size
    pool_size: Size of pooling area for max pooling
    '''

    model = Sequential()
    model.add(Conv2D(filters, kernel_size,
                     padding='valid',
                     input_shape=(28,28,1)))
    model.add(Activation('relu'))
    model.add(Conv2D(filters, kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    for layer_size in dense_layer_sizes:
        model.add(Dense(layer_size))
        model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model


def run_model1(model_architecture,x_train,y_train,x_test,y_test):
    
    
    # use grid search cross validation on desired model
    
    dense_size_candidates = [[32]]
    my_classifier = KerasClassifier(model_architecture, batch_size=32)
    validator = GridSearchCV(my_classifier,
                             param_grid={'dense_layer_sizes': dense_size_candidates,
                                         # epochs is avail for tuning even when not
                                         # an argument to model building function
                                         'epochs': [3],
                                         'filters': [8],
                                         'kernel_size': [3],
                                         'pool_size': [2]},
                             scoring='accuracy',
                             n_jobs=1)
    
    np.isnan(x_train)
    np.isnan(y_train)
    validator.fit(x_train, y_train)
    
    print('The parameters of the best model are: ')
    print(validator.best_params_)
    
    # validator.best_estimator_ returns sklearn-wrapped version of best model.
    # validator.best_estimator_.model returns the (unwrapped) keras model
    best_model = validator.best_estimator_.model
    metric_names = best_model.metrics_names
    metric_values = best_model.evaluate(x_test, y_test)
    for metric, value in zip(metric_names, metric_values):
        print(metric, ': ', value)
    return(best_model)
    
    
def small_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=[28,28,1]))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    return(model)
def medium_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, activation='relu',padding='same', input_shape=(28,28,1)))
    model.add(Conv2D(32, kernel_size=3, activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=2,strides=2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2,strides=2))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))
    
    return(model)
    
    
def run_model(epochs,model,x_train ,y_train, x_test,y_test):
    
    model = model()
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    
    hist = model.fit(x_train, y_train,
              batch_size=200,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    print(hist.history)
    return(score,hist.history)