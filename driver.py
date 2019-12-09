#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 10:08:06 2019

@author: max


"""

import local_dp_data as dp
import nueral_nets as nn
import numpy as np
from mnist_utils import load_mnist
import k_means_clustering as kmeans
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import keras
# load in mnist data set
train_images=np.load('data/train_images.npy')
train_labels=np.load('data/train_labels.npy')
test_images=np.load('data/test_images.npy')
test_labels = np.load('data/test_labels.npy')


# ------------------------------------------------ Apply local data privacy ---------------------------------------
# apply cutoff --> want reasoning on this
binary_train = train_images > 115

# set probability of bit flip


eps = 1925 # preserving total epsilon of 2000

local_train_image = np.array(dp.gaussian_examples(train_images,eps,1/((28*28)**2)))
local_test_image = np.array(dp.gaussian_examples(test_images,eps,1/((28*28)**2)))

q = 0.00000000000000001
p = 1-q

# calculate eps cost( roughyl 75)
e = dp.ue_eps(p,q) + eps
e
local_train_label = dp.perturb_labels(train_labels,p,q)
local_test_label = dp.perturb_labels(test_labels,p,q)


# train nn on noised data
x_train,x_test,y_train, y_test = nn.format_data(local_train_image, local_test_image,local_train_label, local_test_label)

# test out three different levels of depth
models = [nn.small_model, nn.medium_model]

epochs = 20
for idx,model in enumerate(models):

    score, history = nn.run_model(epochs,model,x_train,y_train,x_test,y_test)
    val_acc = history['val_accuracy']
    acc = history['accuracy']
    
    #make plots
    fig = plt.figure(figsize = (8,8))
    plt.plot(list(range(epochs)),val_acc,color ='red',label = 'Validation Accuracy')
    plt.plot(list(range(epochs)),acc,color ='blue',label = 'Test Accuracy')
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    if idx == 0:
        plt.title("Small Model Accuracy with Epsilon = " + str(e))
    if idx == 1:
        plt.title("Meduim Model Accuracy with Epsilon = " + str(e))
    
    






# train on reclustered data
# ----------------------------------------- Recluster noised data ----------------------------------

# stack test and train sets
local_images = np.concatenate((local_train_image,local_test_image))
local_labels = np.concatenate((local_train_label,local_test_label))

images = np.concatenate((local_train_image,local_test_image))
labels = np.concatenate((train_labels,test_labels))



clustered_local_labels_train, clustered_local_labels_test = kmeans.main(local_images,local_labels)
# one hot encode these




# generate confusion matrix
cm = confusion_matrix(clustered_local_labels_train, train_labels)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cmap = plt.cm.Blues
fig, ax = plt.subplots(figsize = (10,10))
im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
ax.figure.colorbar(im, ax=ax)
classes = [0,1,2,3,4,5,6,7,8,9]
# We want to show all ticks...
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       # ... and label them with the respective list entries
       xticklabels=classes, yticklabels=classes,
       title='Confusion Matrix Kmeans',
       ylabel='True label',
       xlabel='Predicted label')

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
# Loop over data dimensions and create text annotations.
fmt = '.2f' 
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")



# now, data pairs are local_train_images

# training nueral network on reclustered data

x_train,x_test,y_train, y_test = nn.format_data(local_train_image, local_test_image,clustered_local_labels_train, clustered_local_labels_test)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

for idx,model in enumerate(models):

    score, history = nn.run_model(epochs,model,x_train,y_train,x_test,y_test)
    val_acc = history['val_accuracy']
    acc = history['accuracy']
    e = 1925
    #make plots
    fig = plt.figure(figsize = (8,8))
    plt.plot(list(range(epochs)),val_acc,color ='red',label = 'Validation Accuracy')
    plt.plot(list(range(epochs)),acc,color ='blue',label = 'Test Accuracy')
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    



# benchmark model with no dp


x_train,x_test,y_train, y_test = nn.format_data(train_images, test_images,train_labels, test_labels)

for idx,model in enumerate(models):

    score, history = nn.run_model(epochs,model,x_train,y_train,x_test,y_test)
    val_acc = history['val_accuracy']
    acc = history['accuracy']
    
    #make plots
    fig = plt.figure(figsize = (8,8))
    plt.plot(list(range(epochs)),val_acc,color ='red',label = 'Validation Accuracy')
    plt.plot(list(range(epochs)),acc,color ='blue',label = 'Test Accuracy')
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    if idx == 0:
        plt.title("Small Model Accuracy with No DP")
    if idx == 1:
        plt.title("Meduim Model Accuracy with No DP")
