#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 09:26:56 2019

@author: max

Reclassify unary encoded pixel sets as NMIST digits based on majority count. 

"""

from sklearn.cluster import KMeans  
from sklearn.metrics import jaccard_score
import numpy as np


# for now, load regular MNIST Data set

def main(train_im_path, test_im_path, t_lab_path, test_lab_path:
    train_images = np.load('data/train_images.npy')
    test_images = np.load('data/test_images.npy')
    train_labels = np.load('data/train_labels.npy')
    test_labels = np.load('data/test_labels.npy')
    
    images = np.concatenate((train_images,test_images),axis = 0)
    image_reshape = images.reshape(70000,28*28)
    labels =  np.concatenate((train_labels,test_labels),axis = 0)
    
    # train on entire data set and classify all 
    km = KMeans(n_clusters = 10)
    
    y_km = km.fit_predict(image_reshape)
    
    j_score = jaccard_score(labels, y_km, average = None)
    
    print('jaccard score = ' , jaccard_score)
    return(y_km)
