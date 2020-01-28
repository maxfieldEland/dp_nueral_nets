#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 09:26:56 2019

@author: max

Reclassify unary encoded pixel sets as NMIST digits based on majority count. 

"""


from sklearn.cluster import KMeans  
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import numpy as np

# for now, load regular MNIST Data set

def main(images,labels):
   
    # train on entire data set and classify all 
    km = KMeans(n_clusters = 10)
    #reshape images
    images = images.reshape((120000,28*28))
    
    y_km = km.fit_predict(images)
    
    y_km_train = y_km[:60000]
    y_km_test = y_km[60000:]
    return(y_km_train, y_km_test)

def main_report_acc(images,labels):
   
    # train on entire data set and classify all 
    km = KMeans(n_clusters = 10)
    #reshape images
    images = images.reshape((120000,28*28))
    
    y_km = km.fit_predict(images)
    
    # one hot encode integer labels labels
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = y_km.reshape(len(y_km), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    
    # report jaccard score for each class of labels
    j_score = jaccard_score(labels, onehot_encoded, average = None)
    print('jaccard score = ' , j_score)
    
    y_km_train = onehot_encoded[:60000]
    y_km_test = onehot_encoded[60000:]
    return(y_km_train, y_km_test)


from sklearn.cluster import AgglomerativeClustering
def nick_kmeans(images, labels):
    raw_images, raw_labels, _, _ = load_mnist()
    raw_images = raw_images[:10000] > 115
    # km = AgglomerativeClustering(n_clusters=10)
    km = KMeans(n_clusters=10, n_jobs=-1)

    images_1d = raw_images.reshape(-1, 28*28)
    print(images_1d.dtype)
    pca = PCA(n_components=50)
    images_reduced = pca.fit_transform(images_1d)
    y_pred = km.fit_predict(images_1d)
    for i in range(10):
        show_examples(raw_images[y_pred == i][:12], title=str(raw_labels[:10000][y_pred==i][:12]))

    
    
