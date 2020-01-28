#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 10:08:06 2019

@author: max

Run kmeans clustering on 

"""

import local_dp_data as dp
import nueral_nets as nn
import numpy as np
from mnist_utils import load_mnist
import k_means_clustering as kmeans

# load in mnist data set
train_images, train_labels, test_images, test_labels = load_mnist()


# ------------------------------------------------ Apply local data privacy ---------------------------------------
# apply cutoff --> want reasoning on this
binary_train = train_images > 115

# set probability of bit flip
p = 0.88
q = 0.12

# apply unary encoding
local_train_images = np.array(dp.perturb_examples(binary_train, p, q))
local_train_labels =  np.array(dp.perturb_labels(train_labels, p, q))

local_test_images =  np.array(dp.perturb_examples(binary_train, p, q))
local_test_labels =  np.array(dp.perturb_labels(train_labels, p, q))

# ----------------------------------------- Recluster noised data ----------------------------------

# stack test and train sets
images = np.concatenate((local_train_images,local_test_images))
labels = np.concatenate((local_train_labels,local_test_labels))

# perform k-means clustering with k = 10
clustered_labels_train, clustered_labels_test = kmeans.main(images,labels)

# now, data pairs are local_train_images

# training nueral network on reclustered data
x_train,x_test,y_train, y_test = nn.format_data(local_train_images, local_test_images,clustered_labels_train, clustered_labels_test)



# simple model is a 2d cnn, can be switched out with any architecture 
best_model = nn.run_model(nn.simple_model,x_train,y_train,x_test,y_test)