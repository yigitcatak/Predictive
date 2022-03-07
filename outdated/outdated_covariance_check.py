#%%
# DEFINITIONS
from math import trunc
from numpy.testing._private.utils import requires_memory
import torch
from torch import random
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split

all_files = [f for f in listdir('dataset/segmented')]
J = 24

def Heatmap(x, title):
    x = np.cov(x,rowvar=False)
    plt.figure(figsize=(15,9))
    plt.imshow(x,cmap='hot')
    plt.title(title)
    plt.colorbar()
    plt.savefig(f'{title}.png',bbox_inches='tight')

def Flatten(l):
    def inner():
        for i in l:
            for j in i:
                yield(j)
    return list(inner())

def GroupSamples(l, J):
    def inner():
        for i in range(0, len(l)-(len(l)%J), J):
            yield l[i:i + J]
    return list(inner())

def Whiten(x, mean=None, eigenVecs=None, diagonal_mat=None ):
    #need to type-check because comparing numpy array with None directly 
    #(array == None) raises an error
    t = type(None)
    if (type(mean) == t) or (type(eigenVecs) == t) or (type(diagonal_mat) == t):
        x = np.array(x)
        mean = x.mean(axis=0)
        x -= mean
        cov = np.cov(x, rowvar=False) #unbiased / divided by N-1
        eigenVals, eigenVecs = np.linalg.eig(cov)
        diagonal_mat = np.diag(1/((eigenVals)**0.5))

        uncorrelated = np.dot(x,eigenVecs)
        whitened = np.dot(uncorrelated, diagonal_mat)
        return whitened, mean, eigenVecs, diagonal_mat,
    else:
        x = np.array(x)
        x -= mean
        uncorrelated = np.dot(x,eigenVecs)
        whitened = np.dot(uncorrelated, diagonal_mat)
        return whitened

x_anomaly = []
x_normal_train = []
x_normal_test = []
x_mixed_train = []
x_mixed_test = []
y_mixed_train = []
y_mixed_test = []
weights = []
df = pd.DataFrame()
working_condition = 0 #to be used to track the 4 working conditions of same class
for name in all_files:
    df = df.append(pd.read_csv(f'dataset/segmented/{name}'), ignore_index=True)
    working_condition += 1 
    if working_condition == 4: # when all the working conditions of the same class is collected shuffle it and add to the general list
        working_condition = 0
        data = GroupSamples(df.drop(['label'], axis=1).values,J)

        # get every Jth label as the labels are given with the average of J segments to classifier
        label = df['label'].values.tolist()[:len(data)*J:J] 

        train, test, y_train, y_test = train_test_split(data,label,test_size=0.75)

        # this is the J grouped length but doesn't matter ratios are same
        weights.append(len(train))

        # unzip the samples as the samples are shuffled keeping segment order
        train = Flatten(train) 
        test = Flatten(test)

        if 'normal' in name: # healthy data to train autoencoder
            x_normal_train += train
            x_normal_test += test

        else: # all anomaly data to check autoencoder error
            x_anomaly += train + test

        # mixed data and label to train and test classifier
        x_mixed_train += train
        x_mixed_test += test
        y_mixed_train += y_train
        y_mixed_test += y_test

        df = pd.DataFrame()
    
#%%
# HEATMAPS

x_normal_train2, mean, eigenVecs, diagonal_mat = Whiten(x_normal_train)
x_mixed_test2 = Whiten(x_mixed_test, mean, eigenVecs, diagonal_mat)
x_mixed_train2 = Whiten(x_mixed_train, mean, eigenVecs, diagonal_mat)
x_normal_test2 = Whiten(x_normal_test, mean, eigenVecs, diagonal_mat)
x_anomaly2 = Whiten(x_anomaly, mean, eigenVecs, diagonal_mat)

Heatmap(x_normal_test, 'Healthy Statistics Whitening - Covariance Matrix of the Healthy Data')
Heatmap(x_normal_test2, 'Healthy Statistics Whitening - Covariance Matrix of the Whitened Healthy Data')

Heatmap(x_anomaly, 'Healthy Statistics Whitening - Covariance Matrix of the Anomaly Data')
Heatmap(x_anomaly2, 'Healthy Statistics Whitening - Covariance Matrix of the Whitened Anomaly Data')

Heatmap(x_mixed_test, 'Healthy Statistics Whitening - Covariance Matrix of the Mixed Data')
Heatmap(x_mixed_test2, 'Healthy Statistics Whitening - Covariance Matrix of the Whitened Mixed Data')
