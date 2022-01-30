import torch
import pandas as pd
import numpy as np
from scipy import linalg
from os import listdir
from random import shuffle

Sample_Length = 1200
N = 30
J = int(Sample_Length/N)
FanEnd = [f for f in listdir('CWRU/segmented/fanend')]
DriveEnd = [f for f in listdir('CWRU/segmented/driveend')]

Class_Weights = dict(zip(list(range(10)),list(0 for i in range(10))))

def GroupSamples(l, J):
    def inner():
        for i in range(0, len(l)-(len(l)%J), J):
            yield l[i:i + J]
    return list(inner())

def Flatten(l):
    def inner():
        for i in l:
            for j in i:
                yield(j)
    return list(inner())

def Whiten(x, mean=None, eigen_vecs=None, diagonal_mat=None ):
    #need to type-check because comparing numpy array with None directly 
    #(array == None) raises an error
    t = type(None)
    if (type(mean) == t) or (type(eigen_vecs) == t) or (type(diagonal_mat) == t):
        x = np.array(x)
        mean = x.mean(axis=0)
        x -= mean
        cov = np.cov(x, rowvar=False) #unbiased / divided by N-1
        eigenVals, eigen_vecs = np.linalg.eig(cov)
        diagonal_mat = np.diag(1/((eigenVals)**0.5))

        uncorrelated = np.dot(x,eigen_vecs)
        whitened = np.dot(uncorrelated, diagonal_mat)
        return whitened, mean, eigen_vecs, diagonal_mat
    else:
        x = np.array(x)
        x -= mean
        uncorrelated = np.dot(x,eigen_vecs)
        whitened = np.dot(uncorrelated, diagonal_mat)
        return whitened

x_train = []
x_test = []
y_train = []
y_test = []

for name in FanEnd:
    df = pd.read_csv(f'CWRU/segmented/fanend/{name}')
    data = df.drop(['label'], axis=1).values.tolist()[:len(df)-len(df)%J]

    # We keep 1 label for each J segment, for memory efficicency
    label = df['label'].values.tolist()[:len(data):J]
    Class_Weights[label[0]] += len(label)

    idx = int(len(label)*0.1)
    train = data[0:idx*J]
    test = data[idx*J:]
    train_label = label[0:idx]
    test_label = label[idx:]

    # Group J consecutive segments and stick them with their labels
    # so they don't get lost during shuffling
    train = GroupSamples(train,J)
    train = list(zip(train,train_label))
    shuffle(train)
    train, train_label = zip(*train)
    train = Flatten(train)

    x_train += train
    x_test += test
    y_train += train_label
    y_test += test_label

x_train, mean, eigen_vecs, diagonal_mat = Whiten(x_train)
x_test = Whiten(x_test, mean, eigen_vecs, diagonal_mat)

x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)
Class_Weights = torch.tensor(list(Class_Weights.values()),dtype=torch.float32)
Class_Weights = 1/Class_Weights
Class_Weights = Class_Weights/Class_Weights.sum()

torch.save(Class_Weights,'saves/CWRU_Class_Weights.pt')
torch.save(x_train,'CWRU/presplit/nsaelcn/x_train.pt')
torch.save(x_test,'CWRU/presplit/nsaelcn/x_test.pt')
torch.save(y_train,'CWRU/presplit/nsaelcn/y_train.pt')
torch.save(y_test,'CWRU/presplit/nsaelcn/y_test.pt')