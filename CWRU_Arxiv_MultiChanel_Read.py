# READ FAN END
import torch
import pandas as pd
import numpy as np
from scipy import linalg
from os import listdir
from random import shuffle, randint, Random


Sample_Length = 1200
N = 30
J = int(Sample_Length/N)
FanEnd = [f for f in listdir('CWRU/segmented/fanend')]
DriveEnd = [f for f in listdir('CWRU/segmented/driveend')]

Seed = randint(0,1e6)

def GroupSamples(l, J): #Group J segments together
    def inner():
        for i in range(0, len(l)-(len(l)%J), J):
            yield l[i:i + J]
    return list(inner())

def Flatten(l): #Unpack Grouped segments
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

x_mixed_train = []
x_mixed_test = []
y_mixed_train = []
y_mixed_test = []

for name in FanEnd:
    df = pd.read_csv(f'CWRU/segmented/fanend/{name}')

    # discard segments that does not fill a sample
    data = df.drop(['label'], axis=1).values.tolist()[:len(df)-len(df)%J]
    # get every Jth label as the labels are given with the average of J segments to classifier
    label = df['label'].values.tolist()[:len(data):J]

    # spare first 25% for training
    idx = int(len(label)*0.1)
    train = data[0:idx*J]
    test = data[idx*J:]
    y_train = label[0:idx]
    y_test = label[idx:]

    # group segments so that sample integrity is kept during the shuffling
    train = GroupSamples(train,J)
    temp = list(zip(train,y_train))
    Random(Seed).shuffle(temp)
    train, y_train = zip(*temp)
    train = Flatten(train)

    x_mixed_train += train
    x_mixed_test += test
    y_mixed_train += y_train
    y_mixed_test += y_test

x_mixed_train, mean, eigen_vecs, diagonal_mat = Whiten(x_mixed_train)
x_mixed_test = Whiten(x_mixed_test, mean, eigen_vecs, diagonal_mat)

x_mixed_train = torch.tensor(x_mixed_train,dtype=torch.float32)
x_mixed_test = torch.tensor(x_mixed_test,dtype=torch.float32)
y_train = torch.tensor(y_mixed_train,dtype=torch.long)
y_test = torch.tensor(y_mixed_test,dtype=torch.long)

# READ DRIVE END
x_mixed_train2 = []
x_mixed_test2 = []

for name in DriveEnd:
    df = pd.read_csv(f'CWRU/segmented/driveend/{name}')

    # discard segments that does not fill a sample
    data = df.drop(['label'], axis=1).values.tolist()[:len(df)-len(df)%J]
    # get every Jth label as the labels are given with the average of J segments to classifier
    label = df['label'].values.tolist()[:len(data):J]

    # spare first 25% for training
    idx = int(len(label)*0.1)
    train = data[0:idx*J]
    test = data[idx*J:]

    # group segments so that sample integrity is kept during the shuffling
    train = GroupSamples(train,J)
    Random(Seed).shuffle(temp)
    train = Flatten(train)

    x_mixed_train2 += train
    x_mixed_test2 += test

x_mixed_train2, mean2, eigen_vecs2, diagonal_mat2 = Whiten(x_mixed_train2)
x_mixed_test2 = Whiten(x_mixed_test2, mean2, eigen_vecs2, diagonal_mat2)

x_mixed_train2 = torch.tensor(x_mixed_train2,dtype=torch.float32)
x_mixed_test2 = torch.tensor(x_mixed_test2,dtype=torch.float32)

x_train = torch.cat([torch.unsqueeze(x_mixed_train,dim=1),torch.unsqueeze(x_mixed_train2,dim=1)],dim=1)
x_test = torch.cat([torch.unsqueeze(x_mixed_test,dim=1),torch.unsqueeze(x_mixed_test2,dim=1)],dim=1)

x_train = torch.unsqueeze(x_train,dim=1)
x_test = torch.unsqueeze(x_test,dim=1)

torch.save(x_train,'CWRU/presplit/arxiv_multichannel/x_train.pt')
torch.save(x_test,'CWRU/presplit/arxiv_multichannel/x_test.pt')
torch.save(y_train,'CWRU/presplit/arxiv_multichannel/y_train.pt')
torch.save(y_test,'CWRU/presplit/arxiv_multichannel/y_test.pt')