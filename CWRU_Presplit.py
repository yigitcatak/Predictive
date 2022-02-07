# READ FAN END
from msilib.schema import Class
import torch
import pandas as pd
import numpy as np
from scipy import linalg
from os import listdir
from random import shuffle, randint, Random

Sample_Length = 1200 #0.1 sec
J = 40
N = int(Sample_Length/J)

FanEnd = [f for f in listdir('datasets/CWRU/segmented/fan_end')]
DriveEnd = [f for f in listdir('datasets/CWRU/segmented/drive_end')]
Class_Weights = dict(zip(list(range(10)),list(0 for i in range(10))))
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
x_mixed_train2 = []
x_mixed_test2 = []

for name in FanEnd:
    df = pd.read_csv(f'datasets/CWRU/segmented/fan_end/{name}')

    # discard segments that does not fill a sample
    data = df.drop(['label'], axis=1).values.tolist()[:len(df)-len(df)%J]
    # get every Jth label as the labels are given with the average of J segments to classifier
    label = df['label'].values.tolist()[:len(data):J]

    # spare first 10% for training
    idx = int(len(label)*0.1)
    train = data[0:idx*J]
    test = data[idx*J:]
    y_train = label[0:idx]
    y_test = label[idx:]
    Class_Weights[y_train[0]] += len(y_train)

    # group segments so that sample integrity is kept during the shuffling
    train = GroupSamples(train,J)
    train = list(zip(train,y_train))

    x_mixed_train += train
    x_mixed_test += test
    y_mixed_test += y_test

# READ DRIVE END

for name in DriveEnd:
    df = pd.read_csv(f'datasets/CWRU/segmented/drive_end/{name}')

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

    x_mixed_train2 += train
    x_mixed_test2 += test

Random(Seed).shuffle(x_mixed_train2)
x_mixed_train2 = Flatten(x_mixed_train2)

Random(Seed).shuffle(x_mixed_train)
x_mixed_train, y_mixed_train = zip(*x_mixed_train)
x_mixed_train = Flatten(x_mixed_train)

x_mixed_train, mean, eigen_vecs, diagonal_mat = Whiten(x_mixed_train)
x_mixed_test = Whiten(x_mixed_test, mean, eigen_vecs, diagonal_mat)

x_mixed_train2, mean2, eigen_vecs2, diagonal_mat2 = Whiten(x_mixed_train2)
x_mixed_test2 = Whiten(x_mixed_test2, mean2, eigen_vecs2, diagonal_mat2)

x_mixed_train = torch.tensor(x_mixed_train,dtype=torch.float32)
x_mixed_test = torch.tensor(x_mixed_test,dtype=torch.float32)
x_mixed_train2 = torch.tensor(x_mixed_train2,dtype=torch.float32)
x_mixed_test2 = torch.tensor(x_mixed_test2,dtype=torch.float32)
y_train = torch.tensor(y_mixed_train,dtype=torch.long)
y_test = torch.tensor(y_mixed_test,dtype=torch.long)

Class_Weights = torch.tensor(list(Class_Weights.values()),dtype=torch.float32)
Class_Weights = 1/Class_Weights
Class_Weights = Class_Weights/Class_Weights.sum()

# x_train = torch.cat([torch.unsqueeze(x_mixed_train,dim=1),torch.unsqueeze(x_mixed_train2,dim=1)],dim=1)
# x_test = torch.cat([torch.unsqueeze(x_mixed_test,dim=1),torch.unsqueeze(x_mixed_test2,dim=1)],dim=1)

# x_train = torch.unsqueeze(x_train,dim=1)
# x_test = torch.unsqueeze(x_test,dim=1)

torch.save(Class_Weights,'datasets/CWRU/presplit/class_weights.pt')
torch.save(x_mixed_train,'datasets/CWRU/presplit/x_train_fan_end.pt')
torch.save(x_mixed_test,'datasets/CWRU/presplit/x_test_fan_end.pt')
torch.save(x_mixed_train2,'datasets/CWRU/presplit/x_train_drive_end.pt')
torch.save(x_mixed_test2,'datasets/CWRU/presplit/x_test_drive_end.pt')
torch.save(y_train,'datasets/CWRU/presplit/y_train.pt')
torch.save(y_test,'datasets/CWRU/presplit/y_test.pt')