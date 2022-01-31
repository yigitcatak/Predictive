import torch
import pandas as pd
import numpy as np
from scipy import linalg
from os import listdir
from random import shuffle, randint, Random

Sample_Length = 6400
N = 160
J = int(Sample_Length/N)
All_Files = [f for f in listdir('datasets/Paderborn/segmented')]
Class_Weights = dict(zip(list(range(3)),list(0 for i in range(3))))
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

train_names = ['K002','KA01','KA05','KA07','KI01','KI05','KI07']

for name in All_Files:
    df = pd.read_csv(f'datasets/Paderborn/segmented/{name}')
    name = name[-11:-7]

    # discard segments that does not fill a sample
    data = df.drop(['label'], axis=1).values.tolist()[:len(df)-len(df)%J]
    # get every Jth label as the labels are given with the average of J segments to classifier
    label = df['label'].values.tolist()[:len(data):J]

    if name in train_names:
        train = data
        y_train = label
        train = GroupSamples(train,J)
        train = list(zip(train,y_train))
        x_mixed_train += train

        Class_Weights[label[0]] += len(train)

    else:
        x_mixed_test += data
        y_mixed_test += label

Random(Seed).shuffle(x_mixed_train)
x_mixed_train, y_mixed_train = zip(*x_mixed_train)
x_mixed_train = Flatten(x_mixed_train)

x_mixed_train, mean, eigen_vecs, diagonal_mat = Whiten(x_mixed_train)
x_mixed_test = Whiten(x_mixed_test, mean, eigen_vecs, diagonal_mat)

x_mixed_train = torch.tensor(x_mixed_train,dtype=torch.float32)
x_mixed_test = torch.tensor(x_mixed_test,dtype=torch.float32)
y_train = torch.tensor(y_mixed_train,dtype=torch.long)
y_test = torch.tensor(y_mixed_test,dtype=torch.long)

x_train = torch.unsqueeze(torch.unsqueeze(x_mixed_train,dim=1),dim=1)
x_test = torch.unsqueeze(torch.unsqueeze(x_mixed_test,dim=1),dim=1)

Class_Weights = torch.tensor(list(Class_Weights.values()),dtype=torch.float32)
Class_Weights = 1/Class_Weights
Class_Weights = Class_Weights/Class_Weights.sum()

torch.save(Class_Weights,'saves/Paderborn_Class_Weights.pt')
torch.save(x_train,'datasets/Paderborn/presplit/arxiv_singlechannel/x_train.pt')
torch.save(x_test,'datasets/Paderborn/presplit/arxiv_singlechannel/x_test.pt')
torch.save(y_train,'datasets/Paderborn/presplit/arxiv_singlechannel/y_train.pt')
torch.save(y_test,'datasets/Paderborn/presplit/arxiv_singlechannel/y_test.pt')