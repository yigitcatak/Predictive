import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import listdir
from random import shuffle, randint, Random, sample
import scipy.io
import time

# NETWORKS
class NSAELCN(nn.Module):
    def __init__(self,input_dim,encoding_dim):
        super(NSAELCN,self).__init__()
        self.Encoder = nn.Linear(input_dim, encoding_dim, bias=False)

    def forward(self,x):
        encoded_features = F.relu(self.Encoder(x))
        reconstructed = F.linear(encoded_features, self.Encoder.weight.t())
        return encoded_features, reconstructed

class Arxiv(nn.Module):
    def __init__(self,input_dim,encoding_dim,channel_count):
        super(Arxiv,self).__init__()
        self.drp1 = nn.Dropout(p = 0.1)
        self.flat = nn.Flatten()
        self.relu = nn.ReLU()
        self.input_dim = input_dim
        # ENCODER
        self.conv1 = nn.Conv2d(1, 40, (channel_count,1))
        self.conv2 = nn.Conv2d(40,40,(1,2),2)
        self.conv3 = nn.Conv2d(40,40, (1,11),padding=(0,5))
        # LINEAR
        self.lin1 = nn.Linear((input_dim//2)*40,(input_dim//4)*40)
        self.lin2 = nn.Linear((input_dim//4)*40,encoding_dim)
        self.lin3 = nn.Linear(encoding_dim,(input_dim//4)*40)
        self.lin4 = nn.Linear((input_dim//4)*40,(input_dim//2)*40)
        # DECODER
        self.tconv1 = nn.ConvTranspose2d(40, 1, (channel_count,1))
        self.tconv2 = nn.ConvTranspose2d(40,40,(1,2),2) 
        self.tconv3 = nn.ConvTranspose2d(40,40,(1,11),padding=(0,5))

    def forward(self,x):
        encoder_out = self.conv3(self.relu(self.drp1(self.relu(self.conv2(self.drp1(self.conv1(x)))))))
        encoder_out = self.lin1(self.flat(encoder_out))
        bottleneck = self.lin2(encoder_out)
        decoder_out = torch.reshape(self.lin4(self.lin3(bottleneck)),(-1,40,1,(self.input_dim//2)))
        decoder_out = self.tconv1(self.drp1(self.relu(self.tconv2(self.drp1(self.relu(self.tconv3(decoder_out)))))))
        return bottleneck, decoder_out

class Classifier(nn.Module):
    def __init__(self,encoding_dim,class_count,segment_count,MLP=False):
        super(Classifier,self).__init__()
        self.lin1 = nn.Linear(encoding_dim, encoding_dim, bias=False)
        self.lin2 = nn.Linear(encoding_dim, class_count, bias=False)
        self.relu = nn.ReLU()
        self.segment_count = segment_count
        self.isMLP = MLP            

    def forward(self,x):
        # get the mean segment of each J segment
        features = torch.stack([torch.mean(x[range(i,i+self.segment_count)], dim=0) for i in range(0,len(x),self.segment_count)])

        logits = self.lin2(self.relu(self.lin1(features))) if self.isMLP else self.lin2(features)
        return logits

# EVALUATIONS
def AutoencoderLoss(x,ae,isBatched=False):
    with torch.no_grad():
        loss = 0
        MSE = nn.MSELoss()
        if isBatched:
            for x_batch in x:
                temp_loss = 0
                encoded, x_hat = ae(x_batch)
                for e in encoded:
                    temp_loss += e.norm(1)
                temp_loss = temp_loss*0.25/(len(x_batch)*len(x))
                temp_loss += MSE(x_hat, x_batch)/len(x)
                loss += temp_loss
        else:
            encoded, x_hat = ae(x)
            for e in encoded:
                loss += e.norm(1)
            loss = loss*0.25/(len(x))
            loss += MSE(x_hat, x)

    return loss.item()

def ClassifierEvaluate(x,y,ae,cl,isBatched=False):
    with torch.no_grad():
        acc = 0
        loss = 0
        CrossEntropy = nn.CrossEntropyLoss()
        if isBatched:
            for x_batch,y_batch in zip(x,y):
                x_batch, _ = ae(x_batch)
                logits = cl(x_batch)
                loss += CrossEntropy(logits,y_batch)/len(x)
                acc += (torch.argmax(F.log_softmax(logits,dim=1),dim=1) == y_batch).float().mean()
            acc /= len(x)
        else:
            x, _ = ae(x)
            logits = cl(x)
            loss += CrossEntropy(logits,y)
            acc += (torch.argmax(F.log_softmax(logits,dim=1),dim=1) == y).float().mean()
    return loss.item(), acc.item()

def ClassifierAccuracy(x,y,ae,cl,isBatched=False):
    with torch.no_grad():
        acc = 0
        if isBatched:
            for x_batch,y_batch in zip(x,y):
                x_batch, _ = ae(x_batch)
                logits = cl(x_batch)
                acc += (torch.argmax(F.log_softmax(logits,dim=1),dim=1) == y_batch).float().mean()
            acc /= len(x)
        else:
            x, _ = ae(x)
            logits = cl(x)
            acc += (torch.argmax(F.log_softmax(logits,dim=1),dim=1) == y).float().mean()
    return acc.item()

# OTHER
def Batch(l, J): #Also used to group segments as a sample
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

def RandomCombination(l, r):
    n = len(l)
    indices = sorted(sample(range(n), r))
    return list(l[i] for i in indices)

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

def PlotResults(train_results,test_results=None,label=None,ylabel=None,isSave=False,savename='figure'):
    plt.figure()
    plt.plot(range(1,len(train_results)+1),train_results,label='Train' if label is None else f'Train {label}')
    if test_results is not None:
        plt.plot(range(1,len(test_results)+1),test_results,label='Test' if label is None else f'Test {label}')
    plt.xlim(1,len(train_results))
    if label is not None:
        plt.title(f'{label} vs Epochs')
    plt.xlabel('Epoch')
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.legend()
    if isSave:
        plt.savefig(f'{savename}.png',bbox_inches='tight')
    else:
        plt.show()

    