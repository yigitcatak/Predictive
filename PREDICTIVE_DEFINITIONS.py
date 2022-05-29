import time
from os import listdir
from random import Random, randint, sample, shuffle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix


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
        self.drp = nn.Dropout(p = 0.1)
        self.drp_bottleneck = nn.Dropout(p = 0.9)
        self.drp_linear = nn.Dropout(p = 0.0)
        self.flat = nn.Flatten()
        self.relu = nn.ReLU()
        self.input_dim = input_dim
        # ENCODER
        self.conv1 = nn.Conv2d(1, 40, (channel_count,1))
        self.conv2 = nn.Conv2d(40,40,(1,2),2)
        self.conv3 = nn.Conv2d(40,40, (1,11),padding=(0,5))
        # LINEAR
        self.lin1 = nn.Linear(input_dim*20,input_dim*10)
        self.lin2 = nn.Linear(input_dim*10,encoding_dim)
        self.lin3 = nn.Linear(encoding_dim,input_dim*10)
        self.lin4 = nn.Linear(input_dim*10,input_dim*20)
        # DECODER
        self.tconv1 = nn.ConvTranspose2d(40, 1, (channel_count,1))
        self.tconv2 = nn.ConvTranspose2d(40,40,(1,2),2) 
        self.tconv3 = nn.ConvTranspose2d(40,40,(1,11),padding=(0,5))

    def forward(self,x):
        encoder_out = self.conv3(self.relu(self.drp(self.relu(self.conv2(self.drp(self.conv1(x)))))))
        encoder_out = self.flat(encoder_out)
        bottleneck = self.drp_bottleneck(self.lin2(self.drp_linear(self.lin1(encoder_out))))
        decoder_out = self.lin4(self.drp_linear((self.lin3(bottleneck))))
        decoder_out = torch.reshape(decoder_out,(-1,40,1,(self.input_dim//2)))
        decoder_out = self.tconv1(self.drp(self.relu(self.tconv2(self.drp(self.relu(self.tconv3(decoder_out)))))))
        return bottleneck, decoder_out

class Classifier(nn.Module):
    def __init__(self,encoding_dim,class_count,segment_count,MLP=False):
        super(Classifier,self).__init__()
        self.lin1 = nn.Linear(encoding_dim, encoding_dim, bias=False)
        self.lin2 = nn.Linear(encoding_dim, class_count, bias=False)
        self.relu = nn.ReLU()
        self.segment_count = segment_count
        self.isMLP = MLP    
        self.drp = nn.Dropout(0.0)
        self.drp = nn.Dropout(0.0)     

    def forward(self,x):
        # get the mean segment of each J segment
        features = torch.stack([torch.mean(x[range(i,i+self.segment_count)], dim=0) for i in range(0,len(x),self.segment_count)])
        logits = self.lin2(self.drp(self.relu(self.lin1(self.drp(features))))) if self.isMLP else self.lin2(features)
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
def Settings(dataset):
    if dataset == 'CWRU':
        Sample_Length = 1200
        J = 30
    elif dataset == 'Paderborn':
        Sample_Length = 6400
        J = 30
    N = (Sample_Length//J) - ((Sample_Length//J)%4)
    return N,J

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

def Whiten(x, mean=None, whitening_mat=None ):
    #need to type-check because comparing numpy array with None directly 
    #(array == None) raises an error
    t = type(None)
    if (type(mean) == t) or (type(whitening_mat) == t):
        x = np.array(x)
        mean = x.mean(axis=0)
        x -= mean
        cov = np.cov(x, rowvar=False) #unbiased / divided by N-1
        eigenVals, eigen_vecs = np.linalg.eig(cov)
        diagonal_mat = np.diag(1/((eigenVals)**0.5))
        
        # uncorrelated = np.dot(x,eigen_vecs)
        # whitened = np.dot(uncorrelated, diagonal_mat)

        whitening_mat = eigen_vecs @ diagonal_mat
        whitened = x @ whitening_mat
        return whitened, mean, whitening_mat
    else:
        x = np.array(x)
        x -= mean
        whitened = x @ whitening_mat
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
        plt.savefig(f'graphs/{savename}.png',bbox_inches='tight')
    else:
        plt.show()

def ConfusionMat(x,y,ae,cl,class_count,isBatched=False,plot=True):
    with torch.no_grad():
        cm = np.zeros((class_count,class_count),dtype=np.int64)
        if isBatched:
            for x_batch,y_batch in zip(x,y):
                x_batch, _ = ae(x_batch)
                logits = cl(x_batch)
                pred = torch.argmax(F.log_softmax(logits,dim=1),dim=1)
                pred = pred.to('cpu')
                y_batch = y_batch.to('cpu')
                cm += confusion_matrix(y_batch,pred,labels=[i for i in range(class_count)])
        else:
            x, _ = ae(x)
            logits = cl(x)
            pred = torch.argmax(F.log_softmax(logits,dim=1),dim=1)
            pred = pred.to('cpu')
            y = y.to('cpu')
            cm += confusion_matrix(y,pred,labels=[i for i in range(class_count)])
    
    if plot:
        plt.figure()
        ax = sns.heatmap(cm, annot=True, fmt='d', linewidths=.5, square=True)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
    return cm    
