#%%
# DEFINITIONS
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
from random import shuffle
from sklearn.metrics import confusion_matrix
import seaborn as sns

N = 30 # input dim
Sample_Length = 1200
K = 40 # encoding dim
J = int(Sample_Length/N) # number of segments in a sample
Class_Count = 10

class TiedAutoencoder(nn.Module):
    def __init__(self):
        super(TiedAutoencoder,self).__init__()
        self.Encoder = nn.Linear(N, K, bias=False)
    
    def forward(self,x):
        encoded_features = F.relu(self.Encoder(x))
        reconstructed = F.linear(encoded_features, self.Encoder.weight.t())
        return encoded_features, reconstructed

class Classifier(nn.Module):
    def __init__(self,Wloc=None):
        super(Classifier,self).__init__()
        self.Wloc = nn.Linear(N,K)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(K, Class_Count, bias=False)
        self.lin1 = nn.Linear(K, K, bias=False)

        if Wloc is not None:
            self.Wloc.weight = nn.Parameter(Wloc, requires_grad=False)
        
    def forward(self,x):
        with torch.no_grad():
            local_features = self.relu(self.Wloc(x))

        # get the mean segment of each J segment
        features = torch.stack([torch.mean(local_features[i:i+J], dim=0) for i in range(0,len(local_features),J)])
        logits = self.lin2(self.relu(self.lin1(features)))
        # logits = self.lin2(features)
        return logits

def AutoencoderLoss(x, model):
    with torch.no_grad():
        MSE = nn.MSELoss()
        encoded, x_hat = model(x)

        loss = 0
        for e in encoded:
            loss += e.norm(1)
        loss = loss*0.25/(len(x)*K)
        loss += MSE(x_hat, x)
    return loss

def ClassifierLoss(x,y,model):
    with torch.no_grad():
        CrossEntropy = nn.CrossEntropyLoss()
        logits = model(x)

        loss = CrossEntropy(logits,y)
    return loss.item()

def ClassifierAccuracy(x,y,model):
    with torch.no_grad():
        logits = model(x)
        acc = (torch.argmax(F.log_softmax(logits,dim=1),dim=1) == y).float().mean()
    return acc

def ClassifierEvaluate(x,y,model):
    with torch.no_grad():
        Crossentropy = nn.CrossEntropyLoss()
        logits = model(x)
        loss = CrossEntropy(logits,y)
        acc = (torch.argmax(F.log_softmax(logits,dim=1),dim=1) == y).float().mean()
    return loss, acc

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

def ConfusionMat(x,y,model,plot=True):
    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(F.log_softmax(logits,dim=1),dim=1)
        cm = confusion_matrix(y,pred)
    
    if plot:
        plt.figure(figsize=(15,9))
        ax = sns.heatmap(cm, annot=True, fmt='d', linewidths=.5, square=True)
    return cm

# READ DATA
x_train = torch.load('CWRU/presplit/singlechannel/x_train.pt')
y_train = torch.load('CWRU/presplit/singlechannel/y_train.pt')
x_test = torch.load('CWRU/presplit/singlechannel/x_test.pt')
y_test = torch.load('CWRU/presplit/singlechannel/y_test.pt')
weights = torch.load('saves/CWRU_Class_Weights.pt')

# Cuda
device = torch.device("cuda")
x_train = x_train.to(device)
x_test = x_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)
weights = weights.to(device)

# AUTOENCODER

MSE = nn.MSELoss()

ae = TiedAutoencoder().cuda()
ae_opt = torch.optim.Adam(ae.parameters(), lr=3e-2)

ae_train_history = []
ae_test_history = []

# AUTOENCODER TRAINING LOOP
for epoch in range(10):
    def closure():
        ae_opt.zero_grad()
        encoded_features, reconstructed = ae(x_train)
        loss = 0
        for x in encoded_features:
            loss += x.norm(1)
        loss = loss*0.25/(len(x_train)*K)
        loss += MSE(reconstructed, x_train)
        loss.backward()
        return loss
    ae_opt.step(closure)
    
    ae.eval()
    ae_train_history.append(AutoencoderLoss(x_train,ae))
    ae_test_history.append(AutoencoderLoss(x_test,ae))
    ae.train()
    print(f'epoch: {epoch+1}/10')
    print(f'train loss: {ae_train_history[-1]}')
    print(f'test loss: {ae_test_history[-1]}')

for w in ae.parameters():
    Wloc = w.detach().clone()
    # torch.save(Wloc,'saves/Wloc.pt')

plt.figure(figsize=(15,9))
plt.plot(range(1,len(ae_train_history)+1),ae_train_history,label='train loss')
plt.plot(range(1,len(ae_test_history)+1),ae_test_history,label='test loss')
plt.xlim(1,len(ae_train_history))
plt.title('Train Loss vs Epochs Passed')
plt.xlabel('Epoch')
plt.ylabel('MSE + L1 Norm')
plt.legend()
plt.show()

# CLASSIFIER

CrossEntropy = nn.CrossEntropyLoss(weight=weights)

cl = Classifier(Wloc).cuda()
cl_opt = torch.optim.Adam(cl.parameters(), lr=3e-2)
cl_epochs = 500

cl_train_loss = []
cl_test_loss = []
cl_train_accuracy = []
cl_test_accuracy = []

for epoch in range(cl_epochs):
    print(f"epoch: {epoch+1}/{cl_epochs}")
    cl_opt.zero_grad()
    logits = cl(x_train)
    loss = CrossEntropy(logits, y_train)
    loss.backward()
    cl_opt.step()

    cl.eval()
    train_loss,train_accuracy = ClassifierEvaluate(x_train,y_train,cl)
    test_loss,test_accuracy = ClassifierEvaluate(x_test,y_test,cl)
    cl.train()

    cl_train_loss.append(train_loss)
    cl_test_loss.append(test_loss) 
    cl_train_accuracy.append(train_accuracy)
    cl_test_accuracy.append(test_accuracy)

    print(f"train loss is: {train_loss}")
    print(f"test loss is: {test_loss}")
    print(f"train accuracy is: {train_accuracy}")
    print(f"test accuracy is: {test_accuracy}")

plt.figure(figsize=(15,9))
plt.plot(range(1,len(cl_train_loss)+1),cl_train_loss,label='Train Loss')
plt.plot(range(1,len(cl_test_loss)+1),cl_test_loss,label='Test Loss')
plt.xlim(1,len(cl_train_loss))
plt.title('Loss vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy Loss')
plt.legend()
plt.show()

plt.figure(figsize=(15,9))
plt.plot(range(1,len(cl_train_accuracy)+1),cl_train_accuracy,label='Train Accuracy')
plt.plot(range(1,len(cl_test_accuracy)+1),cl_test_accuracy,label='Test Accuracy')
plt.xlim(1,len(cl_train_accuracy))
plt.title('Accuracy vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# cm = ConfusionMat(x_test, y_test, cl)