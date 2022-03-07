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
from random import shuffle, randint, Random
from sklearn.metrics import confusion_matrix
import seaborn as sns

FanEnd = [f for f in listdir('dataset/segmented/FanEnd')]
DriveEnd = [f for f in listdir('dataset/segmented/DriveEnd')]

SampleLength = 1200
N = 40 # input dim
J = int(SampleLength/N) # number of segments in a sample
K = 400 # encoding dim
ClassCount = 10
Seed = randint(0,1e6)

class TiedAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(TiedAutoencoder,self).__init__()
        self.Encoder = nn.Linear(input_dim, encoding_dim, bias=False)
    
    def forward(self,x):
        encoded_features = F.relu(self.Encoder(x))
        reconstructed = F.linear(encoded_features, self.Encoder.weight.t())
        return reconstructed, encoded_features

class Classifier(nn.Module):
    def __init__(self, encoding_dim, class_count, W1, W2, J):
        super(Classifier,self).__init__()
        self.W1 = nn.parameter.Parameter(W1, requires_grad=False)
        self.W2 = nn.parameter.Parameter(W2, requires_grad=False)
        self.Linear = nn.Linear(encoding_dim*2, class_count, bias=False)
        self.J = J

    def forward(self,x,y):
        with torch.no_grad():
            local_features = F.relu(F.linear(x,self.W1))
            local_features2 = F.relu(F.linear(y,self.W2))

        # get the mean segment of each J segment
            features = [torch.mean(local_features[range(i,i+self.J)], dim=0) for i in range(0,len(local_features),self.J)] 
            features = torch.tensor([t.numpy() for t in features])
            features2 = [torch.mean(local_features[range(i,i+self.J)], dim=0) for i in range(0,len(local_features),self.J)] 
            features2 = torch.tensor([t.numpy() for t in features])

            features = torch.cat((features,features2),1)

        # apply the linear layer
        logits = self.Linear(features)
        return logits

def AutoencoderLoss(x, model, text='test', verbose=True):
    with torch.no_grad():
        prediction, encoded_features = model(x)
        loss = encoded_features.sum() * 0.25

        if len(x.shape) == 2: #if a list of inputs are given get the average loss
            loss /= len(x)

        loss_fn = nn.MSELoss() #this one takes the average loss automatically
        loss += loss_fn(prediction, x)
        if verbose:
            print(f'{text} loss is: {loss.item()}')
        return loss.item()

def ClassifierLoss(x,x2,y,model,text='test',verbose=True):
    with torch.no_grad():
        logits = model(x,x2)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits,y)
        if verbose:
            print(f'{text} loss is: {loss.item()}')
        return loss.item()

def ClassifierAccuracy(x,x2,y,model,verbose=True):
    with torch.no_grad():
        logits = model(x,x2)
        acc = (torch.argmax(F.log_softmax(logits,dim=1),dim=1) == y).float().mean()
        if verbose:
            print(f'test accuracy is {acc}')
        return acc

def Whiten(x, mean=None, eigenVecs=None, diagonalMat=None ):
    #need to type-check because comparing numpy array with None directly 
    #(array == None) raises an error
    t = type(None)
    if (type(mean) == t) or (type(eigenVecs) == t) or (type(diagonalMat) == t):
        x = np.array(x)
        mean = x.mean(axis=0)
        x -= mean
        cov = np.cov(x, rowvar=False) #unbiased / divided by N-1
        eigenVals, eigenVecs = np.linalg.eig(cov)
        diagonalMat = np.diag(1/((eigenVals)**0.5))

        uncorrelated = np.dot(x,eigenVecs)
        whitened = np.dot(uncorrelated, diagonalMat)
        return whitened, mean, eigenVecs, diagonalMat
    else:
        x = np.array(x)
        x -= mean
        uncorrelated = np.dot(x,eigenVecs)
        whitened = np.dot(uncorrelated, diagonalMat)
        return whitened

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

def ConfusionMat(model,x,x2,y,plot=True):
    with torch.no_grad():
        logits = classifier(x,x2)
        pred = torch.argmax(F.log_softmax(logits,dim=1),dim=1)
        cm = confusion_matrix(y,pred)
    
    if plot:
        plt.figure(figsize=(15,9))
        ax = sns.heatmap(cm, annot=True, fmt='d', linewidths=.5, square=True)
    return cm

#%%
# READ DATA

# READ FAN END
x_mixed_train = []
x_mixed_test = []
y_mixed_train = []
y_mixed_test = []

for name in FanEnd:
    df = pd.read_csv(f'dataset/segmented/FanEnd/{name}')

    # discard segments that does not fill a sample
    data = df.drop(['label'], axis=1).values.tolist()[:len(df)-len(df)%J]
    # get every Jth label as the labels are given with the average of J segments to classifier
    label = df['label'].values.tolist()[:len(data):J]

    # spare first 25% for training
    idx = int(len(label)*0.01)
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

x_mixed_train, mean, eigenVecs, diagonalMat = Whiten(x_mixed_train)
x_mixed_test = Whiten(x_mixed_test, mean, eigenVecs, diagonalMat)

x_mixed_train = torch.tensor(x_mixed_train,dtype=torch.float32)
x_mixed_test = torch.tensor(x_mixed_test,dtype=torch.float32)
y_mixed_train = torch.tensor(y_mixed_train,dtype=torch.long)
y_mixed_test = torch.tensor(y_mixed_test,dtype=torch.long)

torch.save(x_mixed_train,'dataset/presplit/x_mixed_train.pt')
torch.save(x_mixed_test,'dataset/presplit/x_mixed_test.pt')
torch.save(y_mixed_train,'dataset/presplit/y_mixed_train.pt')
torch.save(y_mixed_test,'dataset/presplit/y_mixed_test.pt')


# READ DRIVE END
x_mixed_train2 = []
x_mixed_test2 = []
y_mixed_train2 = []
y_mixed_test2 = []

for name in DriveEnd:
    df = pd.read_csv(f'dataset/segmented/DriveEnd/{name}')

    # discard segments that does not fill a sample
    data = df.drop(['label'], axis=1).values.tolist()[:len(df)-len(df)%J]
    # get every Jth label as the labels are given with the average of J segments to classifier
    label = df['label'].values.tolist()[:len(data):J]

    # spare first 25% for training
    idx = int(len(label)*0.25)
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

    x_mixed_train2 += train
    x_mixed_test2 += test
    y_mixed_train2 += y_train
    y_mixed_test2 += y_test

x_mixed_train2, mean2, eigenVecs2, diagonalMat2 = Whiten(x_mixed_train2)
x_mixed_test2 = Whiten(x_mixed_test2, mean2, eigenVecs2, diagonalMat2)

x_mixed_train2 = torch.tensor(x_mixed_train2,dtype=torch.float32)
x_mixed_test2 = torch.tensor(x_mixed_test2,dtype=torch.float32)
y_mixed_train2 = torch.tensor(y_mixed_train2,dtype=torch.long)
y_mixed_test2 = torch.tensor(y_mixed_test2,dtype=torch.long)

torch.save(x_mixed_train2,'dataset/presplit/x_mixed_train2.pt')
torch.save(x_mixed_test2,'dataset/presplit/x_mixed_test2.pt')
torch.save(y_mixed_train2,'dataset/presplit/y_mixed_train2.pt')
torch.save(y_mixed_test2,'dataset/presplit/y_mixed_test2.pt')

#%%
# AUTOENCODER

MSE_loss_fn = nn.MSELoss()

autoencoder = TiedAutoencoder(N,K)
opt = torch.optim.LBFGS(autoencoder.parameters(), lr=1e-2)

for epoch in range(10):
    def closure():
        opt.zero_grad()
        reconstructed, encoded_features = autoencoder(x_mixed_train)

        loss = encoded_features.sum() * 0.25 / len(x_mixed_train)
        loss += MSE_loss_fn(reconstructed, x_mixed_train)
        loss.backward()
        return loss
    loss = opt.step(closure)

for w in autoencoder.parameters():
    WAutoencoder = w.detach().clone()


# AUTOENCODER2

MSE_loss_fn = nn.MSELoss()

autoencoder = TiedAutoencoder(N,K)
opt = torch.optim.LBFGS(autoencoder.parameters(), lr=1e-2)

for epoch in range(10):
    def closure():
        opt.zero_grad()
        reconstructed, encoded_features = autoencoder(x_mixed_train2)

        loss = encoded_features.sum() * 0.25 / len(x_mixed_train2)
        loss += MSE_loss_fn(reconstructed, x_mixed_train2)
        loss.backward()
        return loss
    loss = opt.step(closure)

for w in autoencoder.parameters():
    WAutoencoder2 = w.detach().clone()

#%%
# CLASSIFIER PARAMETERS

CE_loss_fn = nn.CrossEntropyLoss()
epochs2 = 500

classifier = Classifier(K,ClassCount,WAutoencoder,WAutoencoder2,J)
opt2 = torch.optim.Adam(classifier.parameters(), lr=3e-2)

train_history = []
test_history = []
acc_history = []

# CLASSIFIER TRAINING LOOP

for epoch in range(epochs2):
    opt2.zero_grad()
    logits = classifier(x_mixed_train, x_mixed_train2)
    loss = CE_loss_fn(logits, y_mixed_train)
    loss.backward()
    opt2.step()
    if (epoch+1)%10 == 0:
        print(f"epoch: {len(train_history)+1}/{len(train_history)-(len(train_history)%epochs2)+epochs2}")
        print(f"train loss is: {loss.item()}")
        ClassifierLoss(x_mixed_test,x_mixed_test2,y_mixed_test,classifier)
        ClassifierAccuracy(x_mixed_test,x_mixed_test2,y_mixed_test,classifier)

    train_history.append(loss.item())
    test_history.append(ClassifierLoss(x_mixed_test,x_mixed_test2,y_mixed_test,classifier,verbose=False))
    acc_history.append(ClassifierAccuracy(x_mixed_test,x_mixed_test2,y_mixed_test,classifier,verbose=False))

plt.figure(figsize=(15,10))
plt.plot(range(1,len(train_history)+1),train_history,label='Train Loss')
plt.plot(range(1,len(test_history)+1),test_history,label='Test Loss')
plt.xlim(1,len(train_history))
plt.title('Loss vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy Loss')
plt.legend()
plt.show()

plt.figure(figsize=(15,10))
plt.plot(range(1,len(acc_history)+1),acc_history)
plt.xlim(1,len(acc_history))
plt.title('Accuracy vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

cm = ConfusionMat(classifier, x_mixed_test, x_mixed_test2, y_mixed_test)