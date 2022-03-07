#%%
#DEFINITIONS
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy import linalg
from os import listdir
from os.path import isfile, join
from random import shuffle
from dataset_segment import N #input dim

AllDataFiles = [f for f in listdir('dataset/segmented')]

SampleLength = 1200
J = int(SampleLength/N) # number of segments in a sample
K = 400 # encoding dim
ClassCount = 10

class TiedAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(TiedAutoencoder,self).__init__()
        self.Encoder = nn.Linear(input_dim, encoding_dim, bias=False)
    
    def forward(self,x):
        encoded_features = F.relu(self.Encoder(x))
        reconstructed = F.linear(encoded_features, self.Encoder.weight.t())
        return reconstructed, encoded_features

class Classifier(nn.Module):
    def __init__(self, encoding_dim, class_count, W1, J, W2=None):
        super(Classifier,self).__init__()
        self.W1 = nn.parameter.Parameter(W1, requires_grad=False)
        self.J = J
        if W2 is None:
            self.Train = True
            self.W2 = nn.Linear(encoding_dim, class_count, bias=False)
        else:
            self.Train = False
            self.W2 = nn.parameter.Parameter(W2, requires_grad=False)

    def forward(self,x):
        with torch.no_grad():
            local_features = F.relu(F.linear(x,self.W1))

        # get the mean segment of each J segment
        features = [torch.mean(local_features[range(i,i+self.J)], dim=0) for i in range(0,len(local_features),self.J)] 
        features = torch.tensor([t.numpy() for t in features])

        # apply the linear layer
        if self.Train:
            logits = self.W2(features)
        else:
            logits = F.linear(features,self.W2)
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

def ClassifierLoss(x,y,model,text='test',verbose=True):
    with torch.no_grad():
        logits = model(x)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits,y)
        if verbose:
            print(f'{text} loss is: {loss.item()}')
        return loss.item()

def ClassifierAccuracy(x,y,model,verbose=True):
    with torch.no_grad():
        logits = model(x)
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

#%%
#READ FILES

x_mixed_train = []
x_mixed_test = []
y_mixed_train = []
y_mixed_test = []

for name in AllDataFiles:
    df = pd.read_csv(f'dataset/segmented/{name}')

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
    shuffle(temp)
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

pd.DataFrame(eigenVecs).to_csv('saves/eigenVecs.csv',index=False)
pd.DataFrame(diagonalMat).to_csv('saves/diagonalMat.csv',index=False)
pd.DataFrame(mean).to_csv('saves/mean.csv',index=False)
#%%
# AUTOENCODER

MSE_loss_fn = nn.MSELoss()

autoencoder = TiedAutoencoder(N,K)
opt = torch.optim.LBFGS(autoencoder.parameters(), lr=1e-2)

ae_train_history = []
ae_test_history = []

for epoch in range(10):
    def closure():
        opt.zero_grad()
        reconstructed, encoded_features = autoencoder(x_mixed_train)

        loss = encoded_features.sum() * 0.25 / len(x_mixed_train)
        loss += MSE_loss_fn(reconstructed, x_mixed_train)
        loss.backward()
        return loss
    loss = opt.step(closure)
    ae_train_history.append(loss.item())
    ae_test_history.append(AutoencoderLoss(x_mixed_test, autoencoder,verbose=False))

for w in autoencoder.parameters():
    WAutoencoder = w.detach().clone()

pd.DataFrame({'train loss': ae_train_history,'test loss': ae_test_history}).to_csv('history/autoencoder_history.csv',index=False)
#%%
# CLASSIFIER

CE_loss_fn = nn.CrossEntropyLoss()
classifier = Classifier(K,ClassCount,WAutoencoder,J)
opt2 = torch.optim.Adam(classifier.parameters(), lr=3e-2)
epochs = 1000

train_history = []
test_history = []
acc_history = []

for epoch in range(epochs):
    opt2.zero_grad()
    logits = classifier(x_mixed_train)
    loss = CE_loss_fn(logits, y_mixed_train)
    loss.backward()
    opt2.step()
    train_history.append(loss.item())
    test_history.append(ClassifierLoss(x_mixed_test,y_mixed_test,classifier,verbose=False))
    acc_history.append(ClassifierAccuracy(x_mixed_test,y_mixed_test,classifier,verbose=False))

acc_history = [float(i) for i in acc_history]
pd.DataFrame({'train loss':train_history,'test loss':test_history,'test accuracy':acc_history}).to_csv('history/classifier_history.csv',index=False)

for w in classifier.parameters():
    W = w.detach().clone()
    torch.save(W,f'saves/WClassifier{W.shape[1]}x{W.shape[0]}.pt')
