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

N = 40
SampleLength = 1200
K = 400 # encoding dim
J = int(SampleLength/N) # number of segments in a sample
AllFiles = [f for f in listdir('dataset/segmented/FanEnd')]
ClassCount = 10

use_presplit = False
update_presplit = True

class TiedAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(TiedAutoencoder,self).__init__()
        self.Encoder = nn.Linear(input_dim, encoding_dim, bias=False)
    
    def forward(self,x):
        encoded_features = F.relu(self.Encoder(x))
        reconstructed = F.linear(encoded_features, self.Encoder.weight.t())
        return reconstructed, encoded_features

class Classifier(nn.Module):
    def __init__(self, encoding_dim, class_count, Wloc, J):
        super(Classifier,self).__init__()
        self.Wloc = nn.parameter.Parameter(Wloc, requires_grad=False)
        self.J = J
        self.Linear = nn.Linear(encoding_dim, class_count, bias=False)
        
    def forward(self,x):
        with torch.no_grad():
            local_features = F.relu(F.linear(x,self.Wloc))
        
        # get the mean segment of each J segment
        features = [torch.mean(local_features[range(i,i+self.J)], dim=0) for i in range(0,len(local_features),self.J)] 
        features = torch.tensor([t.numpy() for t in features])

        # apply the linear layer
        logits = self.Linear(features)
        return logits

def AutoencoderLoss(x, model):
    with torch.no_grad():
        prediction, encoded_features = model(x)
        loss = encoded_features.sum() * 0.25

        loss_fn = nn.MSELoss() #this one takes the average loss automatically
        loss += loss_fn(prediction, x)
        return loss.item()

def ClassifierLoss(x,y,model):
    with torch.no_grad():
        logits = model(x)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits,y)
        return loss.item()

def ClassifierAccuracy(x,y,model):
    with torch.no_grad():
        logits = model(x)
        acc = (torch.argmax(F.log_softmax(logits,dim=1),dim=1) == y).float().mean()
        return acc

def Whiten(x, mean=None, eigenVecs=None, diagonal_mat=None ):
    #need to type-check because comparing numpy array with None directly 
    #(array == None) raises an error
    t = type(None)
    if (type(mean) == t) or (type(eigenVecs) == t) or (type(diagonal_mat) == t):
        x = np.array(x)
        mean = x.mean(axis=0)
        x -= mean
        cov = np.cov(x, rowvar=False) #unbiased / divided by N-1
        eigen_vals, eigenVecs = np.linalg.eig(cov)
        diagonal_mat = np.diag(1/((eigen_vals)**0.5))

        uncorrelated = np.dot(x,eigenVecs)
        whitened = np.dot(uncorrelated, diagonal_mat)
        return whitened, mean, eigenVecs, diagonal_mat
    else:
        x = np.array(x)
        x -= mean
        uncorrelated = np.dot(x,eigenVecs)
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

def ConfusionMat(model,x,y,plot=True):
    with torch.no_grad():
        logits = classifier(x)
        pred = torch.argmax(F.log_softmax(logits,dim=1),dim=1)
        cm = confusion_matrix(y,pred)
    
    if plot:
        plt.figure(figsize=(15,9))
        ax = sns.heatmap(cm, annot=True, fmt='d', linewidths=.5, square=True)
    return cm

#%%
# READ DATA
if not use_presplit:
    # read segmented data (segments fo Nloc) and shuffle samples

    # there is data imbalance between healthy class and others, and also between normal_1 and normal_2,3,4 data
    # check 'all_file_lengths.txt'
    # normal_1 has twice the data of anomalies, normal_2,3,4 has twice the data of normal_1
    # so there are 3.5 times more data for healthy class than others
    
    x_normal_train = []
    x_normal_test = []
    x_mixed_train = []
    x_mixed_test = []
    x_anomaly_train = []
    x_anomaly_test = []
    
    y_mixed_train = []
    y_mixed_test = []
    y_anomaly_train = []
    y_anomaly_test = []

    for name in AllFiles:
        df = pd.read_csv(f'dataset/segmented/FanEnd/{name}')
        data = df.drop(['label'], axis=1).values.tolist()[:len(df)-len(df)%J]
        # get every Jth label as the labels are given with the average of J segments to classifier
        label = df['label'].values.tolist()[:len(data):J]

        idx = int(len(label)*0.1)
        train = data[0:idx*J]
        test = data[idx*J:]
        y_train = label[0:idx]
        y_test = label[idx:]

        train = GroupSamples(train,J)
        temp = list(zip(train,y_train))
        shuffle(temp)
        train, y_train = zip(*temp)
        train = Flatten(train)

        if 'normal' in name: # healthy data to train autoencoder
            x_normal_train += train
            x_normal_test += test

        else: # all anomaly data to check autoencoder error
            x_anomaly_train += train
            x_anomaly_test += test

        # mixed data and label
        x_mixed_train += train
        x_mixed_test += test
        y_mixed_train += y_train
        y_mixed_test += y_test

    x_mixed_train, mean, eigenVecs, diagonal_mat = Whiten(x_mixed_train)
    x_mixed_test = Whiten(x_mixed_test, mean, eigenVecs, diagonal_mat)
    x_normal_train = Whiten(x_normal_train, mean, eigenVecs, diagonal_mat)
    x_normal_test = Whiten(x_normal_test, mean, eigenVecs, diagonal_mat)
    x_anomaly_train = Whiten(x_anomaly_train, mean, eigenVecs, diagonal_mat)
    x_anomaly_test = Whiten(x_anomaly_test, mean, eigenVecs, diagonal_mat)

    x_anomaly_train = torch.tensor(x_anomaly_train,dtype=torch.float32)
    x_anomaly_test = torch.tensor(x_anomaly_test,dtype=torch.float32)
    x_normal_train = torch.tensor(x_normal_train,dtype=torch.float32)
    x_normal_test = torch.tensor(x_normal_test,dtype=torch.float32)
    x_mixed_train = torch.tensor(x_mixed_train,dtype=torch.float32)
    x_mixed_test = torch.tensor(x_mixed_test,dtype=torch.float32)
    
    y_mixed_train = torch.tensor(y_mixed_train,dtype=torch.long)
    y_mixed_test = torch.tensor(y_mixed_test,dtype=torch.long)

    if update_presplit:
        torch.save(x_anomaly_train,'dataset/presplit/x_anomaly_train.pt')
        torch.save(x_anomaly_test,'dataset/presplit/x_anomaly_test.pt')
        torch.save(x_normal_train,'dataset/presplit/x_anomaly_train.pt')
        torch.save(x_normal_test,'dataset/presplit/x_normal_test.pt')
        torch.save(x_mixed_train,'dataset/presplit/x_mixed_train.pt')
        torch.save(x_mixed_test,'dataset/presplit/x_mixed_test.pt')
        torch.save(y_mixed_train,'dataset/presplit/y_mixed_train.pt')
        torch.save(y_mixed_test,'dataset/presplit/y_mixed_test.pt')

else: #use pre-split data
    x_mixed_train = torch.load('dataset/presplit/x_mixed_train.pt')
    x_mixed_test = torch.load('dataset/presplit/x_mixed_test.pt')
    x_normal_train = torch.load('dataset/presplit/x_normal_train.pt')
    x_normal_test = torch.load('dataset/presplit/x_normal_test.pt')
    x_anomaly_train = torch.load('dataset/presplit/x_anomaly_test.pt')
    x_anomaly_test = torch.load('dataset/presplit/x_anomaly_test.pt')
    y_mixed_train = torch.load('dataset/presplit/y_mixed_train.pt')
    y_mixed_test = torch.load('dataset/presplit/y_mixed_test.pt')

#%%
# AUTOENCODER PARAMETERS

MSE_loss_fn = nn.MSELoss()

autoencoder = TiedAutoencoder(N,K)
opt = torch.optim.LBFGS(autoencoder.parameters(), lr=1e-2)

ae_train_history = []
ae_test_history = []
ae_anomaly_history = []

# AUTOENCODER TRAINING LOOP
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
    ae_test_history.append(AutoencoderLoss(x_normal_test,autoencoder,verbose=False))
    ae_anomaly_history.append(AutoencoderLoss(x_anomaly_test,autoencoder,verbose=False))

print(f"train loss is: {loss.item()}")
AutoencoderLoss(x_normal_test,autoencoder,'normal')
AutoencoderLoss(x_anomaly_test,autoencoder,'anomaly')
if update_presplit:
    for w in autoencoder.parameters():
        Wloc = w.detach().clone()
    torch.save(Wloc,'saves/Wloc.pt')

plt.figure(figsize=(15,9))
plt.plot(range(1,len(ae_train_history)+1),ae_train_history,label='train loss')
plt.plot(range(1,len(ae_test_history)+1),ae_test_history,label='normal loss')
plt.plot(range(1,len(ae_anomaly_history)+1),ae_anomaly_history,label='anomaly loss')
plt.xlim(1,len(ae_train_history))
plt.title('Train Loss vs Epochs Passed')
plt.xlabel('Epoch')
plt.ylabel('Reconstruction Error (MSE Loss)')
for xcor in range(10,len(ae_train_history),10):
    plt.axvline(x=xcor,color='r',alpha=0.3)
plt.legend()
plt.show()

#%%
# CLASSIFIER PARAMETERS

Wloc = torch.load('saves/Wloc.pt')

CE_loss_fn = nn.CrossEntropyLoss()
epochs2 = 500

classifier = Classifier(K,ClassCount,Wloc,J)
opt2 = torch.optim.Adam(classifier.parameters(), lr=3e-2)

train_history = []
test_history = []
acc_history = []

# CLASSIFIER TRAINING LOOP

for epoch in range(epochs2):
    opt2.zero_grad()
    logits = classifier(x_mixed_train)
    loss = CE_loss_fn(logits, y_mixed_train)
    loss.backward()
    opt2.step()
    if (epoch+1)%10 == 0:
        print(f"epoch: {len(train_history)+1}/{len(train_history)-(len(train_history)%epochs2)+epochs2}")
        print(f"train loss is: {loss.item()}")
        ClassifierLoss(x_mixed_test,y_mixed_test,classifier)
        ClassifierAccuracy(x_mixed_test,y_mixed_test,classifier)

    train_history.append(loss.item())
    test_history.append(ClassifierLoss(x_mixed_test,y_mixed_test,classifier,verbose=False))
    acc_history.append(ClassifierAccuracy(x_mixed_test,y_mixed_test,classifier,verbose=False))

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

cm = ConfusionMat(classifier, x_mixed_test, y_mixed_test)