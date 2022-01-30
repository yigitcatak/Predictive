#%%
# Definitions
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy import linalg
from os import listdir
from os.path import isfile, join
from random import shuffle, randint, Random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

Class_Count = 10
Subband_Count = 1
Channel_Count = 1
Sample_Length = 1200
N = 40
K = 20
J = int(Sample_Length/N)

class Arxiv(nn.Module):
    def __init__(self):
        super(Arxiv,self).__init__()

        self.ChannelNumber = Subband_Count*Class_Count #10
        self.LinSize = int(N*self.ChannelNumber/2) #20
        self.LinSize2 = int(self.LinSize/3)+(10-int(self.LinSize/3)%10) #70

        self.drp1 = nn.Dropout(p = 0.1)
        self.flat = nn.Flatten()
        self.relu = nn.ReLU()

        # ENCODER
        self.conv1 = nn.Conv2d(1, 40, (Channel_Count,1))
        self.conv2 = nn.Conv2d(40,40,(1,Channel_Count),2)
        self.conv3 = nn.Conv2d(40,40, (1,11),padding=(0,5))

        # LINEAR
        self.lin1 = nn.Linear(int(N/2)*40,800)
        self.lin2 = nn.Linear(800,K)
        self.lin3 = nn.Linear(K,800)
        self.lin4 = nn.Linear(800,int(N/2)*40)

        # DECODER
        self.tconv1 = nn.ConvTranspose2d(40, 1, (Channel_Count,1))
        self.tconv2 = nn.ConvTranspose2d(40,40,(1,Channel_Count),2) if Channel_Count == 2 else nn.ConvTranspose2d(40,40,(1,Channel_Count),2,output_padding=(0,1)) 
        self.tconv3 = nn.ConvTranspose2d(40,40,(1,11),padding=(0,5))

    def forward(self,x):
        encoder_out = self.conv3(self.relu(self.drp1(self.relu(self.conv2(self.drp1(self.conv1(x)))))))
        encoder_out = self.lin1(self.flat(encoder_out))
        bottleneck = self.lin2(encoder_out)
        decoder_out = torch.reshape(self.lin4(self.lin3(bottleneck)),(-1,40,1,int(N/2)))
        decoder_out = self.tconv1(self.drp1(self.relu(self.tconv2(self.drp1(self.relu(self.tconv3(decoder_out)))))))

        return bottleneck, decoder_out

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()
        self.lin1 = nn.Linear(K, K, bias=False)
        self.lin2 = nn.Linear(K, Class_Count, bias=False)
        self.relu = nn.ReLU()


    def forward(self,x):
        # get the mean segment of each J segment
        features = torch.stack([torch.mean(x[range(i,i+J)], dim=0) for i in range(0,len(x),J)]) 
        logits = self.lin2(self.relu(self.lin1(features)))
        return logits

def AutoencoderLoss(x, model):
    with torch.no_grad():
        loss = 0
        MSE = nn.MSELoss()
        encoded, x_hat = model(x)
        for e in encoded:
            loss += e.norm(1)
        loss = loss*0.25/len(x)
        loss += MSE(x_hat, x)

    return loss.item()

def AutoencoderBatchedLoss(x, model):
    with torch.no_grad():
        loss = 0
        MSE = nn.MSELoss()
        for x_batch in x:
            l = 0.25/(len(x)*len(x_batch))
            encoded, x_hat = model(x_batch)
            for e in encoded:
                loss += e.norm(1)*l
            loss += MSE(x_hat, x_batch)/len(x)

    return loss.item()

def ClassifierLoss(x,y,ae,model):
    with torch.no_grad():
        loss = 0
        for x_batch,y_batch in zip(x,y):
            x_batch, _ = ae(x_batch)
            logits = model(x_batch)
            CrossEntropy = nn.CrossEntropyLoss()
            loss += CrossEntropy(logits,y_batch)/len(x)
    return loss

def ClassifierAccuracy(x,y,ae,model):

    with torch.no_grad():
        acc = 0
        for x_batch,y_batch in zip(x,y):
            x_batch, _ = ae(x_batch)
            logits = model(x_batch)
            acc += (torch.argmax(F.log_softmax(logits,dim=1),dim=1) == y_batch).float().mean()/len(x)
    return acc

def ConfusionMat(x,y,ae,model,plot=True):
    with torch.no_grad():
        x, _ = ae(x)
        logits = model(x)
        pred = torch.argmax(F.log_softmax(logits,dim=1),dim=1)
        cm = confusion_matrix(y,pred)
    
    if plot:
        plt.figure(figsize=(15,9))
        ax = sns.heatmap(cm, annot=True, fmt='d', linewidths=.5, square=True)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
    return cm

def Batch(l, J):
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

# Read Data
if Channel_Count == 1:
    x_train = torch.load('CWRU/presplit/arxiv_singlechannel/x_train.pt')
    y_train = torch.load('CWRU/presplit/arxiv_singlechannel/y_train.pt')
    x_test = torch.load('CWRU/presplit/arxiv_singlechannel/x_test.pt')
    y_test = torch.load('CWRU/presplit/arxiv_singlechannel/y_test.pt')

if Channel_Count == 2:
    x_train = torch.load('CWRU/presplit/arxiv_multichannel/x_train.pt')
    y_train = torch.load('CWRU/presplit/arxiv_multichannel/y_train.pt')
    x_test = torch.load('CWRU/presplit/arxiv_multichannel/x_test.pt')
    y_test = torch.load('CWRU/presplit/arxiv_multichannel/y_test.pt')

weights = torch.load('saves/CWRU_Class_Weights.pt')

# Confusion Matrix
# ae = Arxiv()
# cl = Classifier()
# ae.load_state_dict(torch.load('saves/Arxiv.pt'))
# cl.load_state_dict(torch.load('saves/Arxiv_Classifier.pt'))
# cm = ConfusionMat(x_test, y_test, ae, cl)

# Cuda
device = torch.device("cuda")
x_train = x_train.to(device)
x_test = x_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)
weights = weights.to(device)

# Autoencoder
x_test = Batch(x_test,J*100)
y_test = Batch(y_test,100)
ae = Arxiv().cuda()

MSE = nn.MSELoss()
ae_opt = torch.optim.Adam(ae.parameters(), lr=2e-4)
ae_epochs = 4

ae_train_history = []
ae_test_history = []

for epoch in range(ae_epochs):
    ae_opt.zero_grad()
    encoded_features, reconstructed = ae(x_train)
    loss = 0
    for x in encoded_features:
        loss += x.norm(1)
    loss = loss*0.25/len(x_train)
    loss += MSE(reconstructed, x_train)
    loss.backward()
    ae_opt.step()
    
    ae.eval()
    ae_train_history.append(AutoencoderLoss(x_train,ae))
    ae_test_history.append(AutoencoderBatchedLoss(x_test,ae))
    ae.train()
    print(f'epoch: {epoch+1}/{ae_epochs}')
    print(f'train loss: {ae_train_history[-1]}')
    print(f'test loss: {ae_test_history[-1]}')


plt.figure(figsize=(15,9))
plt.plot(range(1,ae_epochs+1),ae_train_history,label='train loss')
plt.plot(range(1,ae_epochs+1),ae_test_history,label='test loss')
plt.xlim(1,ae_epochs)
plt.title('Train Loss vs Epochs Passed')
plt.xlabel('Epoch')
plt.ylabel('MSE + L1 Norm')
plt.legend()
plt.show()

torch.save(ae.state_dict(),'saves/Arxiv_20.pt')

# Classifier
x_train = Batch(x_train,J*100)
y_train = Batch(y_train,100)

# ae = Arxiv().cuda()
# ae.load_state_dict(torch.load('saves/Arxiv_40.pt'))
ae.eval()

CrossEntropy = nn.CrossEntropyLoss(weight=weights)

cl = Classifier().cuda()
cl_opt = torch.optim.Adam(cl.parameters(), lr=1e-1)
cl_epochs = 100

cl_train_loss = []
cl_test_loss = []
cl_train_accuracy = []
cl_test_accuracy = []

for epoch in range(cl_epochs):
    for x, y in zip(x_train,y_train):
        cl_opt.zero_grad()
        with torch.no_grad():
            encoded, _  = ae(x)
        logits = cl(encoded)
        loss = CrossEntropy(logits, y)
        loss.backward()
        cl_opt.step()
    
    cl.eval()
    cl_train_loss.append(ClassifierLoss(x_train,y_train,ae,cl))
    cl_test_loss.append(ClassifierLoss(x_test,y_test,ae,cl))
    cl_train_accuracy.append(ClassifierAccuracy(x_train,y_train,ae,cl))
    cl_test_accuracy.append(ClassifierAccuracy(x_test,y_test,ae,cl))
    cl.train()
    
    print(f"epoch: {epoch+1}/{cl_epochs}")
    print(f"train loss is: {cl_train_loss[-1]}")
    print(f"test loss is: {cl_test_loss[-1]}")
    print(f"train accuracy is: {cl_train_accuracy[-1]}")
    print(f"test accuracy is: {cl_test_accuracy[-1]}")

plt.figure(figsize=(15,9))
plt.plot(range(1,cl_epochs+1),cl_train_loss,label='Train Loss')
plt.plot(range(1,cl_epochs+1),cl_test_loss,label='Test Loss')
plt.xlim(1,cl_epochs)
plt.title('Loss vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy Loss')
plt.legend()
plt.show()

plt.figure(figsize=(15,9))
plt.plot(range(1,cl_epochs+1),cl_train_accuracy,label='Train Accuracy')
plt.plot(range(1,cl_epochs+1),cl_test_accuracy,label='Test Accuracy')
plt.xlim(1,cl_epochs)
plt.title('Accuracy vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

torch.save(cl.state_dict(),'saves/Arxiv_20_Classifier.pt')
