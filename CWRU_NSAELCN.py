#%%
# Definitions
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

Class_Count = 10
Sample_Length = 1200
J = 30
N = int(Sample_Length/J)
Ks = [20,40,80,100,200,300,400]

MeanAccs = []
StdAccs = []
for K in [400]:
    class TiedAutoencoder(nn.Module):
        def __init__(self):
            super(TiedAutoencoder,self).__init__()
            self.Encoder = nn.Linear(N, K, bias=False)
        def forward(self,x):
            encoded_features = F.relu(self.Encoder(x))
            reconstructed = F.linear(encoded_features, self.Encoder.weight.t())
            return encoded_features, reconstructed
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
            loss = loss*0.25/(len(x))
            loss += MSE(x_hat, x)
        return loss.item()

    def ClassifierEvaluate(x,y,ae,model):
        with torch.no_grad():
            acc = 0
            loss = 0
            CrossEntropy = nn.CrossEntropyLoss()
            x, _ = ae(x)
            logits = model(x)
            loss += CrossEntropy(logits,y)/len(x)
            acc += (torch.argmax(F.log_softmax(logits,dim=1),dim=1) == y).float().mean()
        return loss.item(), acc.item()

    def ClassifierAccuracy(x,y,ae,model):
        with torch.no_grad():
            acc = 0
            x, _ = ae(x)
            logits = model(x)
            acc += (torch.argmax(F.log_softmax(logits,dim=1),dim=1) == y).float().mean()
        return acc.item()

    def Batch(l, J):
        def inner():
            for i in range(0, len(l)-(len(l)%J), J):
                yield l[i:i + J]
        return list(inner())

    # Read Data
    
    x_train = torch.load('datasets/CWRU/presplit/x_train_fan_end.pt')
    x_test = torch.load('datasets/CWRU/presplit/x_test_fan_end.pt')
    y_train = torch.load('datasets/CWRU/presplit/y_train.pt')
    y_test = torch.load('datasets/CWRU/presplit/y_test.pt')
    weights = torch.load('datasets/CWRU/presplit/class_weights.pt')

#%%
    # Cuda
    x_train = x_train.cuda()
    x_test = x_test.cuda()
    y_train = y_train.cuda()
    y_test = y_test.cuda()
    weights = weights.cuda()

    Accuracies = []
    times = []
    for i in range(10):
        print(f'trial {i}')
        # Autoencoder
        ae = TiedAutoencoder().cuda()
        MSE = nn.MSELoss()
        ae_opt = torch.optim.LBFGS(ae.parameters(), lr=3e-2)
        ae_epochs = 10
        start = time.time()
        for epoch in range(ae_epochs):
            # print(f'epoch: {epoch+1}/{ae_epochs}')
            def closure():
                ae_opt.zero_grad()
                encoded_features, reconstructed = ae(x_train)
                loss = 0
                for x in encoded_features:
                    loss += x.norm(1)
                loss = loss*0.25/len(x_train)
                loss += MSE(reconstructed, x_train)
                loss.backward()
                return loss
            ae_opt.step(closure)

        # Classifier
        ae.eval()
        CrossEntropy = nn.CrossEntropyLoss(weight=weights)

        cl = Classifier().cuda()
        cl_opt = torch.optim.Adam(cl.parameters(), lr=1e-1)
        cl_epochs = 100

        for epoch in range(cl_epochs):
            # print(f"epoch: {epoch+1}/{cl_epochs}")
            cl_opt.zero_grad()
            with torch.no_grad():
                encoded, _  = ae(x_train)
            logits = cl(encoded)
            loss = CrossEntropy(logits, y_train)
            loss.backward()
            cl_opt.step()
        end = time.time()
        # cl.eval()
        # Accuracies.append(ClassifierAccuracy(x_test,y_test,ae,cl))
        # print(f'Accuracy of trial {i}: {Accuracies[-1]}')
        del ae, cl
        torch.cuda.empty_cache()
        times.append(end-start)
        
    del x_train, x_test, y_train, y_test
    torch.cuda.empty_cache()
    print(f'average time: {np.mean(times)}, std time: {np.std(times)}')

#     MeanAccs.append(np.mean(Accuracies))
#     StdAccs.append(np.std(Accuracies))
#     print(f'K: {K}, mean: {MeanAccs[-1]}, std: {StdAccs[-1]}')

# pd.DataFrame(MeanAccs).to_csv(f'CWRU_NSAELCN_Accs.csv',index=False)
# pd.DataFrame(StdAccs).to_csv(f'CWRU_NSAELCN_Stds.csv',index=False)