#%%
# DEFINITIONS & DATA
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class TiedAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, Wloc):
        super(TiedAutoencoder,self).__init__()
        self.Wloc = Wloc
    
    def forward(self,x):
        encoded_features = F.relu( F.linear(x, self.Wloc) )
        reconstructed = F.linear(encoded_features, self.Wloc.t())
        return reconstructed, encoded_features

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

normal = torch.load('dataset/presplit/x_normal_test.pt')
anomaly = torch.load('dataset/presplit/x_anomaly_test.pt')

#%%
# ROC
resolution = 500
Wloc = torch.load('saves/healthy_wloc.pt')
model = TiedAutoencoder(50,400,Wloc)

minloss = 1e20
maxloss = 0
for x in normal:
    loss = AutoencoderLoss(x, model, verbose=False)
    if loss < minloss:
        minloss = loss
print(f'min loss: {minloss}')

for x in anomaly:
    loss = AutoencoderLoss(x, model, verbose=False)
    if loss > maxloss:
        maxloss = loss
print(f'max loss: {maxloss}')
stepsize = (maxloss - minloss)/resolution

results = []
thresholds = [minloss + i*stepsize for i in range(resolution)]
for threshold in thresholds:
    print(f'step: {int(threshold/stepsize)}/{resolution}')
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for x in normal:
        loss = AutoencoderLoss(x,model,verbose=False)
        if loss <= threshold:
            true_negative += 1
        else:
            false_positive += 1

    for x in anomaly:
        loss = AutoencoderLoss(x,model,verbose=False)
        if loss <= threshold:
            false_negative += 1
        else:
            true_positive += 1
    
    results.append([true_positive,false_positive,true_negative,false_negative])
            
df = pd.DataFrame(results)
df.columns = ['true_positive','false_positive','true_negative','false_negative']
tpr = df['true_positive']/len(anomaly)
fpr = df['false_positive']/len(normal)
plt.figure(figsize=(10,10))
plt.plot(fpr,tpr)
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
df.to_csv('roc results.csv')