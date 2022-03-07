#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

#import doesn't conclude, takes too much time ??
#from train import SampleLength, N, J, K, ClassCount, Whiten, Classifier

stringlabels = {
            0: 'Ball Fault 0.007"', 1: 'Ball Fault 0.014"', 2: 'Ball Fault 0.021"',
            3: 'Inner Ring Fault 0.007"', 4: 'Inner Ring Fault 0.014"', 5: 'Inner Ring Fault 0.021"',
            6: 'Healthy',
            7: 'Outter Ring Fault 0.007"', 8: 'Outter Ring Fault 0.014"', 9: 'Outter Ring Fault 0.021"'
         }

SampleLength = 1200
N = 50
K = 400
J = int(SampleLength/N)
ClassCount = 10

class Classifier(nn.Module):
    def __init__(self, encoding_dim, class_count, W1, J, W2):
        super(Classifier,self).__init__()
        self.W1 = nn.parameter.Parameter(W1, requires_grad=False)
        self.J = J
        self.W2 = nn.parameter.Parameter(W2, requires_grad=False)

    def forward(self,x):
        with torch.no_grad():
            local_features = F.relu(F.linear(x,self.W1))

        # get the mean segment of each J segment
        features = [torch.mean(local_features[range(i,i+self.J)], dim=0) for i in range(0,len(local_features),self.J)] 
        features = torch.tensor([t.numpy() for t in features])

        # apply the linear layer
        logits = F.linear(features,self.W2)
        return logits

def Whiten(x, mean, eigenVecs, diagonalMat):
    x = np.array(x)
    x -= mean
    uncorrelated = np.dot(x,eigenVecs)
    whitened = np.dot(uncorrelated, diagonalMat)
    return whitened


x_mixed_test = torch.load('dataset/presplit/x_mixed_test.pt')
mean = pd.read_csv('saves/mean.csv')['0']
eigenVecs = pd.read_csv('saves/eigenVecs.csv')
diagonalMat = pd.read_csv('saves/diagonalMat.csv')
x_mixed_test = torch.tensor(Whiten(x_mixed_test, mean, eigenVecs, diagonalMat), dtype=torch.float32)

W1 = torch.load(f'saves/WClassifier{N}x{K}.pt')
W2 = torch.load(f'saves/WClassifier{K}x{ClassCount}.pt')

classifier = Classifier(K,ClassCount,W1,J,W2)
with torch.no_grad():
    logits = classifier(x_mixed_test)
    predictions = torch.argmax(F.log_softmax(logits,dim=1),dim=1)

l = []
for intlabel in predictions:
    l.append(stringlabels[int(intlabel)])

pd.DataFrame(l).to_csv('history/results.csv',index=False)