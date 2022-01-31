#%%
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

SampleLength = 6400
N = 160 # input dim
K = 400 # encoding dim
J = int(SampleLength/N) # number of segments in a sample
AllFiles = [f for f in listdir('dataset_paderborn/segmented')]
ClassCount = 4

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: {}'.format(device))

use_presplit = False
update_presplit = True

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
        self.Linear = nn.Linear(K, ClassCount, bias=False)

        if Wloc is not None:
            self.Wloc.weight = nn.Parameter(Wloc, requires_grad=False)
        
    def forward(self,x):
        with torch.no_grad():
            local_features = F.relu(self.Wloc(x))
        # get the mean segment of each J segment
        features = torch.stack([torch.mean(local_features[i:i+J], dim=0) for i in range(0,len(local_features),J)])
        logits = self.Linear(features)
        return logits

def AutoencoderLoss(x, model):
    with torch.no_grad():
        loss = 0
        MSE = nn.MSELoss()
        for x_batch in x:
            l = 0.25/(len(x)*len(x_batch))
            encoded, x_hat = model(x_batch)
            for e in encoded:
                loss += e.norm(1)*l
            loss += MSE(x_hat, x_batch)/len(x)
        return loss

def ClassifierLossAcc(x,y,model):
    with torch.no_grad():
        CrossEntropy = nn.CrossEntropyLoss()
        loss = 0
        acc = 0
        for x_batch,y_batch in zip(x,y):
            logits = model(x_batch)
            loss += CrossEntropy(logits,y_batch)/len(x)
            acc += (torch.argmax(F.log_softmax(logits,dim=1),dim=1) == y_batch).float().mean()/len(x)

        return loss, acc

def ClassifierAccuracy(x,y,model):
    with torch.no_grad():
        acc = 0
        for x_batch,y_batch in zip(x,y):
            logits = model(x_batch)
            acc += (torch.argmax(F.log_softmax(logits,dim=1),dim=1) == y_batch).float().mean()/len(x)
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
        eigenVals, eigenVecs = np.linalg.eigh(cov)
        diagonal_mat = np.diag(1.0/((eigenVals)**0.5))

        intermediate_mat = np.dot(eigenVecs, diagonal_mat)
        whitened = np.dot(x, intermediate_mat)

        #uncorrelated = np.dot(x,eigenVecs)
        #whitened = np.dot(uncorrelated, diagonal_mat)
        return whitened, mean, eigenVecs, diagonal_mat
    else:
        x = np.array(x)
        x -= mean
        
        intermediate_mat = np.dot(eigenVecs, diagonal_mat)
        whitened = np.dot(x, intermediate_mat)

        #uncorrelated = np.dot(x,eigenVecs)
        #whitened = np.dot(uncorrelated, diagonal_mat)
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

def ConfusionMat(x,y,ae,model,plot=True):
    # GPU'da hata veriyor, confusion_matrix() fonksiyonu, CPU'da çalıştırın
    with torch.no_grad():
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

# GroupSamples ile aynı şu an ama ben dosya okumayı tamamen ayrı .py dosyasına aldım
# O yüzden böyle eklemiştim burada da Batch olarak kalsın ismen daha anlaşılır olur
def Batch(l,J):
    def inner():
        for i in range(0, len(l)-(len(l)%J),J):
            yield l[i:i+J]
    return list(inner())

#%%
# READ DATA
use_presplit = False
if not use_presplit:
   
    x_mixed_train = []
    x_mixed_test = []
    y_mixed_train = []
    y_mixed_test = []

    for name in AllFiles:
        df = pd.read_csv(f'dataset_paderborn/segmented/{name}')
        data = df.drop(['label'], axis=1).values.tolist()[:len(df)-len(df)%J]
        label = df['label'].values.tolist()[:len(df)-len(df)%J:J]  # sıkıntı var
        if ('K002' in name) or ('KA01' in name) or ('KA05' in name) or ('KA07' in name) or ('KI01' in name) or ('KI05' in name) or ('KI07' in name): 
            
            # get every Jth label as the labels are given with the average of J segments to classifier
            train = data
            y_train = label
            
            train = GroupSamples(train,J)
            temp = list(zip(train,y_train))
            shuffle(temp)
            train, y_train = zip(*temp)
            train = Flatten(train)
            x_mixed_train += train
            y_mixed_train += y_train

        elif ('K001' in name) or ('KA22' in name) or ('KA04' in name) or ('KA15' in name) or ('KA30' in name) or ('KA16' in name) or ('KI14' in name) or ('KI21' in name) or ('KI17' in name) or ('KI18' in name) or ('KI16' in name) :
            x_mixed_test += data
            y_mixed_test += label

        
    x_mixed_train, mean, eigenVecs, diagonal_mat = Whiten(x_mixed_train)
    x_mixed_test = Whiten(x_mixed_test, mean, eigenVecs, diagonal_mat)

    x_mixed_train = torch.tensor(x_mixed_train,dtype=torch.float32)
    x_mixed_test = torch.tensor(x_mixed_test,dtype=torch.float32)
    
    y_mixed_train = torch.tensor(y_mixed_train,dtype=torch.long)
    y_mixed_test = torch.tensor(y_mixed_test,dtype=torch.long)

    if update_presplit:
        torch.save(x_mixed_train,'dataset_paderborn/presplit/x_mixed_train.pt')
        torch.save(x_mixed_test,'dataset_paderborn/presplit/x_mixed_test.pt')
        torch.save(y_mixed_train,'dataset_paderborn/presplit/y_mixed_train.pt')
        torch.save(y_mixed_test,'dataset_paderborn/presplit/y_mixed_test.pt')

else: #use pre-split data
    x_mixed_train = torch.load('dataset_paderborn/presplit/x_mixed_train.pt')
    x_mixed_test = torch.load('dataset_paderborn/presplit/x_mixed_test.pt')
    y_mixed_train = torch.load('dataset_paderborn/presplit/y_mixed_train.pt')
    y_mixed_test = torch.load('dataset_paderborn/presplit/y_mixed_test.pt')

x_mixed_train = x_mixed_train.to(device)
x_mixed_test = x_mixed_test.to(device)
y_mixed_train = y_mixed_train.to(device)
y_mixed_test = y_mixed_test.to(device)

#%%
# train datası zaten shuffled, her segmentte J timeseries datapoint ardışık olacak şekilde
# önce gruplanıp, labelları ile eşleştirilip öyle shufflelandı, eğer random crossval
# alacaksanız yine aynı şekilde önce gruplamanız gerekir
# ya da zaten shuffled diye doğrudan ayırabilirsiniz %100/k noktalarından



# bunların uzunuluğu ne kadar oluyor bilmediğim için şimdilik ufak tutmaya çalıştım
# her bir batch 50 segmente olacak şekilde aldım
x_mixed_train = Batch(x_mixed_train, 50*J)
x_mixed_test = Batch(x_mixed_test, 50*J)
y_mixed_train = Batch(y_mixed_train, 50)
y_mixed_test = Batch(y_mixed_test, 50)

print('Training Autoencoder')

autoencoder = TiedAutoencoder().to(device)
ae_opt = torch.optim.LBFGS(autoencoder.parameters(), lr=1e-2)
MSE = nn.MSELoss()
ae_epochs = 10

ae_test_loss = []
for epoch in range(ae_epochs):
    print(f'AE training epoch: {epoch+1}/{ae_epochs}')
    for x_batch in x_mixed_train:
        def closure():
            ae_opt.zero_grad()
            encoded_features, reconstructed = autoencoder(x_batch)
            loss = 0
            for e in encoded_features:
                loss += e.norm(1)
            loss = loss * 0.25/len(x_batch)
            loss += MSE(reconstructed, x_batch)
            loss.backward()
            return loss
        ae_opt.step(closure)

    # Her batch için değil, her epoch için loss hesapla
    autoencoder.eval()
    ae_test_loss.append(AutoencoderLoss(x_mixed_test, autoencoder))
    autoencoder.train()
    print(f'AE validation loss: {ae_test_loss[-1]}')
    
for w in autoencoder.parameters():
    Wloc = w.detach().clone()

print('Training Classifier')

classifier = Classifier(Wloc).to(device)
cl_opt = torch.optim.Adam(classifier.parameters(), lr=3e-2)
CrossEntropy = nn.CrossEntropyLoss()
cl_epochs = 1000

cl_train_loss = []
cl_test_loss = []
cl_train_accuracy = []
cl_test_accuracy = []

for epoch in range(cl_epochs):
    print(f'CL training epoch: {epoch+1}/{cl_epochs}')
    for x_batch, y_batch in zip(x_mixed_train,y_mixed_train):
        cl_opt.zero_grad()
        logits = classifier(x_batch)
        loss = CrossEntropy(logits, y_batch)
        loss.backward()
        cl_opt.step()
    
    classifier.eval()
    val_loss, val_acc = ClassifierLossAcc(x_mixed_test,y_mixed_test,classifier)
    train_loss, train_acc = ClassifierLossAcc(x_mixed_train,y_mixed_train,classifier)
    classifier.train()
    cl_train_loss.append(train_loss)
    cl_train_accuracy.append(train_acc)
    cl_test_loss.append(val_loss)
    cl_test_accuracy.append(val_acc)
    
    print(f'CL train loss: {cl_train_loss[-1]}')
    print(f'CL train accuracy: {cl_test_loss[-1]}')
    print(f'CL validation loss: {cl_train_accuracy[-1]}')
    print(f'CL validation accuracy: {cl_test_accuracy[-1]}')
          

pd.DataFrame(cl_train_loss).to_csv('train_loss.csv',index=False)
pd.DataFrame(cl_test_loss).to_csv('train_acc.csv',index=False)
pd.DataFrame(cl_train_accuracy).to_csv('test_loss.csv',index=False)
pd.DataFrame(cl_test_accuracy).to_csv('test_acc.csv',index=False)

plt.figure(figsize=(15,9))
plt.plot(range(1,cl_epochs+1),cl_train_loss,label='Train Loss')
plt.plot(range(1,cl_epochs+1),cl_test_loss,label='Test Loss')
plt.xlim(1,cl_epochs)
plt.title('Loss vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy Loss')
plt.legend()
plt.savefig('Loss.png', bbox_inches='tight')

plt.figure(figsize=(15,9))
plt.plot(range(1,cl_epochs+1),cl_train_accuracy,label='Train Accuracy')
plt.plot(range(1,cl_epochs+1),cl_test_accuracy,label='Test Accuracy')
plt.xlim(1,cl_epochs)
plt.title('Accuracy vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('Accuracy.png', bbox_inches='tight')


# Classifier'ı kaydetmek istersen

# torch.save(classifier.state_dict(),'konum/dosyaismi.pt')

# Geri yüklemek için

# classifier = Classifier()
# classifier.load_state_dict(torch.load('dosyaismi.pt'))