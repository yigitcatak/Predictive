#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg
from os import listdir
from os.path import isfile, join
from random import shuffle

SampleLength = 64000
N = 50 # input dim
K = 400 # encoding dim
J = int(SampleLength/N) # number of segments in a sample
AllFiles = [f for f in listdir('dataset_paderborn/segmented')]
ClassCount = 3

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    return loss.item()

def ClassifierEvaluate(x,y,model):
    with torch.no_grad():
        CrossEntropy = nn.CrossEntropyLoss()
        loss = 0
        acc = 0
        for x_batch,y_batch in zip(x,y):
            logits = model(x_batch)
            loss += CrossEntropy(logits,y_batch)/len(x)
            acc += (torch.argmax(F.log_softmax(logits,dim=1),dim=1) == y_batch).float().mean()/len(x)

    return loss.item(), acc.item()

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

# GroupSamples ile aynı şu an ama ben dosya okumayı tamamen ayrı .py dosyasına aldım
# O yüzden böyle eklemiştim burada da Batch olarak kalsın ismen daha anlaşılır olur
def Batch(l,J):
    def inner():
        for i in range(0, len(l)-(len(l)%J),J):
            yield l[i:i+J]
    return list(inner())

# READ DATA
splits = [['K002','KA01','KI01'],['K002','KA01','KI05'],['K002','KA01','KI07'],['K002','KA05','KI01'],['K002','KA05','KI05'],['K002','KA05','KI07'],['K002','KA07','KI01'],['K002','KA07','KI05'],['K002','KA07','KI07']]
alltrain = ['K002','KA01','KI01','KA05','KI05','KA07','KI07']
train_loss_all = []
train_acc_all = []
test_loss_all = []
test_acc_all = []

for k in range(len(splits)):
    x_mixed_train = []
    x_mixed_test = []
    y_mixed_train = []
    y_mixed_test = []

    for name in AllFiles:
        df = pd.read_csv(f'dataset_paderborn/segmented/{name}')
        data = df.drop(['label'], axis=1).values.tolist()[:len(df)-len(df)%J]
        label = df['label'].values.tolist()[:len(df)-len(df)%J:J]

        for i in range(len(splits[k])):
            if splits[k][i] in name: 
                train = data
                y_train = label
                train = GroupSamples(train,J)
                x_mixed_train += list(zip(train,y_train))

            else:
                for j in range(len(alltrain)):
                    if alltrain[j] in name:
                        x_mixed_test += data
                        y_mixed_test += label
        
    shuffle(x_mixed_train)
    x_mixed_train, y_mixed_train = zip(*x_mixed_train)
    x_mixed_train = Flatten(x_mixed_train)

    x_mixed_train, mean, eigenVecs, diagonal_mat = Whiten(x_mixed_train)
    x_mixed_test = Whiten(x_mixed_test, mean, eigenVecs, diagonal_mat)

    x_mixed_train = torch.tensor(x_mixed_train,dtype=torch.float32)
    x_mixed_test = torch.tensor(x_mixed_test,dtype=torch.float32)

    y_mixed_train = torch.tensor(y_mixed_train,dtype=torch.long)
    y_mixed_test = torch.tensor(y_mixed_test,dtype=torch.long)

    x_mixed_train = x_mixed_train.to(device)
    x_mixed_test = x_mixed_test.to(device)
    y_mixed_train = y_mixed_train.to(device)
    y_mixed_test = y_mixed_test.to(device)

    # bunların uzunuluğu ne kadar oluyor bilmediğim için şimdilik ufak tutmaya çalıştım
    x_mixed_train = Batch(x_mixed_train, 128*J)
    x_mixed_test = Batch(x_mixed_test, 128*J)
    y_mixed_train = Batch(y_mixed_train, 128)
    y_mixed_test = Batch(y_mixed_test, 128)

    print('Training Autoencoder')
    autoencoder = TiedAutoencoder().to(device)
    ae_opt = torch.optim.LBFGS(autoencoder.parameters(), lr=1e-2)
    MSE = nn.MSELoss()
    ae_epochs = 10

    ae_validation_history = []
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
        ae_validation_history.append(AutoencoderLoss(x_mixed_test, autoencoder))
        autoencoder.train()
        print(f'AE validation loss: {ae_validation_history[-1]}')
        
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
        val_loss, val_acc = ClassifierEvaluate(x_mixed_test,y_mixed_test,classifier)
        train_loss, train_acc = ClassifierEvaluate(x_mixed_train,y_mixed_train,classifier)
        classifier.train()
        
        cl_train_loss.append(train_loss)
        cl_train_accuracy.append(train_acc)
        cl_test_loss.append(val_loss)
        cl_test_accuracy.append(val_acc)
        
        print(f'CL train loss: {cl_train_loss[-1]}')
        print(f'CL train accuracy: {cl_train_accuracy[-1]}')
        print(f'CL validation loss: {cl_test_loss[-1]}')
        print(f'CL validation accuracy: {cl_test_accuracy[-1]}')
        
    train_loss_all.append(cl_train_loss)
    train_acc_all.append(cl_train_accuracy)
    test_loss_all.append(cl_test_loss)
    test_acc_all.append(cl_test_accuracy)

    del x_mixed_train, x_mixed_test, y_mixed_train, y_mixed_test, autoencoder, classifier
    torch.cuda.empty_cache()

pd.DataFrame(train_loss_all).to_csv('train_loss_all.csv',index=False)
pd.DataFrame(train_acc_all).to_csv('train_acc_all.csv',index=False)
pd.DataFrame(test_loss_all).to_csv('test_loss_all.csv',index=False)
pd.DataFrame(test_acc_all).to_csv('test_acc_all.csv',index=False)

# Classifier'ı kaydetmek istersen

# torch.save(classifier.state_dict(),'konum/dosyaismi.pt')

# Geri yüklemek için

# classifier = Classifier()
# classifier.load_state_dict(torch.load('dosyaismi.pt'))