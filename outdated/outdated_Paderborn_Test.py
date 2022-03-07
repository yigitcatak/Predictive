import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

Class_Count = 3
Subband_Count = 1
Channel_Count = 1
Sample_Length = 6400
J = 40
N = int(Sample_Length/J)
PARAM = 40
K = 7*N
class Arxiv(nn.Module):
    def __init__(self):
        super(Arxiv,self).__init__()
        self.drp1 = nn.Dropout(p = 0.1)
        self.drp2 = nn.Dropout(p = 0.9)
        self.flat = nn.Flatten()
        self.relu = nn.ReLU()
        self.bnorm = nn.BatchNorm2d(PARAM)

        # ENCODER
        self.conv1 = nn.Conv2d(1, PARAM, (Channel_Count,1))
        self.conv2 = nn.Conv2d(PARAM,PARAM,(1,Channel_Count),2)
        self.conv3 = nn.Conv2d(PARAM,PARAM, (1,11),padding=(0,5))

        # LINEAR
        self.lin1 = nn.Linear(int(N/2)*PARAM,int(N/2)*PARAM)
        self.lin2 = nn.Linear(int(N/2)*PARAM,K)
        self.lin3 = nn.Linear(K,int(N/2)*PARAM)
        self.lin4 = nn.Linear(int(N/2)*PARAM,int(N/2)*PARAM)

        # DECODER
        self.tconv1 = nn.ConvTranspose2d(PARAM, 1, (Channel_Count,1))
        self.tconv2 = nn.ConvTranspose2d(PARAM,PARAM,(1,Channel_Count),2) if Channel_Count == 2 else nn.ConvTranspose2d(PARAM,PARAM,(1,Channel_Count),2,output_padding=(0,1)) 
        self.tconv3 = nn.ConvTranspose2d(PARAM,PARAM,(1,11),padding=(0,5))

    def forward(self,x):
        encoder_out = self.drp2(self.bnorm(self.conv3(self.drp1(self.relu(self.bnorm(self.conv2((self.drp1(self.bnorm(self.conv1(x)))))))))))
        encoder_out = self.lin1(self.flat(encoder_out))
        bottleneck = self.lin2(encoder_out)
        decoder_out = torch.reshape(self.lin4(self.lin3(bottleneck)),(-1,PARAM,1,int(N/2)))
        decoder_out = self.tconv1(self.drp1(self.relu(self.tconv2(self.drp1(self.relu(self.tconv3(decoder_out)))))))
        return bottleneck, decoder_out
class Arxiv2(nn.Module):
    def __init__(self):
        super(Arxiv2,self).__init__()
        self.drp1 = nn.Dropout(p = 0.1)
        self.drp2 = nn.Dropout(p = 0.9)
        self.flat = nn.Flatten()
        self.relu = nn.ReLU()
        # ENCODER
        self.conv1 = nn.Conv2d(1, PARAM, (Channel_Count,1))
        self.conv2 = nn.Conv2d(PARAM,PARAM,(1,Channel_Count),2)
        self.conv3 = nn.Conv2d(PARAM,PARAM, (1,11),padding=(0,5))
        # LINEAR
        self.lin1 = nn.Linear(int(N/2)*PARAM,int(N/2)*PARAM)
        self.lin2 = nn.Linear(int(N/2)*PARAM,K)
        self.lin3 = nn.Linear(K,int(N/2)*PARAM)
        self.lin4 = nn.Linear(int(N/2)*PARAM,int(N/2)*PARAM)
        # DECODER
        self.tconv1 = nn.ConvTranspose2d(PARAM, 1, (Channel_Count,1))
        self.tconv2 = nn.ConvTranspose2d(PARAM,PARAM,(1,Channel_Count),2) if Channel_Count == 2 else nn.ConvTranspose2d(PARAM,PARAM,(1,Channel_Count),2,output_padding=(0,1)) 
        self.tconv3 = nn.ConvTranspose2d(PARAM,PARAM,(1,11),padding=(0,5))
    def forward(self,x):
        encoder_out = self.drp2(self.conv3(self.relu(self.drp1(self.relu(self.conv2(self.drp1(self.conv1(x))))))))
        encoder_out = self.lin1(self.flat(encoder_out))
        bottleneck = self.lin2(encoder_out)
        decoder_out = torch.reshape(self.lin4(self.lin3(bottleneck)),(-1,PARAM,1,int(N/2)))
        decoder_out = self.tconv1(self.drp1(self.relu(self.tconv2(self.drp1(self.relu(self.tconv3(decoder_out)))))))
        return bottleneck, decoder_out
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()
        self.lin1 = nn.Linear(K, K, bias=False)
        self.lin2 = nn.Linear(K, Class_Count, bias=False)
        self.relu = nn.ReLU()
        self.drp = nn.Dropout(p = 0.9)

    def forward(self,x):
        # get the mean segment of each J segment
        features = torch.stack([torch.mean(x[range(i,i+J)], dim=0) for i in range(0,len(x),J)]) 
        # logits = self.lin2(features)
        logits = self.lin2(self.drp(self.relu(self.lin1(features))))
        return logits
def ClassifierAccuracy(x,y,ae,model):
    with torch.no_grad():
        acc = 0
        for x_batch,y_batch in zip(x,y):
            x_batch, _ = ae(x_batch)
            logits = model(x_batch)
            acc += (torch.argmax(F.log_softmax(logits,dim=1),dim=1) == y_batch).float().mean()/len(x)
    return acc.item()
def Batch(l, J):
    def inner():
        for i in range(0, len(l)-(len(l)%J), J):
            yield l[i:i + J]
    return list(inner())

x_test = torch.load('datasets/Paderborn/presplit/x_test_vibration.pt')
y_test = torch.load('datasets/Paderborn/presplit/y_test.pt')
x_test = torch.unsqueeze(torch.unsqueeze(x_test,dim=1),dim=1)

x_test_CWRUstyle = torch.load('datasets/Paderborn/presplit/x_test_CWRUstyle.pt')
y_test_CWRUstyle = torch.load('datasets/Paderborn/presplit/y_test_CWRUstyle.pt')
x_test_CWRUstyle = torch.unsqueeze(torch.unsqueeze(x_test_CWRUstyle,dim=1),dim=1)

x_test = x_test.cuda()
y_test = y_test.cuda()
x_test_CWRUstyle = x_test_CWRUstyle.cuda()
y_test_CWRUstyle = y_test_CWRUstyle.cuda()

x_test = Batch(x_test,1024*J)
y_test = Batch(y_test,1024)
x_test_CWRUstyle = Batch(x_test_CWRUstyle,1024*J)
y_test_CWRUstyle = Batch(y_test_CWRUstyle,1024)

ae1 = Arxiv().cuda()
cl1 = Classifier().cuda()

ae2 = Arxiv().cuda()
cl2 = Classifier().cuda()

ae1.load_state_dict(torch.load('saves/Paderborn_Arxiv_AE.pt'))
cl1.load_state_dict(torch.load('saves/Paderborn_Arxiv_Classifier.pt'))
ae1.eval()
cl1.eval()

ae2.load_state_dict(torch.load('saves/Paderborn_Arxiv_AE_CWRUstyle.pt'))
cl2.load_state_dict(torch.load('saves/Paderborn_Arxiv_Classifier_CWRUstyle.pt'))
ae2.eval()
cl2.eval()

print(f'actual classifier: {ClassifierAccuracy(x_test,y_test,ae1,cl1)}')
print(f'CWRU style classifier: {ClassifierAccuracy(x_test,y_test,ae2,cl2)}')
print(f'actual classifier, CWRU style autoencoder: {ClassifierAccuracy(x_test,y_test,ae2,cl1)}')
print(f'CWRU classifier, actual autoencoder: {ClassifierAccuracy(x_test,y_test,ae1,cl2)}')