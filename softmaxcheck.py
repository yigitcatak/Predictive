#%%
from PREDICTIVE_DEFINITIONS import *
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHANNEL_COUNT = 1
CLASS_COUNT = 3
N, J = Settings('Paderborn')
K = N//2
x_train = torch.load('datasets/Paderborn/presplit/x_train_vibration.pt')
x_test = torch.load('datasets/Paderborn/presplit/x_test_vibration.pt')
# y_train = torch.load('datasets/Paderborn/presplit/y_train.pt')
y_test = torch.load('datasets/Paderborn/presplit/y_test.pt')
# x_train = torch.unsqueeze(torch.unsqueeze(x_train,dim=1),dim=1)
x_test = torch.unsqueeze(torch.unsqueeze(x_test,dim=1),dim=1)
# x_train = x_train.to(DEVICE)
x_test = x_test.to(DEVICE)
# y_train = y_train.to(DEVICE)
y_test = y_test.to(DEVICE)
ae = Arxiv(N,K,CHANNEL_COUNT).to(DEVICE)
cl = Classifier(K,CLASS_COUNT,J,MLP=True).to(DEVICE)
ae.load_state_dict(torch.load('saves/Paderborn_Arxiv_AE.pt'))
ae.eval()
cl.load_state_dict(torch.load('saves/Paderborn_Arxiv_CL.pt'))
cl.eval()

y_test = y_test[:-(len(y_test)%256)]
label0 = []
label1 = []
label2 = []
for i in range(len(y_test)):
    if y_test[i] == 0:
        label0.append(i)
    elif y_test[i] == 1:
        label1.append(i)
    else:
        label2.append(i)

softmax = []
with torch.no_grad():
    x_test = Batch(x_test, 256*J)
    
    for x_batch in x_test:
        encoded, _  = ae(x_batch)
        t = cl(encoded)
        softmax += F.softmax(t,dim=-1)

    softmax = torch.stack(softmax)

    print(f'mean softmax0 is: {softmax[label0].mean(dim=0)}')
    print(f'mean softmax1 is: {softmax[label1].mean(dim=0)}')
    print(f'mean softmax2 is: {softmax[label2].mean(dim=0)}')

y_test = Batch(y_test,256)
_ = ConfusionMat(x_test,y_test,ae,cl,CLASS_COUNT,isBatched=True)