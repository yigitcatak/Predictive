#%%
# CROSS TEST
from PREDICTIVE_DEFINITIONS import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAIN_AE = True
CLASS_COUNT = 2
N, J = Settings('CWRU')

ARXIV = 0
K = 200

# x_test = torch.load('datasets/CWRU/presplit/x_test_transfer_fan_end.pt')
# y_test = torch.load('datasets/CWRU/presplit/y_test_transfer.pt')

x_test = torch.load('datasets/Cozum/presplit/x.pt')
y_test = torch.load('datasets/Cozum/presplit/y.pt')

if ARXIV:
    x_test = torch.unsqueeze(torch.unsqueeze(x_test,dim=1),dim=1)
    ae = Arxiv(N,K,1).to(DEVICE)
    ae.load_state_dict(torch.load('saves/Paderborn_Arxiv_AE3.pt'))
    ae.eval()

    cl = Classifier(K,CLASS_COUNT,J,MLP=True).to(DEVICE)
    cl.load_state_dict(torch.load('saves/Paderborn_Arxiv_CL3.pt'))
    cl.eval()

else: 
    ae = NSAELCN(N,K).to(DEVICE)
    ae.load_state_dict(torch.load('saves/Paderborn_NSAELCN_AE2.pt'))
    ae.eval()

    cl = Classifier(K,CLASS_COUNT,J,MLP=True).to(DEVICE)
    cl.load_state_dict(torch.load('saves/Paderborn_NSAELCN_CL2.pt'))
    cl.eval()

x_test = x_test.to(DEVICE)
y_test = y_test.to(DEVICE)


_,test_accuracy = ClassifierEvaluate(x_test,y_test,ae,cl,isBatched=False)
_ = ConfusionMat(x_test,y_test,ae,cl,CLASS_COUNT,isBatched=False)
print(f'test accuracy: {test_accuracy}')