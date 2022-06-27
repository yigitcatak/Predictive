#%%
# CROSS TEST
from PREDICTIVE_DEFINITIONS import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CLASS_COUNT = 2
N, J = Settings('CWRU') 

K = 200

x_test = torch.load('datasets/Cozum/presplit/x.pt')
y_test = torch.load('datasets/Cozum/presplit/y.pt')

ae = NSAELCN(N,K).to(DEVICE)
ae.load_state_dict(torch.load('saves/Paderborn_NSAELCN_AE2.pt', map_location='cpu'))
ae.eval()

cl = Classifier(K,CLASS_COUNT,J,MLP=True).to(DEVICE)
cl.load_state_dict(torch.load('saves/Paderborn_NSAELCN_CL2.pt', map_location='cpu'))
cl.eval()

x_test = x_test.to(DEVICE)
y_test = y_test.to(DEVICE)


_,test_accuracy = ClassifierEvaluate(x_test,y_test,ae,cl,isBatched=False)
_ = ConfusionMat(x_test,y_test,ae,cl,CLASS_COUNT,isBatched=False)
print(f'test accuracy: {test_accuracy}')