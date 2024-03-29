#%%
# CWRU - ARXIV
# Definitions
from PREDICTIVE_DEFINITIONS import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CHANNEL_COUNT = 2
CLASS_COUNT = 2
N, J = Settings('CWRU')
K = N//2

# Read Data
if CHANNEL_COUNT == 1:
    x_train = torch.load('datasets/CWRU/presplit/x_train_fan_end.pt')
    x_test = torch.load('datasets/CWRU/presplit/x_test_fan_end.pt')
    y_train = torch.load('datasets/CWRU/presplit/y_train.pt')
    y_test = torch.load('datasets/CWRU/presplit/y_test.pt')
    x_train = torch.unsqueeze(torch.unsqueeze(x_train,dim=1),dim=1)
    x_test = torch.unsqueeze(torch.unsqueeze(x_test,dim=1),dim=1)

if CHANNEL_COUNT == 2:
    x_train = torch.load('datasets/CWRU/presplit/x_train_fan_end.pt')
    x_test = torch.load('datasets/CWRU/presplit/x_test_fan_end.pt')
    x_train2 = torch.load('datasets/CWRU/presplit/x_train_drive_end.pt')
    x_test2 = torch.load('datasets/CWRU/presplit/x_test_drive_end.pt')
    y_train = torch.load('datasets/CWRU/presplit/y_train.pt')
    y_test = torch.load('datasets/CWRU/presplit/y_test.pt')
    x_train = torch.unsqueeze(torch.cat([torch.unsqueeze(x_train,dim=1),torch.unsqueeze(x_train2,dim=1)],dim=1),dim=1)
    x_test = torch.unsqueeze(torch.cat([torch.unsqueeze(x_test,dim=1),torch.unsqueeze(x_test2,dim=1)],dim=1),dim=1)
    del x_train2, x_test2

weights = torch.load('datasets/CWRU/presplit/class_weights.pt')

x_train = x_train.to(DEVICE)
x_test = x_test.to(DEVICE)
y_train = y_train.to(DEVICE)
y_test = y_test.to(DEVICE)
weights = weights.to(DEVICE)

start = time.time()

# Autoencoder
ae = Arxiv(N,K,CHANNEL_COUNT).to(DEVICE)
MSE = nn.MSELoss()
ae_opt = torch.optim.Adam(ae.parameters(), lr=2e-4)
ae_epochs = 4

ae_train_loss = []
ae_test_loss = []
for epoch in range(ae_epochs):
    print(f'epoch: {epoch+1}/{ae_epochs}')
    ae_opt.zero_grad()
    encoded_features, reconstructed = ae(x_train)
    loss = 0
    for x in encoded_features:
        loss += x.norm(1)
    loss = loss*0.25/len(x_train)
    loss += MSE(reconstructed, x_train)
    loss.backward()
    ae_opt.step()
    
    # ae.eval()
    # ae_train_loss.append(AutoencoderLoss(x_train,ae))
    # ae_test_loss.append(AutoencoderLoss(x_test,ae))
    # ae.train()
    # print(f'train loss: {ae_train_loss[-1]}')
    # print(f'test loss: {ae_test_loss[-1]}')

# Classifier
ae.eval()
CrossEntropy = nn.CrossEntropyLoss(weight=weights)

cl = Classifier(K,CLASS_COUNT,J).to(DEVICE)
cl_opt = torch.optim.Adam(cl.parameters(), lr=1e-1)
cl_epochs = 25

cl_train_loss = []
cl_test_loss = []
cl_train_accuracy = []
cl_test_accuracy = []
for epoch in range(cl_epochs):
    print(f"epoch: {epoch+1}/{cl_epochs}")
    cl_opt.zero_grad()
    with torch.no_grad():
        encoded, _  = ae(x_train)
    logits = cl(encoded)
    loss = CrossEntropy(logits, y_train)
    loss.backward()
    cl_opt.step()

    cl.eval()
    train_loss,train_accuracy = ClassifierEvaluate(x_train,y_train,ae,cl)
    test_loss,test_accuracy = ClassifierEvaluate(x_test,y_test,ae,cl)
    cl.train()

    cl_train_loss.append(train_loss)
    cl_test_loss.append(test_loss) 
    cl_train_accuracy.append(train_accuracy)
    cl_test_accuracy.append(test_accuracy)

    print(f"train loss is: {train_loss}")
    print(f"test loss is: {test_loss}")
    print(f"train accuracy is: {train_accuracy}")
    print(f"test accuracy is: {test_accuracy}")
end = time.time()

torch.save(ae.state_dict(), 'saves/CWRU_Arxiv_AE.pt')
torch.save(cl.state_dict(), 'saves/CWRU_Arxiv_CL.pt')
print(f'time elapsed: {(end-start)//60:.0f} minutes {(end-start)%60:.0f} seconds')
# PlotResults(ae_train_loss,ae_test_loss,'Loss','MSE + L1 Norm')
PlotResults(cl_train_loss,cl_test_loss,'Loss','Cross Entropy Loss',isSave=True,savename='CWRU_Arxiv_Loss')
PlotResults(cl_train_accuracy,cl_test_accuracy,'Accuracy','Accuracy',isSave=True,savename='CWRU_Arxiv_Accuracy')
_ = ConfusionMat(x_test,y_test,ae,cl,CLASS_COUNT,isBatched=False)