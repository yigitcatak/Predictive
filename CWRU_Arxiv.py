#%%
# Definitions
from PREDICTIVE_DEFINITIONS import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

Channel_Count = 2
Class_Count = 10
Sample_Length = 1200
J = 30
N = (Sample_Length//J) - ((Sample_Length//J)%4)
K = 400

# Read Data
if Channel_Count == 1:
    x_train = torch.load('datasets/CWRU/presplit/x_train_fan_end.pt')
    x_test = torch.load('datasets/CWRU/presplit/x_test_fan_end.pt')
    y_train = torch.load('datasets/CWRU/presplit/y_train.pt')
    y_test = torch.load('datasets/CWRU/presplit/y_test.pt')
    x_train = torch.unsqueeze(torch.unsqueeze(x_train,dim=1),dim=1)
    x_test = torch.unsqueeze(torch.unsqueeze(x_test,dim=1),dim=1)

if Channel_Count == 2:
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

#%%
start = time.time()

# Autoencoder
ae = Arxiv(N,K,Channel_Count).to(DEVICE)
MSE = nn.MSELoss()
ae_opt = torch.optim.Adam(ae.parameters(), lr=2e-4)
ae_epochs = 4

ae_train_loss = []
ae_test_loss = []
for epoch in range(ae_epochs):
    print(f'epoch: {epoch+1}/{ae_epochs}')
    for x_batch in x_train:
        ae_opt.zero_grad()
        encoded_features, reconstructed = ae(x_batch)
        loss = 0
        for x in encoded_features:
            loss += x.norm(1)
        loss = loss*0.25/len(x_batch)
        loss += MSE(reconstructed, x_batch)
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

cl = Classifier(K,Class_Count,J).to(DEVICE)
cl_opt = torch.optim.Adam(cl.parameters(), lr=1e-1)
cl_epochs = 100

cl_train_loss = []
cl_test_loss = []
cl_train_accuracy = []
cl_test_accuracy = []
for epoch in range(cl_epochs):
    print(f"epoch: {epoch+1}/{cl_epochs}")
    for x_batch,y_batch in zip(x_train,y_train):
        cl_opt.zero_grad()
        with torch.no_grad():
            encoded, _  = ae(x_batch)
        logits = cl(encoded)
        loss = CrossEntropy(logits, y_batch)
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