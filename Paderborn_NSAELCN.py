#%%
# PADERBORN - NSAELCN
# Definitions
from PREDICTIVE_DEFINITIONS import *
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASS_COUNT = 2
N, J = Settings('CWRU')
K = 200
TRAIN_AE = True

# Read Data
x_train = torch.load('datasets/Paderborn/presplit/x_train_vibration.pt')
x_test = torch.load('datasets/Paderborn/presplit/x_test_vibration.pt')
y_train = torch.load('datasets/Paderborn/presplit/y_train.pt')
y_test = torch.load('datasets/Paderborn/presplit/y_test.pt')

weights = torch.load('datasets/Paderborn/presplit/class_weights.pt')

x_train = x_train.to(DEVICE)
x_test = x_test.to(DEVICE)
y_train = y_train.to(DEVICE)
y_test = y_test.to(DEVICE)
weights = weights.to(DEVICE)

x_train = Batch(x_train,256*J)
x_test = Batch(x_test,256*J)
y_train = Batch(y_train,256)
y_test = Batch(y_test,256)

start = time.time()

# Autoencoder
if TRAIN_AE:
    ae = NSAELCN(N,K).to(DEVICE)
    MSE = nn.MSELoss()
    ae_opt = torch.optim.Adam(ae.parameters(), lr=1e-2)
    ae_epochs = 10

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
        # ae_train_loss.append(AutoencoderLoss(x_train,ae,isBatched=True))
        # ae_test_loss.append(AutoencoderLoss(x_test,ae,isBatched=True))
        # ae.train()
        # print(f'train loss: {ae_train_loss[-1]}')
        # print(f'test loss: {ae_test_loss[-1]}')
    
    torch.save(ae.state_dict(), 'saves/Paderborn_NSAELCN_AE.pt')

# Classifier
ae = NSAELCN(N,K).to(DEVICE)
ae.load_state_dict(torch.load('saves/Paderborn_NSAELCN_AE.pt'))
ae.eval()
CrossEntropy = nn.CrossEntropyLoss(weight=weights)

cl = Classifier(K,CLASS_COUNT,J,MLP=True).to(DEVICE)
cl_opt = torch.optim.Adam(cl.parameters(), lr=0.2e-1)
cl_epochs = 5

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
    train_loss,train_accuracy = ClassifierEvaluate(x_train,y_train,ae,cl,isBatched=True)
    test_loss,test_accuracy = ClassifierEvaluate(x_test,y_test,ae,cl,isBatched=True)
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

torch.save(cl.state_dict(), 'saves/Paderborn_NSAELCN_CL.pt')
print(f'time elapsed: {(end-start)//60:.0f} minutes {(end-start)%60:.0f} seconds')
# PlotResults(ae_train_loss,ae_test_loss,'Loss','MSE + L1 Norm')
PlotResults(cl_train_loss,cl_test_loss,'Loss','Cross Entropy Loss',isSave=True,savename='Paderborn_NSAELCN_Loss')
PlotResults(cl_train_accuracy,cl_test_accuracy,'Accuracy','Accuracy',isSave=True,savename='Paderborn_NSAELCN_Accuracy')
_ = ConfusionMat(x_test,y_test,ae,cl,CLASS_COUNT,isBatched=True)
print('\n\n\n')