#%%
from PREDICTIVE_DEFINITIONS import *
from sklearn.model_selection import train_test_split
ALL_FILES = [f for f in listdir("datasets/CWRU/original")]
TRAIN_SIZE = 1/6
LABELS = {
            'B007': 0, 'B014': 1, 'B021': 2,
            'IR007': 3, 'IR014': 4, 'IR021': 5,
            'normal': 6,
            'OR007@6': 7, 'OR014@6': 8, 'OR021@6': 9
         }

CLASS_COUNT = 10
CHANNEL = {3:'drive_end', 4:'fan_end'}
# SEED = randint(0,1e6)

weights = dict(zip(list(range(CLASS_COUNT)),list(0 for i in range(CLASS_COUNT))))
y1 = []
y2 = []
x1 = []
x2 = []
for channel in [3,4]:
    for name in ALL_FILES:
        mat_file = scipy.io.loadmat(f'datasets/CWRU/original/{name}')
        if name == 'normal_2.mat':
            data = mat_file[list(mat_file)[channel+1]].flatten() # normal_2.mat has a wrong convention
        else:
            data = mat_file[list(mat_file)[channel]].flatten() # 3: Drive End, 4: Fan End

        name = name[:name.find('.')]
        labelname = name[:name.find('_')]
        label = LABELS[labelname]

        segmented = Batch(data, 64)
        segmented = Batch(segmented,64)

        y_segmented = [label for i in range(len(segmented))]
 
        if channel == 3:
            weights[label] += len(segmented)*TRAIN_SIZE
        x1 += segmented
        y1 += y_segmented
        # else:
        #     x2 += list(zip(segmented,y_segmented))        

x_train, x_test, y_train, y_test = train_test_split(x1, y1, train_size=TRAIN_SIZE)

x_train = torch.tensor(np.array(x_train),dtype=torch.float32)
x_test = torch.tensor(np.array(x_test),dtype=torch.float32)
y_train = torch.tensor(np.array(y_train),dtype=torch.long)
y_test = torch.tensor(np.array(y_test),dtype=torch.long)

x_train = torch.unsqueeze(x_train,dim=1)
x_test = torch.unsqueeze(x_test,dim=1)

weights = torch.tensor(list(weights.values()),dtype=torch.float32)
weights = 1/weights
weights = weights/weights.sum()

class LENET(nn.Module):
    def __init__(self):
        super(LENET,self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1,32,(5,5),padding=2),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(32,64,(3,3),padding=1),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(64,128,(3,3),padding=1),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(128,256,(3,3),padding=1),
            nn.MaxPool2d((2,2)),
            nn.Flatten(),
            nn.Linear(256*4*4,2560),
            nn.Linear(2560,768),
            nn.Linear(768,10)
        )

    def forward(self,x):
        return self.network(x)
    
def Evaluate(x,y,nw):
    with torch.no_grad():
        acc = 0
        loss = 0
        CrossEntropy = nn.CrossEntropyLoss()
        logits = nw(x)
        loss += CrossEntropy(logits,y)
        acc += (torch.argmax(F.log_softmax(logits,dim=1),dim=1) == y).float().mean()
    return loss.item(), acc.item()

#%%
nw = LENET()
CrossEntropy = nn.CrossEntropyLoss(weight=None)
opt = torch.optim.Adam(nw.parameters(), lr=3e-4)
epochs = 10

hist_train_loss = []
hist_test_loss = []
hist_train_accuracy = []
hist_test_accuracy = []

start = time.time()
for epoch in range(epochs):
    print(f"epoch: {epoch+1}/{epochs}")
    opt.zero_grad()
    logits = nw(x_train)
    loss = CrossEntropy(logits, y_train)
    loss.backward()
    opt.step()

    nw.eval()
    train_loss,train_accuracy = Evaluate(x_train,y_train,nw)
    test_loss,test_accuracy = Evaluate(x_test,y_test,nw)
    nw.train()

    hist_train_loss.append(train_loss)
    hist_test_loss.append(test_loss) 
    hist_train_accuracy.append(train_accuracy)
    hist_test_accuracy.append(test_accuracy)

    print(f"train loss is: {train_loss}")
    print(f"test loss is: {test_loss}")
    print(f"train accuracy is: {train_accuracy}")
    print(f"test accuracy is: {test_accuracy}")
end = time.time()

print(f'time elapsed: {(end-start)//60:.0f} minutes {(end-start)%60:.0f} seconds')
PlotResults(hist_train_loss,hist_test_loss,'Loss','Cross Entropy Loss',isSave=False)
PlotResults(hist_train_accuracy,hist_test_accuracy,'Accuracy','Accuracy',isSave=False)