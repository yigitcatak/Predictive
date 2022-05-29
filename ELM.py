#%%
# ELM
from PREDICTIVE_DEFINITIONS import *

def ELMAccuracy(pred,y):
    with torch.no_grad():
        acc = (torch.argmax(F.log_softmax(pred,dim=1),dim=1) == y).float().mean()
    return acc.item()

class ELM(nn.Module):
    def __init__(self):
        super(ELM,self).__init__()
        self.relu = nn.ReLU()
        self.alpha1 = nn.init.uniform_(torch.empty(512, 650), a=-1., b=1.)
        self.alpha2 = nn.init.uniform_(torch.empty(650, 150), a=-1., b=1.)
        self.alpha3 = nn.init.uniform_(torch.empty(150, 30), a=-1., b=1.)
        self.alpha4 = nn.init.uniform_(torch.empty(30, 30), a=-1., b=1.)

        self.bias1 = nn.init.uniform_(torch.empty(1,650), a=-1., b=1.)
        self.bias2 = nn.init.uniform_(torch.empty(1,150), a=-1., b=1.)
        self.bias3 = nn.init.uniform_(torch.empty(1,30), a=-1., b=1.)
        self.bias4 = nn.init.uniform_(torch.empty(1,30), a=-1., b=1.)

        self.beta1t = None
        self.beta2t = None
        self.beta3t = None
        self.beta_out = None

    def forward(self,x):
        return (x @ self.beta1t @ self.beta2t @ self.beta3t @ self.beta_out)

    def fit(self,x,y_onehot):
        # ELM-AE 1
        hidden1 = self.relu(x @ self.alpha1 + self.bias1)
        hidden1_t = torch.t(hidden1)
        # hidden nodes > input nodes : 650 > 512
        # ß = inverse(Ht * H) * Ht * X
        self.beta1t = torch.t(torch.pinverse(hidden1_t @ hidden1) @ hidden1_t @ x)

        # ELM-AE 2
        x2 = self.relu(x @ self.beta1t)
        hidden2 = self.relu(x2 @ self.alpha2 + self.bias2)
        hidden2_t = torch.t(hidden2)
        # hidden nodes < input nodes : 150 < 650
        # ß = Ht * inverse(Ht * H) * X - invalid
        self.beta2t = torch.t(torch.pinverse(hidden2_t @ hidden2) @ hidden2_t @ x2)

        # ELM-AE 3
        x3 = self.relu(x2 @ self.beta2t)
        hidden3 = self.relu(x3 @ self.alpha3 + self.bias3)
        hidden3_t = torch.t(hidden3)
        # hidden nodes < input nodes : 30 < 150
        # ß = Ht * inverse(Ht * H) * X - invalid
        self.beta3t = torch.t(torch.pinverse(hidden3_t @ hidden3) @ hidden3_t @  x3)

        # ELM-AE 4 (classifier)
        x4 = self.relu(x3 @ self.beta3t)
        hidden4 = x4
        hidden4_t = torch.t(hidden4)
        # hidden nodes < input nodes : 10 < 30
        # ß = Ht * inverse(Ht * H) * X - invalid
        self.beta_out = torch.pinverse(hidden4_t @ hidden4) @ hidden4_t @  y_onehot

x_train_fan = torch.load('datasets/CWRU/presplit/elm_x_train_fan_end.pt')
x_test_fan = torch.load('datasets/CWRU/presplit/elm_x_test_fan_end.pt')
x_train_drive = torch.load('datasets/CWRU/presplit/elm_x_train_drive_end.pt')
x_test_drive = torch.load('datasets/CWRU/presplit/elm_x_test_drive_end.pt')
y_train_onehot = torch.load('datasets/CWRU/presplit/elm_y_train_onehot.pt')
y_train = torch.load('datasets/CWRU/presplit/elm_y_train.pt')
y_test = torch.load('datasets/CWRU/presplit/elm_y_test.pt')


# pred2 = e(x_train_fan)

mean = []
for i in range(1):
    e = ELM()
    e.fit(x_train_fan,y_train_onehot)
    pred1 = e(x_test_fan)
    mean.append(ELMAccuracy(pred1,y_test))
    print(mean[-1])

# print(np.mean(mean))
# print(np.std(mean))