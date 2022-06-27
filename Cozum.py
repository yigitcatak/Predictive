#%%
from PREDICTIVE_DEFINITIONS import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASS_COUNT = 2
N, J = Settings('CWRU') 
K = 200

df = pd.read_csv('datasets/Cozum/original/motor_xyz.csv')
df2 = pd.read_csv('datasets/Cozum/original/motor_xyz_saglkli_5hz.csv')

df.drop(['X ','Y ','Z '], axis=1, inplace=True)
df.columns = ['X','Y','Z']
df = df - df.iloc[0]
df2.drop(['X ','Y ','Z '], axis=1, inplace=True)
df2.columns = ['X','Y','Z']
df2 - df2.iloc[0]

df = df[:len(df)-(len(df)%(N*J))]
df2 = df2[:len(df2)-(len(df2)%(N*J))]

len1 = len(df)//(N*J)
len2 = len(df2)//(N*J)

data = np.array(pd.concat((df,df2))['Z'])

mean = torch.load('saves/Paderborn_mean.pt')
whitening_mat = torch.load('saves/Paderborn_matrix.pt')

segmented = Batch(data, N)

x_test = segmented[:len(segmented)-(len(segmented)%J)]

y_test = [1 for i in range(len1)] + [0 for i in range(len2)]

x_test =  Whiten(x_test, mean, whitening_mat)

x_test = torch.tensor(x_test,dtype=torch.float32)
y_test = torch.tensor(y_test,dtype=torch.long)

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