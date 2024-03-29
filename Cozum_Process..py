#%%
from PREDICTIVE_DEFINITIONS import *

N, J = Settings('CWRU')

df = pd.read_csv('datasets/Cozum/original/motor_xyz_saglkli_5hz.csv')

df.drop(['X ','Y ','Z '], axis=1, inplace=True)
df.columns = ['X','Y','Z']

data = np.array(df['X'])

mean = torch.load('saves/Paderborn_mean.pt')
whitening_mat = torch.load('saves/Paderborn_matrix.pt')

label = 0

segmented = Batch(data, N)

segmented = segmented[:len(segmented)-(len(segmented)%J)]

test = segmented

y_test = [label for i in range(0,len(test),J)]

test =  Whiten(test, mean, whitening_mat)

test = torch.tensor(test,dtype=torch.float32)
y_test = torch.tensor(y_test,dtype=torch.long)

torch.save(test,'datasets/Cozum/presplit/x.pt')
torch.save(y_test,'datasets/Cozum/presplit/y.pt')
#%%
# COZUM DATA PREPEARATION
from PREDICTIVE_DEFINITIONS import *
N, J = Settings('CWRU')
data = pd.read_csv('datasets/Cozum/original/fftdata.csv')['Data']
data = np.array([float(i.replace(',', '')) for i in data])
idata = np.fft.irfft(data)
idata = np.concatenate((idata,[0 for i in range(1200-len(idata))]))*(1e-8)

mean = torch.load('saves/Paderborn_mean.pt')
whitening_mat = torch.load('saves/Paderborn_matrix.pt')
label = 0

segmented = Batch(idata, N)
segmented = segmented[:len(segmented)-(len(segmented)%J)]
test = segmented
y_test = [label for i in range(0,len(test),J)]

test =  Whiten(test, mean, whitening_mat)

test = torch.tensor(test,dtype=torch.float32)
y_test = torch.tensor(y_test,dtype=torch.long)

torch.save(test,'datasets/Cozum/presplit/x2.pt')
torch.save(y_test,'datasets/Cozum/presplit/y2.pt')
#%%
# COZUM DATA PREPEARATION
from PREDICTIVE_DEFINITIONS import *
N, J = Settings('CWRU')
data = np.array(pd.read_csv('datasets/Cozum/original/vibration_datas.csv')['Data']*(1e-10))

mean = torch.load('saves/Paderborn_mean.pt')
whitening_mat = torch.load('saves/Paderborn_matrix.pt')
label = 1

segmented = Batch(data, N)
segmented = segmented[:len(segmented)-(len(segmented)%J)]
test = segmented
y_test = [label for i in range(0,len(test),J)]

test =  Whiten(test, mean, whitening_mat)

test = torch.tensor(test,dtype=torch.float32)
y_test = torch.tensor(y_test,dtype=torch.long)

torch.save(test,'datasets/Cozum/presplit/x.pt')
torch.save(y_test,'datasets/Cozum/presplit/y.pt')