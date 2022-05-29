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