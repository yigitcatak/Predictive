#%%
from PREDICTIVE_DEFINITIONS import *
N, J = Settings('CWRU')
data = np.array(pd.read_csv('datasets/Cozum/original/vibration_datas.csv')['Data']*(1e-10))
segmented = Batch(data, N)
segmented = segmented[:len(segmented)-(len(segmented)%J)]