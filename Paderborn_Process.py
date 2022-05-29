#%%
# PADERBORN DATA PREPEARATION
from PREDICTIVE_DEFINITIONS import *

ALL_FILES = [f for f in listdir('datasets/Paderborn/original')]
ALL_TYPES = ['K001','K002','K003','K004','K005','KA04','KA15','KA16','KA22','KA30','KI04','KI14','KI16','KI18','KI21']
HEALTHY = ['K001','K002','K003','K004','K005']
INNER = ['KI04','KI14','KI16','KI18','KI21']
OUTTER = ['KA04','KA15','KA16','KA22','KA30']
TRAIN_NAMES = RandomCombination(HEALTHY,3) + RandomCombination(INNER,3) + RandomCombination(OUTTER,3)
N, J = Settings('CWRU')
SEED = randint(0,1e6)
CLASS_COUNT = 2
TRAIN_SIZE = 1

weights = dict(zip(list(range(CLASS_COUNT)),list(0 for i in range(CLASS_COUNT))))
y_mixed_train = []
y_mixed_test = []
x_mixed_train = []
x_mixed_test = []

for name in ALL_FILES:
    mat_file = scipy.io.loadmat(f'datasets/Paderborn/original/{name}') 

        # (3. index) 2: Y cell'i
        # (5. index) 6: Vibration
        # (6. index) 2: Data'nin kendisini secmek icin
    data = mat_file[list(mat_file)[-1]][0][0][2][0][6][2][0]
    name = name[12:16]
    labelname = name[:2]
    if 'K0' == labelname:
        label = 0
    else:
        label = 1

    data = data[::5] # downsample
    segmented = Batch(data, N)
    segmented = segmented[:len(segmented)-(len(segmented)%J)]    

    if name in TRAIN_NAMES:
        idx = int(len(segmented)*TRAIN_SIZE - (len(segmented)*TRAIN_SIZE)%J)
        segmented = segmented[:idx]
        y = [label for i in range(0,len(segmented),J)]
        segmented = Batch(segmented,J)
        segmented = list(zip(segmented,y))
        x_mixed_train += segmented
        weights[label] += len(segmented)
    
    else:
        y = [label for i in range(0,len(segmented),J)]
        x_mixed_test += segmented
        y_mixed_test += y

Random(SEED).shuffle(x_mixed_train)
x_mixed_train, y_mixed_train = zip(*x_mixed_train)
x_mixed_train = Flatten(x_mixed_train)
x_mixed_train, mean, whitening_mat = Whiten(x_mixed_train)
x_mixed_test = Whiten(x_mixed_test, mean, whitening_mat)

x_mixed_train = torch.tensor(x_mixed_train,dtype=torch.float32)
x_mixed_test = torch.tensor(x_mixed_test,dtype=torch.float32)
y_train = torch.tensor(y_mixed_train,dtype=torch.long)
y_test = torch.tensor(y_mixed_test,dtype=torch.long)

weights = torch.tensor(list(weights.values()),dtype=torch.float32)
weights = 1/weights
weights = weights/weights.sum()
torch.save(weights,'datasets/Paderborn/presplit/class_weights.pt')
torch.save(x_mixed_train,'datasets/Paderborn/presplit/x_train_vibration.pt')
torch.save(x_mixed_test,'datasets/Paderborn/presplit/x_test_vibration.pt')
torch.save(y_train,'datasets/Paderborn/presplit/y_train.pt')
torch.save(y_test,'datasets/Paderborn/presplit/y_test.pt')
torch.save(mean,'saves/Paderborn_mean.pt')
torch.save(whitening_mat,'saves/Paderborn_matrix.pt')

with open('saves/Paderborn_Train_Names.txt','w') as namesfile:
    for name in TRAIN_NAMES: 
        namesfile.write(name + '\n')