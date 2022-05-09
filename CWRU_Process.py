#%%
# CWRU DATA PREPERATION
from PREDICTIVE_DEFINITIONS import *

ALL_FILES = [f for f in listdir("datasets/CWRU/original")]
N, J = Settings('CWRU')
TRAIN_SIZE = 0.1
LABELS = {
            'B007': 0, 'B014': 1, 'B021': 2,
            'IR007': 3, 'IR014': 4, 'IR021': 5,
            'normal': 6,
            'OR007@6': 7, 'OR014@6': 8, 'OR021@6': 9
         }
CHANNEL = {3:'drive_end', 4:'fan_end'}
SEED = randint(0,1e6)

weights = dict(zip(list(range(len(LABELS))),list(0 for i in range(len(LABELS)))))
y_mixed_train = []
y_mixed_test = []

for channel in [3,4]:
    x_mixed_train = []
    x_mixed_test = []
    for name in ALL_FILES:
        mat_file = scipy.io.loadmat(f'datasets/CWRU/original/{name}')
        if name == 'normal_2.mat':
            data = mat_file[list(mat_file)[channel+1]].flatten() # normal_2.mat has a wrong convention
        else:
            data = mat_file[list(mat_file)[channel]].flatten() # 3: Drive End, 4: Fan End

        name = name[:name.find('.')]
        labelname = name[:name.find('_')]
        label = LABELS[labelname]

        segmented = Batch(data, N)
        segmented = segmented[:len(segmented)-(len(segmented)%J)]
        idx = int(len(segmented)*TRAIN_SIZE - (len(segmented)*TRAIN_SIZE)%J)

        train = segmented[:idx]
        test = segmented[idx:]
        y_train = [label for i in range(0,len(train),J)]
        y_test = [label for i in range(0,len(test),J)]

        train = Batch(train,J)

        if channel == 3: # generate labels on only one channel (other one is same)
            weights[label] += len(y_train)
            train = list(zip(train,y_train))
            y_mixed_test += y_test

        x_mixed_train += train
        x_mixed_test += test
        

    Random(SEED).shuffle(x_mixed_train)
    
    if channel == 3:
        x_mixed_train, y_mixed_train = zip(*x_mixed_train)
    x_mixed_train = Flatten(x_mixed_train)

    x_mixed_train, mean, whitening_mat = Whiten(x_mixed_train)
    x_mixed_test = Whiten(x_mixed_test, mean, whitening_mat)

    x_mixed_train = torch.tensor(x_mixed_train,dtype=torch.float32)
    x_mixed_test = torch.tensor(x_mixed_test,dtype=torch.float32)
    y_train = torch.tensor(y_mixed_train,dtype=torch.long)
    y_test = torch.tensor(y_mixed_test,dtype=torch.long)

    if channel == 3:
        weights = torch.tensor(list(weights.values()),dtype=torch.float32)
        weights = 1/weights
        weights = weights/weights.sum()
        torch.save(weights,'datasets/CWRU/presplit/class_weights.pt')
        torch.save(x_mixed_train,'datasets/CWRU/presplit/x_train_fan_end.pt')
        torch.save(x_mixed_test,'datasets/CWRU/presplit/x_test_fan_end.pt')
        torch.save(y_train,'datasets/CWRU/presplit/y_train.pt')
        torch.save(y_test,'datasets/CWRU/presplit/y_test.pt')
    else:
        torch.save(x_mixed_train,'datasets/CWRU/presplit/x_train_drive_end.pt')
        torch.save(x_mixed_test,'datasets/CWRU/presplit/x_test_drive_end.pt')