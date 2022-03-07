from PREDICTIVE_DEFINITIONS import *

Sample_Length = 1200 #0.1 sec
J = 30
N = Sample_Length//J

FanEnd = [f for f in listdir('datasets/CWRU/segmented/fan_end')]
DriveEnd = [f for f in listdir('datasets/CWRU/segmented/drive_end')]
Class_Weights = dict(zip(list(range(10)),list(0 for i in range(10))))
Seed = randint(0,1e6)

x_mixed_train = []
x_mixed_val = []
x_mixed_test = []
y_mixed_train = []
y_mixed_val = []
y_mixed_test = []
x_mixed_train2 = []
x_mixed_val2 = []
x_mixed_test2 = []

# READ FAN END
for name in FanEnd:
    df = pd.read_csv(f'datasets/CWRU/segmented/fan_end/{name}')

    # discard segments that does not fill a sample
    data = df.drop(['label'], axis=1).values.tolist()[:len(df)-len(df)%J]
    # get every Jth label as the labels are given with the average of J segments to classifier
    label = df['label'].values.tolist()[:len(data):J]
    # spare first x% for training
    idx = int(len(label)*0.05)
    idx2 = int(len(label)*0.1)
    train = data[0:idx*J]
    val = data[idx*J:idx2*J]
    test = data[idx2*J:]
    y_train = label[0:idx]
    y_val = label[idx:idx2]
    y_test = label[idx2:]
    Class_Weights[y_train[0]] += len(y_train)
    # group segments so that sample integrity is kept during the shuffling
    train = Batch(train,J)
    train = list(zip(train,y_train))
    x_mixed_train += train
    x_mixed_val += val
    x_mixed_test += test
    y_mixed_val += y_val
    y_mixed_test += y_test

# READ DRIVE END
for name in DriveEnd:
    df = pd.read_csv(f'datasets/CWRU/segmented/drive_end/{name}')

    # discard segments that does not fill a sample
    data = df.drop(['label'], axis=1).values.tolist()[:len(df)-len(df)%J]
    # get every Jth label as the labels are given with the average of J segments to classifier
    label = df['label'].values.tolist()[:len(data):J]

    # spare first x% for training
    idx = int(len(label)*0.05)
    idx2 = int(len(label)*0.1)
    train = data[0:idx*J]
    val = data[idx*J:idx2*J]
    test = data[idx2*J:]

    # group segments so that sample integrity is kept during the shuffling
    train = Batch(train,J)

    x_mixed_train2 += train
    x_mixed_val2 += val
    x_mixed_test2 += test

Random(Seed).shuffle(x_mixed_train2)
x_mixed_train2 = Flatten(x_mixed_train2)

Random(Seed).shuffle(x_mixed_train)
x_mixed_train, y_mixed_train = zip(*x_mixed_train)
x_mixed_train = Flatten(x_mixed_train)

x_mixed_train, mean, eigen_vecs, diagonal_mat = Whiten(x_mixed_train)
x_mixed_test = Whiten(x_mixed_test, mean, eigen_vecs, diagonal_mat)
x_mixed_val = Whiten(x_mixed_val, mean, eigen_vecs, diagonal_mat)

x_mixed_train2, mean2, eigen_vecs2, diagonal_mat2 = Whiten(x_mixed_train2)
x_mixed_test2 = Whiten(x_mixed_test2, mean2, eigen_vecs2, diagonal_mat2)
x_mixed_val2 = Whiten(x_mixed_val2, mean2, eigen_vecs2, diagonal_mat2)

x_mixed_train = torch.tensor(x_mixed_train,dtype=torch.float32)
x_mixed_val = torch.tensor(x_mixed_val,dtype=torch.float32)
x_mixed_test = torch.tensor(x_mixed_test,dtype=torch.float32)
x_mixed_train2 = torch.tensor(x_mixed_train2,dtype=torch.float32)
x_mixed_val2 = torch.tensor(x_mixed_val2,dtype=torch.float32)
x_mixed_test2 = torch.tensor(x_mixed_test2,dtype=torch.float32)
y_train = torch.tensor(y_mixed_train,dtype=torch.long)
y_val = torch.tensor(y_mixed_val,dtype=torch.long)
y_test = torch.tensor(y_mixed_test,dtype=torch.long)

Class_Weights = torch.tensor(list(Class_Weights.values()),dtype=torch.float32)
Class_Weights = 1/Class_Weights
Class_Weights = Class_Weights/Class_Weights.sum()

torch.save(Class_Weights,'datasets/CWRU/presplit/class_weights.pt')
torch.save(x_mixed_train,'datasets/CWRU/presplit/x_train_fan_end.pt')
torch.save(x_mixed_val,'datasets/CWRU/presplit/x_val_fan_end.pt')
torch.save(x_mixed_test,'datasets/CWRU/presplit/x_test_fan_end.pt')
torch.save(x_mixed_train2,'datasets/CWRU/presplit/x_train_drive_end.pt')
torch.save(x_mixed_val2,'datasets/CWRU/presplit/x_val_drive_end.pt')
torch.save(x_mixed_test2,'datasets/CWRU/presplit/x_test_drive_end.pt')
torch.save(y_train,'datasets/CWRU/presplit/y_train.pt')
torch.save(y_val,'datasets/CWRU/presplit/y_val.pt')
torch.save(y_test,'datasets/CWRU/presplit/y_test.pt')