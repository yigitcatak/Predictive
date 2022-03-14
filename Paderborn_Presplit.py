#%%
from PREDICTIVE_DEFINITIONS import *

Sample_Length = 6400 #0.1 sec
J = 30
N = (Sample_Length//J) - ((Sample_Length//J)%4)

Vibration = [f for f in listdir('datasets/Paderborn/segmented/vibration')]
# Current1 = [f for f in listdir('datasets/Paderborn/segmented/current1')]
# Current2 = [f for f in listdir('datasets/Paderborn/segmented/current2')]

Class_Weights = dict(zip(list(range(3)),list(0 for i in range(3))))
Seed = randint(0,1e6)

x_mixed_train = []
x_mixed_test = []
x_mixed_train2 = []
x_mixed_test2 = []
x_mixed_train3 = []
x_mixed_test3 = []
y_mixed_train = []
y_mixed_test = []

Healthy = ['K001','K002','K003','K004','K005']
Inner = ['KI04','KI14','KI16','KI18','KI21']
Outter = ['KA04','KA15','KA16','KA22','KA30']

train_names = RandomCombination(Healthy,3) + RandomCombination(Inner,3) + RandomCombination(Outter,3)

for name in Vibration:
    df = pd.read_csv(f'datasets/Paderborn/segmented/vibration/{name}')
    name = name[-8:-4]

    # discard segments that does not fill a sample
    data = df.drop(['label'], axis=1).values.tolist()[:len(df)-len(df)%J]
    # get every Jth label as the labels are given with the average of J segments to classifier
    label = df['label'].values.tolist()[:len(data):J]

    if name in train_names:
        train = data
        y_train = label
        train = Batch(train,J)
        train = list(zip(train,y_train))
        x_mixed_train += train
        Class_Weights[label[0]] += len(train)

    else:
        x_mixed_test += data
        y_mixed_test += label

# for name in Current1:
#     df = pd.read_csv(f'datasets/Paderborn/segmented/current1/{name}')
#     name = name[-8:-4]

#     # discard segments that does not fill a sample
#     data = df.drop(['label'], axis=1).values.tolist()[:len(df)-len(df)%J]

#     if name in train_names:
#         train = data
#         train = Batch(train,J)
#         x_mixed_train2 += train

#     else:
#         x_mixed_test2 += data

# for name in Current2:
#     df = pd.read_csv(f'datasets/Paderborn/segmented/current2/{name}')
#     name = name[-8:-4]

#     # discard segments that does not fill a sample
#     data = df.drop(['label'], axis=1).values.tolist()[:len(df)-len(df)%J]

#     if name in train_names:
#         train = data
#         train = Batch(train,J)
#         x_mixed_train3 += train

#     else:
#         x_mixed_test3 += data

Random(Seed).shuffle(x_mixed_train)
x_mixed_train, y_mixed_train = zip(*x_mixed_train)
x_mixed_train = Flatten(x_mixed_train)

# Random(Seed).shuffle(x_mixed_train2)
# x_mixed_train2 = Flatten(x_mixed_train2)

# Random(Seed).shuffle(x_mixed_train3)
# x_mixed_train3 = Flatten(x_mixed_train3)

x_mixed_train, mean, eigen_vecs, diagonal_mat = Whiten(x_mixed_train)
x_mixed_test = Whiten(x_mixed_test, mean, eigen_vecs, diagonal_mat)

# x_mixed_train2, mean, eigen_vecs, diagonal_mat = Whiten(x_mixed_train2)
# x_mixed_test2 = Whiten(x_mixed_test2, mean, eigen_vecs, diagonal_mat)

# x_mixed_train3, mean, eigen_vecs, diagonal_mat = Whiten(x_mixed_train3)
# x_mixed_test3 = Whiten(x_mixed_test3, mean, eigen_vecs, diagonal_mat)

x_mixed_train = torch.tensor(x_mixed_train,dtype=torch.float32)
x_mixed_test = torch.tensor(x_mixed_test,dtype=torch.float32)
# x_mixed_train2 = torch.tensor(x_mixed_train2,dtype=torch.float32)
# x_mixed_test2 = torch.tensor(x_mixed_test2,dtype=torch.float32)
# x_mixed_train3 = torch.tensor(x_mixed_train3,dtype=torch.float32)
# x_mixed_test3 = torch.tensor(x_mixed_test3,dtype=torch.float32)
y_train = torch.tensor(y_mixed_train,dtype=torch.long)
y_test = torch.tensor(y_mixed_test,dtype=torch.long)

Class_Weights = torch.tensor(list(Class_Weights.values()),dtype=torch.float32)
Class_Weights = 1/Class_Weights
Class_Weights = Class_Weights/Class_Weights.sum()

torch.save(Class_Weights,'datasets/Paderborn/presplit/class_weights.pt')
torch.save(x_mixed_train,'datasets/Paderborn/presplit/x_train_vibration.pt')
torch.save(x_mixed_test,'datasets/Paderborn/presplit/x_test_vibration.pt')
# torch.save(x_mixed_train2,'datasets/Paderborn/presplit/x_train_current1.pt')
# torch.save(x_mixed_test2,'datasets/Paderborn/presplit/x_test_current1.pt')
# torch.save(x_mixed_train3,'datasets/Paderborn/presplit/x_train_current2.pt')
# torch.save(x_mixed_test3,'datasets/Paderborn/presplit/x_test_current2.pt')
torch.save(y_train,'datasets/Paderborn/presplit/y_train.pt')
torch.save(y_test,'datasets/Paderborn/presplit/y_test.pt')