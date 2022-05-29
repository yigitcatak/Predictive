#%%
from PREDICTIVE_DEFINITIONS import *

ALL_FILES = [f for f in listdir("datasets/CWRU/original")]
LABELS = {
            'B007': 0, 'B014': 1, 'B021': 2,
            'IR007': 3, 'IR014': 4, 'IR021': 5,
            'normal': 6,
            'OR007@6': 7, 'OR014@6': 8, 'OR021@6': 9
         }
CHANNEL = {3:'drive_end', 4:'fan_end'}
TRAIN_SIZE = 0.5

y_mixed_train = np.array([])
y_mixed_test = np.array([])
for channel in [3,4]:
    x_mixed_train = np.empty([0,512])
    x_mixed_test = np.empty([0,512])
    for name in ALL_FILES:
        mat_file = scipy.io.loadmat(f'datasets/CWRU/original/{name}')
        if name == 'normal_2.mat':
            data = mat_file[list(mat_file)[channel+1]].flatten() # normal_2.mat has a wrong convention
        else:
            data = mat_file[list(mat_file)[channel]].flatten() # 3: Drive End, 4: Fan End

        name = name[:name.find('.')]
        labelname = name[:name.find('_')]
        label = LABELS[labelname]

        data = data[:len(data)-(len(data)%1024)]
        segmented = data.reshape(-1,1024)
        segmented = np.fft.rfft(segmented)
        segmented = np.delete(segmented, np.s_[-1:], axis=1) # there are 513 coeff, drop last one to make it 512
        idx = int(len(segmented)*TRAIN_SIZE)

        train = segmented[:idx]
        test = segmented[idx:]

        if channel == 3: # generate labels on only one channel (other one is same)
            y_train = [label for i in range(len(train))]
            y_test = [label for i in range(len(test))]
            y_mixed_train = np.concatenate([y_mixed_train, y_train])
            y_mixed_test = np.concatenate([y_mixed_test, y_test])

        x_mixed_train = np.concatenate([x_mixed_train, train])
        x_mixed_test = np.concatenate([x_mixed_test, test])

    x_mixed_train = torch.tensor(x_mixed_train,dtype=torch.float32)
    x_mixed_test = torch.tensor(x_mixed_test,dtype=torch.float32)

    if channel == 3:
        y_train = torch.tensor(y_mixed_train,dtype=torch.long)
        y_test = torch.tensor(y_mixed_test,dtype=torch.long)
        y_train_onehot = torch.tensor(F.one_hot(y_train, num_classes=10),dtype=torch.float32)
        torch.save(x_mixed_train,'datasets/CWRU/presplit/elm_x_train_fan_end.pt')
        torch.save(x_mixed_test,'datasets/CWRU/presplit/elm_x_test_fan_end.pt')
        torch.save(y_train_onehot,'datasets/CWRU/presplit/elm_y_train_onehot.pt')
        torch.save(y_train,'datasets/CWRU/presplit/elm_y_train.pt')
        torch.save(y_test,'datasets/CWRU/presplit/elm_y_test.pt')
    else:
        torch.save(x_mixed_train,'datasets/CWRU/presplit/elm_x_train_drive_end.pt')
        torch.save(x_mixed_test,'datasets/CWRU/presplit/elm_x_test_drive_end.pt')