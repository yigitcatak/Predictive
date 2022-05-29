#%%
# CWRU TRANSFER DATA PREPERATION
from PREDICTIVE_DEFINITIONS import *

ALL_FILES = [f for f in listdir("datasets/CWRU/original")]
N, J = Settings('CWRU')
# LABELS = {
#             'B007': 0, 'B014': 1, 'B021': 2,
#             'IR007': 3, 'IR014': 4, 'IR021': 5,
#             'normal': 6,
#             'OR007@6': 7, 'OR014@6': 8, 'OR021@6': 9
#          }
LABELS = {
            'B007': 1, 'B014': 1, 'B021': 1,
            'IR007': 1, 'IR014': 1, 'IR021': 1,
            'normal': 0,
            'OR007@6': 1, 'OR014@6': 1, 'OR021@6': 1
         }
CLASS_COUNT = 2
CHANNEL = {3:'drive_end', 4:'fan_end'}

y_mixed_test = []

mean = torch.load('saves/Paderborn_mean.pt')
whitening_mat = torch.load('saves/Paderborn_matrix.pt')
for channel in [3,4]:
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

        if 'B' in name:
            continue

        segmented = Batch(data, N)
        segmented = segmented[:len(segmented)-(len(segmented)%J)]

        test = segmented
        y_test = [label for i in range(0,len(test),J)]

        if channel == 3: # generate labels on only one channel (other one is same)
            y_mixed_test += y_test

        x_mixed_test += test
        
    x_mixed_test = Whiten(x_mixed_test, mean, whitening_mat)
    # x_mixed_test,_,_ = Whiten(x_mixed_test)

    x_mixed_test = torch.tensor(x_mixed_test,dtype=torch.float32)
    y_test = torch.tensor(y_mixed_test,dtype=torch.long)

    if channel == 3:
        torch.save(x_mixed_test,'datasets/CWRU/presplit/x_test_transfer_fan_end.pt')
        torch.save(y_test,'datasets/CWRU/presplit/y_test_transfer.pt')
    else:
        torch.save(x_mixed_test,'datasets/CWRU/presplit/x_test_transfer_drive_end.pt')