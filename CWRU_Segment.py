#%%
import scipy.io
import pandas as pd
from os import listdir
from os.path import isfile, join

all_files = [f for f in listdir("CWRU/original")]
N = 30 #segments of size N
labels = {
            'B007': 0, 'B014': 1, 'B021': 2,
            'IR007': 3, 'IR014': 4, 'IR021': 5,
            'normal': 6,
            'OR007@6': 7, 'OR014@6': 8, 'OR021@6': 9
         }
d = {3:'DriveEnd', 4:'FanEnd'}

def segments(l, n):
    def inner():
        for i in range(0, len(l) - (len(l)%n), n):
            yield l[i:i + n]
    return list(inner())
#%%
for channel in range(3,5):
    for name in all_files:
        mat_file = scipy.io.loadmat(f'CWRU/original/{name}') 
        if name == 'normal_2.mat':
            df = pd.DataFrame(mat_file[list(mat_file)[channel+1]]) # 3: Drive End, 4: Fan End
        else:
            df = pd.DataFrame(mat_file[list(mat_file)[channel]])

        label = name[:name.find('_')]
        label = labels[label]
        name = name[:name.find('.')]

        segmented = segments(df[0].tolist(), N)
        df = pd.DataFrame(segmented)
        df["label"] = label
        df.to_csv(f'CWRU/segmented/{d[channel]}/segmented_{name}.csv', index=False)