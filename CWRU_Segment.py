import scipy.io
import pandas as pd
from os import listdir
from os.path import isfile, join

all_files = [f for f in listdir("datasets/CWRU/original")]
N = 30 #segments of size N
labels = {
            'B007': 0, 'B014': 1, 'B021': 2,
            'IR007': 3, 'IR014': 4, 'IR021': 5,
            'normal': 6,
            'OR007@6': 7, 'OR014@6': 8, 'OR021@6': 9
         }
d = {3:'drive_end', 4:'fan_end'}

def segments(l, n):
    def inner():
        for i in range(0, len(l) - (len(l)%n), n):
            yield l[i:i + n]
    return list(inner())

prev_name = all_files[0][:all_files[0].find('_')]
temp_df = pd.DataFrame()

for channel in range(3,5):
    for name in all_files:
        mat_file = scipy.io.loadmat(f'datasets/CWRU/original/{name}') 
        if name == 'normal_2.mat':
            df = pd.DataFrame(mat_file[list(mat_file)[channel+1]]) # 3: Drive End, 4: Fan End
        else:
            df = pd.DataFrame(mat_file[list(mat_file)[channel]])

        name = name[:name.find('_')]
        label = labels[name]

        segmented = segments(df[0].tolist(), N)
        df = pd.DataFrame(segmented)
        df["label"] = label

        if name == prev_name:
            temp_df = temp_df.append(df)

        else:
            temp_df.to_csv(f'datasets/CWRU/segmented/{d[channel]}/segmented_{prev_name}.csv', index=False)
            temp_df = df
        
        prev_name = name
        
    temp_df.to_csv(f'datasets/CWRU/segmented/{d[channel]}/segmented_{prev_name}.csv', index=False)
    temp_df = df