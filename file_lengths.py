#%%
import pandas as pd
from os import listdir
from os.path import isfile, join

all_files = [f for f in listdir("dataset/segmented")]

i = 0
count = 0
lens = []
for name in all_files:
    df = pd.read_csv(f'dataset/segmented/{name}')
    count += df.shape[0]
    i += 1
    if i == 4:
        name = name[name.find('_')+1:]
        name = name[:name.find('_')]
        lens.append([name,count])
        i = 0
        count = 0

lens = pd.DataFrame(lens)
lens.to_csv('file_lengths.csv',index=False)