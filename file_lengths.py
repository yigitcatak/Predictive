import pandas as pd
from os import listdir
from os.path import isfile, join

FanEnd = [f for f in listdir("datasets/CWRU/segmented/fan_end")]
DriveEnd = [f for f in listdir("datasets/CWRU/segmented/drive_end")]
lengths = dict(zip(FanEnd,list(0 for i in range(len(FanEnd)))))
lengths2 = dict(zip(DriveEnd,list(0 for i in range(len(DriveEnd)))))

for name in FanEnd:
    df = pd.read_csv(f'datasets/CWRU/segmented/fan_end/{name}')
    lengths[name] = df.shape[0]

for name in DriveEnd:
    df = pd.read_csv(f'datasets/CWRU/segmented/drive_end/{name}')
    lengths2[name] = df.shape[0]

print(lengths)
print(lengths2)