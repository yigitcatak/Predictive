#%%
from PREDICTIVE_DEFINITIONS import *

all_files = [f for f in listdir('datasets/Paderborn/original')]
all_types = ['K001','K002','K003','K004','K005','KA04','KA15','KA16','KA22','KA30','KI04','KI14','KI16','KI18','KI21']
Sample_Length = 6400
J = 15
N = (Sample_Length//J) - ((Sample_Length//J)%4)

temp_df = pd.DataFrame()

for ftype in all_types:
    count = 0
    for name in all_files:
        if ftype == name[12:16]:
            count+=1
            mat_file = scipy.io.loadmat(f'datasets/Paderborn/original/{name}') 

             # (3. index) 2: Y cell'i
             # (5. index) 6: Vibration
             # (6. index) 2: Data'nin kendisini secmek icin
            df = pd.DataFrame(mat_file[list(mat_file)[-1]][0][0][2][0][6][2][0])
            name = name[12:16]
            if 'K0' in name:
                label = 0
            elif 'KI' in name:
                label = 1
            elif 'KA' in name:
                label = 2
            # else:
            #     label = 1

            segmented = Batch(df[0].tolist(), N)
            df = pd.DataFrame(segmented)
            df['label'] = label
            temp_df = pd.concat([temp_df,df])

            if count == 80:
                temp_df.to_csv(f'datasets/Paderborn/segmented/vibration/segmented_{ftype}.csv', index=False)
                temp_df = pd.DataFrame()
                break
