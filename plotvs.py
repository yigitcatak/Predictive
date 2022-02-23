#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
Ks = [20,40,80,100,200,300,400]

NSAELCNAccs = pd.read_csv('CWRU_NSAELCN_Accs.csv').values.reshape(-1)
Arxiv1Accs = pd.read_csv('CWRU_Arxiv_Accs_Channel1.csv').values.reshape(-1)
Arxiv2Accs = pd.read_csv('CWRU_Arxiv_Accs_Channel2.csv').values.reshape(-1)

NSAELCNStds = pd.read_csv('CWRU_NSAELCN_Stds.csv').values.reshape(-1)
Arxiv1Stds = pd.read_csv('CWRU_Arxiv_Stds_Channel1.csv').values.reshape(-1)
Arxiv2Stds = pd.read_csv('CWRU_Arxiv_Stds_Channel2.csv').values.reshape(-1)

plt.figure(figsize=(11,7))
plt.errorbar(range(1,8),NSAELCNAccs,NSAELCNStds,label='NSO-YBA')
plt.errorbar(range(1,8),Arxiv1Accs,Arxiv1Stds,label='Geliştirilen Tek Kanallı Yöntem')
plt.errorbar(range(1,8),Arxiv2Accs,Arxiv2Stds,label='Geliştirilen Çift Kanallı Yöntem')
plt.xlim(0,8)
plt.xticks(range(1,8),Ks)
plt.xlabel('Öznitelik Boyutu')
plt.ylabel('Sınıflandırma Doğruluğu')
plt.grid(True,axis='y')
# plt.rcParams.update({'font.size': 22})
plt.legend()
plt.savefig('Comparison.png',bbox_inches='tight')

#%%
# plot signal
# plt.figure(figsize=(30,5))
# plt.plot(df[0:1200])
# plt.title('0.1 Saniyelik Bir Sinyal Örneklemi')
# plt.xlabel('Veri Noktası')
# plt.savefig('tam.png',bbox_inches='tight')

# plt.figure()
# plt.plot(df[0:40])
# plt.savefig('e1.png',bbox_inches='tight')

# plt.figure()
# plt.plot(df[40:80])
# plt.savefig('e2.png',bbox_inches='tight')

# plt.figure()
# plt.plot(df[1160:1201])
# plt.savefig('e3.png',bbox_inches='tight')

