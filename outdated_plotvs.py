#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Ks = [0.01,0.05,0.1,0.15,0.2,0.25,0.3]

NSAELCNAccs = pd.read_csv('nsaelcn.csv').values.reshape(-1)
Arxiv1Accs = pd.read_csv('arxiv_single.csv').values.reshape(-1)
Arxiv2Accs = pd.read_csv('arxiv_multi.csv').values.reshape(-1)

plt.figure(figsize=(5,4))
plt.plot(range(1,8),NSAELCNAccs,label='NSO-YBA')
plt.plot(range(1,8),Arxiv1Accs,label='Geliştirilen Tek Kanallı Yöntem')
plt.plot(range(1,8),Arxiv2Accs,label='Geliştirilen Çift Kanallı Yöntem')
plt.xlim(0,9)
plt.xticks(range(1,8),Ks)
plt.xlabel('Eğitim Boyutu')
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

