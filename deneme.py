#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
#%%
Sample_Length = 6400
J = 30
# N = (Sample_Length//J) - ((Sample_Length//J)%4)
N = 212
K = 400
channel_count = 1

flat = nn.Flatten()
conv1 = nn.Conv2d(1, 40, (channel_count,1))
conv2 = nn.Conv2d(40,40,(1,2),2)
conv3 = nn.Conv2d(40,40, (1,11),padding=(0,5))
lin1 = nn.Linear(N*20,N*10)
lin2 = nn.Linear(N*10,K)
lin3 = nn.Linear(K,N*10)
lin4 = nn.Linear(N*10,N*20)
tconv1 = nn.ConvTranspose2d(40, 1, (channel_count,1))
tconv2 = nn.ConvTranspose2d(40,40,(1,2),2) 
tconv3 = nn.ConvTranspose2d(40,40,(1,11),padding=(0,5))

x = torch.randn(1,1,1,212)
e1 = conv3(conv2(conv1(x)))
e2 = lin1(flat(e1))
e3 = lin2(e2)
e4 = torch.reshape(lin4(lin3(e3)),(-1,40,1,(N//2)))
e5 = tconv1(tconv2(tconv3(e4)))

print(f'e1 shape: {e1.shape}')
print(f'e2 shape: {e2.shape}')
print(f'e3 shape: {e3.shape}')
print(f'e4 shape: {e4.shape}')
print(f'e5 shape: {e5.shape}')