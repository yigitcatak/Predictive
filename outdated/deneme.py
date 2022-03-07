#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.randn(1,120,1,20)
#%%
l1 = nn.Conv2d(120,120,(1,11),padding=(0,5))
b = l1(a)
print(b.shape)
#%%
l3 = nn.Flatten()
b = l3(b)
print(b.shape)
l4 = nn.Linear(2400,100)
b = l4(b)
print(b.shape)
l5 = nn.Linear(100,2400)
b = l5(b)
print(b.shape)
b = torch.reshape(b,(1,120,1,20))
print(b.shape)
#%%
l2 = nn.ConvTranspose2d(120,120,(1,11),padding=(0,5))
c = l2(b)
print(c.shape)