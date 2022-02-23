#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
#%%
decoder_out = torch.randn(1,40,1,20)
Channel_Count = 2

tconv1 = nn.ConvTranspose2d(40, 1, (Channel_Count,1))
tconv2 = nn.ConvTranspose2d(40,40,(1,2),2)
tconv3 = nn.ConvTranspose2d(40,40,(1,11),padding=(0,5))
out = tconv3(decoder_out)
print(out.shape)
out = tconv2(out)
print(out.shape)
out = tconv1(out)
print(out.shape)
#%%
networkin = torch.randn(1,1,2,40)
Channel_Count = 2
conv1 = nn.Conv2d(1, 40, (Channel_Count,1))
conv2 = nn.Conv2d(40,40,(1,Channel_Count),2)
conv3 = nn.Conv2d(40,40, (1,11),padding=(0,5))

a = conv1(networkin)
b = conv2(a)
c = conv3(b)

print(a.shape)
print(b.shape)
print(c.shape)
