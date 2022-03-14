#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

Sample_Length = 6400
J = 30
N = (Sample_Length//J) - ((Sample_Length//J)%2)
channel_count = 2

tconv1 = nn.ConvTranspose2d(40, 1, (channel_count,1))
tconv2 = nn.ConvTranspose2d(40,40,(1,2),2) 
tconv3 = nn.ConvTranspose2d(40,40,(1,11),padding=(0,5))