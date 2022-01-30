import torch
import torch.nn as nn
import torch.nn.functional as F

decoder_out = torch.randn(1,40,1,20)
Channel_Count = 2

tconv1 = nn.ConvTranspose2d(40, 1, (Channel_Count,1))
tconv2 = nn.ConvTranspose2d(40,40,(1,Channel_Count),2) if Channel_Count == 2 else nn.ConvTranspose2d(40,40,(1,Channel_Count),2,output_padding=(0,1)) 
tconv3 = nn.ConvTranspose2d(40,40,(1,11),padding=(0,5))
out = tconv3(decoder_out)
print(out.shape)
out = tconv2(out)
print(out.shape)
out = tconv1(out)
print(out.shape)