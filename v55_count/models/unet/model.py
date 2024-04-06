# Adapted from:
# https://towardsdatascience.com/cook-your-first-u-net-in-pytorch-b3297a844cf3

import torch
import torch.nn as nn

from einops import rearrange

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, verbose):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.verbose = verbose

    def forward(self, x):
        if self.verbose: print(f"double conv x: {x.shape}")
        x = self.conv1(x)
        if self.verbose: print(f"double conv conv1: {x.shape}")
        x = self.bn1(x)
        if self.verbose: print(f"double conv bn1: {x.shape}")
        x = self.relu(x)
        if self.verbose: print(f"double conv relu: {x.shape}")
        x = self.conv2(x)
        if self.verbose: print(f"double conv conv2: {x.shape}")
        x = self.bn2(x)
        if self.verbose: print(f"double conv bn2: {x.shape}")
        x = self.relu(x)
        if self.verbose: print(f"double conv relu: {x.shape}")
        return x

class Model(nn.Module):
    def __init__(self, in_channels, multipliers, out_channels, verbose, name):
        super(Model, self).__init__()
        self.verbose = verbose

        # Encoder
        #
        # In the encoder, convolutional layers with the Conv2d function are used
        # to extract features from the input image. 
        #
        # Each block in the encoder consists of two convolutional layers
        # followed by a max-pooling layer, with the exception of the last block
        # which does not include a max-pooling layer.

        out1 = in_channels * multipliers[0]
        self.e11 = nn.Conv2d(in_channels, out1, kernel_size=3, padding=1)
        self.e12 = nn.Conv2d(out1, out1, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        out2 = out1 * multipliers[1]
        self.e21 = nn.Conv2d(out1, out2, kernel_size=3, padding=1) 
        self.e22 = nn.Conv2d(out2, out2, kernel_size=3, padding=1) 
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) 

        out3 = out2 * multipliers[2]
        self.e31 = nn.Conv2d(out2, out3, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(out3, out3, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        out4 = out3 * multipliers[3]
        self.e41 = nn.Conv2d(out3, out4, kernel_size=3, padding=1)
        self.e42 = nn.Conv2d(out4, out4, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=5)

        out5 = out4 * multipliers[4]
        self.e51 = nn.Conv2d(out4, out5, kernel_size=3, padding=1)
        self.e52 = nn.Conv2d(out5, out5, kernel_size=3, padding=1)


        # Decoder
        self.upconv1 = nn.ConvTranspose2d(out5, out4, kernel_size=6, stride=2)
        self.d11 = nn.Conv2d(out5, out4, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(out4, out4, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(out4, out3, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(out4, out3, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(out3, out3, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(out3, out2, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(out3, out2, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(out2, out2, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(out2, out1, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(out2, out1, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(out1, out1, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(out1, out_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.verbose: print(f"\nx: {x.shape}")
        x = rearrange(x, 'b c f h w -> b (c f) h w')
        if self.verbose: print(f"rearrange(x, 'b c f h w -> b (c f) h w'): {x.shape}")
        # Encoder
        xe11 = self.relu(self.e11(x))
        if self.verbose: print(f"xe11: {xe11.shape}")
        xe12 = self.relu(self.e12(xe11))
        if self.verbose: print(f"xe12: {xe12.shape}")
        xp1 = self.pool1(xe12)
        if self.verbose: print(f"xp1: {xp1.shape}")

        xe21 = self.relu(self.e21(xp1))
        if self.verbose: print(f"xe21: {xe21.shape}")
        xe22 = self.relu(self.e22(xe21))
        if self.verbose: print(f"xe22: {xe22.shape}")
        xp2 = self.pool2(xe22)
        if self.verbose: print(f"xp2: {xp2.shape}")

        xe31 = self.relu(self.e31(xp2))
        if self.verbose: print(f"xe31: {xe31.shape}")
        xe32 = self.relu(self.e32(xe31))
        if self.verbose: print(f"xe32: {xe32.shape}")
        xp3 = self.pool3(xe32)
        if self.verbose: print(f"xp3: {xp3.shape}")

        xe41 = self.relu(self.e41(xp3))
        if self.verbose: print(f"xe41: {xe41.shape}")
        xe42 = self.relu(self.e42(xe41))
        if self.verbose: print(f"xe42: {xe42.shape}")
        xp4 = self.pool4(xe42)
        if self.verbose: print(f"xp4: {xp4.shape}")

        xe51 = self.relu(self.e51(xp4))
        if self.verbose: print(f"xe51: {xe51.shape}")
        xe52 = self.relu(self.e52(xe51))
        if self.verbose: print(f"xe52: {xe52.shape}")
        
        # Decoder
        xu1 = self.upconv1(xe52)
        if self.verbose: print(f"xu1: {xu1.shape}")
        xu11 = torch.cat([xu1, xe42], dim=1)
        if self.verbose: print(f"xu11: {xu11.shape}")
        xd11 = self.relu(self.d11(xu11))
        if self.verbose: print(f"xd11: {xd11.shape}")
        xd12 = self.relu(self.d12(xd11))
        if self.verbose: print(f"xd12: {xd12.shape}")

        xu2 = self.upconv2(xd12)
        if self.verbose: print(f"xu2: {xu2.shape}")
        xu22 = torch.cat([xu2, xe32], dim=1)
        if self.verbose: print(f"xu22: {xu22.shape}")
        xd21 = self.relu(self.d21(xu22))
        if self.verbose: print(f"xd21: {xd21.shape}")
        xd22 = self.relu(self.d22(xd21))
        if self.verbose: print(f"xd22: {xd22.shape}")

        xu3 = self.upconv3(xd22)
        if self.verbose: print(f"xu3: {xu3.shape}")
        xu33 = torch.cat([xu3, xe22], dim=1)
        if self.verbose: print(f"xu33: {xu33.shape}")
        xd31 = self.relu(self.d31(xu33))
        if self.verbose: print(f"xd31: {xd31.shape}")
        xd32 = self.relu(self.d32(xd31))
        if self.verbose: print(f"xd32: {xd32.shape}")

        xu4 = self.upconv4(xd32)
        if self.verbose: print(f"xu4: {xu4.shape}")
        xu44 = torch.cat([xu4, xe12], dim=1)
        if self.verbose: print(f"xu44: {xu44.shape}")
        xd41 = self.relu(self.d41(xu44))
        if self.verbose: print(f"xd41: {xd41.shape}")
        xd42 = self.relu(self.d42(xd41))
        if self.verbose: print(f"xd42: {xd42.shape}")

        # Output layer
        out = self.outconv(xd42)
        if self.verbose: print(f"out: {out.shape}")

        out = rearrange(out, 'b c h w -> b h w c')
        if self.verbose: print(f"rearrange(out, 'b c h w -> b h w c'): {out.shape}")

        out = self.sigmoid(out)
        return out
