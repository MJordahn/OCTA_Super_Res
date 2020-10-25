import torch
import math
from torch import nn

# Discriminator class:
class SRResNetDiscriminator(nn.Module):
    def __init__(self, conv_dim=64):
        super(SRResNetDiscriminator, self).__init__()
        self.l1 = nn.Sequential(
                  nn.Conv2d(1, 64, kernel_size=4,  stride=2, padding=2), #out: 48 x 48
                  nn.LeakyReLU(0.2),
                  )
        self.l2 = nn.Sequential(
                  nn.Conv2d(64, 64*2, kernel_size=4, stride=2, padding=2), #out: 24 x 24
                  nn.BatchNorm2d(128),
                  nn.LeakyReLU(0.2),
                  )
        self.l3 = nn.Sequential(
                  nn.Conv2d(64*2, 64*4, kernel_size=4, padding=2),  #out: 12 x 12
                  nn.BatchNorm2d(256),
                  nn.LeakyReLU(0.2),
                  )
        self.l4 = nn.Sequential(
                  nn.Conv2d(64*4, 64*8, kernel_size=4, stride=2, padding=2),  #out: 6x6
                  nn.BatchNorm2d(512),
                  nn.LeakyReLU(0.2),
                  )
        self.out1 = nn.Conv2d(64*1, 1, kernel_size=4, stride=1, padding=2)
        self.out2 = nn.Conv2d(64*2, 1, kernel_size=4, stride=1, padding=2)
        self.out3 = nn.Conv2d(64*4, 1, kernel_size=4, stride=1, padding=2)
        self.out4 = nn.Conv2d(64*8, 1, kernel_size=4, stride=1, padding=2)
    def forward(self, inpt):
        out1 = self.l1(inpt)
        out2 = self.l2(out1)
        out3 = self.l3(out2)
        out4 = self.l4(out3)
        out_a_1 = self.out1(out1)
        out_a_2 = self.out2(out2)
        out_a_3 = self.out3(out3)
        out_a_4 = self.out4(out4)
        return [out_a_1, out_a_2, out_a_3, out_a_4]

# Generator class
class SRResNet(nn.Module):
    def __init__(self, scale_factor, n_res_blocks=8):
        super(SRResNet, self).__init__()
        # Input layer
        self.pre_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, padding=4, stride=1),
            nn.PReLU()
        )
        # ResNet layers
        res_blocks = [ResidualBlock(64) for _ in range(n_res_blocks)]
        self.res_blocks = nn.Sequential(*res_blocks)
        self.mid_block = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        # Upsample layers
        upsample_block_num = int(math.log(scale_factor, 2))
        block = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        self.upsample_block = nn.Sequential(*block)
        self.post_block = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=9, padding=4, stride=1)

    def forward(self, x):
        pre_block = self.pre_block(x)
        res_blocks = self.res_blocks(pre_block)
        mid_block = self.mid_block(res_blocks)
        upsample_block = self.upsample_block(pre_block + mid_block)
        post_block = self.post_block(upsample_block)
        return (torch.tanh(post_block) + 1)*0.5

# Res block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1) # could modify to use reflection padding
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1) # could modify to use reflection padding
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + residual

# Upsample block
class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale, kernel_size=3, padding=1)
        self.pixel_shuffle =  PixelShuffle1D(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

# Implementation of the pixel shuffle
class PixelShuffle1D(torch.nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_depth = x.shape[1]
        short_height = x.shape[2]
        short_width = x.shape[3]

        long_channel_depth = short_channel_depth // self.upscale_factor
        long_height = self.upscale_factor * short_height

        x = x.contiguous().view([batch_size, self.upscale_factor, long_channel_depth, short_height, short_width])
        x = x.permute(0, 2, 3, 1, 4).contiguous()
        x = x.view(batch_size, long_channel_depth, long_height, short_width)

        return x

# Creates transformation matrix
def transformationMatrix(dim, scale_factor, batch_size, GPU=True):
    t = torch.zeros([batch_size, 1, int(dim/scale_factor), dim], dtype=float)
    for i in range(int(dim/scale_factor)):
        t[:,:, i, i*scale_factor] = 1
    if GPU:
        t = t.to(torch.device("cuda:0")).float()
    return t
