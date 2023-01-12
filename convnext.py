import torch
import torch.nn as nn
import numpy as np




class CXBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        layers = [
            nn.Conv2d(dim, 4 * dim, kernel_size=7, groups=dim, padding='same'),
            nn.LayerNorm(),
            nn.ConvConv2d2D(4 * dim, 4 * dim, kernel_size=1, padding='same'),
            nn.GELU(),
            nn.Conv2d(4 * dim, dim, kernel_size=1, padding='same'),
        ]
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return x + self.net(x)

class ConvNext(nn.Module):‚

    def __init__(self, blocks=(3,3,9,3), channels=(96, 192, 384, 768)):
        super().__init__()

        # define layers
        dim_in = channels[0]
        self.stem = nn.Conv2d(dim_in, dim_in, kernel_size=4, stride=4)
        layers = []
        for b, ch in zip(blocks, channels):
            print(b, ch)
            layers.extend([CXBlock(ch) for i in range(b)])
             
            
        # self.blocks = 
        # downlsampling
        # self.avg_pool = 



# class ResidualLayer(nn.Module):

#     def __init__(self, ch, kernel_size, activation=nn.ELU()):
#         super().__init__()
#         self.activation = activation
#         self.conv1 = nn.Conv2d(ch, ch, kernel_size=kernel_size, padding=2)
#         self.conv2 = nn.Conv2d(ch, ch, kernel_size=kernel_size, padding=2)

#     def forward(self, x):
#         x1 = self.conv1(x)
#         x1 = self.activation(x1)
#         res = self.conv2(x1)
#         return x + res

net = ConvNext()