import torch
import torch.nn as nn


def to_pair(x):
    if type(x) == tuple:
        return x
    return (x,x)

class CXBlock(nn.Module):
    """ Inverted bottleneck module of the ConvNext architecture. """
    def __init__(self, dim, h, w):
        super().__init__()
        layers = [
            nn.Conv2d(dim, 4 * dim, kernel_size=7, groups=dim, padding='same'),
            nn.LayerNorm([4 * dim, h, w]),
            nn.Conv2d(4 * dim, 4 * dim, kernel_size=1, padding='same'),
            nn.GELU(),
            nn.Conv2d(4 * dim, dim, kernel_size=1, padding='same'),
        ]
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return x + self.net(x)

class ConvNext(nn.Module):
    """"
    ConvNext from the paper 'A ConvNet for the 2020s', Liu et al., 2022.

    NOTE: Stochastic depth, EMA parameter averaging, and layer scale not implemented.
    """

    def __init__(self, img_size, num_classes, img_ch=3, blocks=(3,3,9,3), channels=(96, 192, 384, 768)):
        super().__init__()
        assert(len(blocks) == len(channels))

        H, W = to_pair(img_size)
        H, W = H//4, W//4 # internal after stem
        assert(H%4 == 0 and W%4 == 0)

        # input stem layer
        self.stem = nn.Sequential(*[
            nn.Conv2d(img_ch, channels[0], kernel_size=4, stride=4),
            nn.LayerNorm([channels[0], H, W])
            ])
        
        # intermediate inverted bottleneck blocks and downsampling
        cx_blocks = []
        for i, (b, ch) in enumerate(zip(blocks, channels)):
             cx_blocks.extend([CXBlock(ch, H//(2**i), W//(2**i)) for j in range(b)])
             if i < len(channels) - 1 : # downsampling
                cx_blocks.append(nn.LayerNorm([ch, H//(2**i), W//(2**i)]))
                cx_blocks.append(nn.Conv2d(ch, channels[i + 1], kernel_size=2, stride=2))
        self.cx_blocks = nn.Sequential(*cx_blocks)

        # output layer
        self.out = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.LayerNorm(channels[-1]),
            nn.Linear(channels[-1], num_classes)
            ])

    def forward(self, x):
        x = self.stem(x)
        x = self.cx_blocks(x)
        x = self.out(x)
        return x