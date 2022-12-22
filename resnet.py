import torch
import torch.nn as nn


class ResNet(nn.Module):
    """ Slightly modified version of ResNet (https://arxiv.org/abs/1512.03385). """

    def __init__(self, n_in=3, n_out=200, ch=[32,64,128], activation=nn.ELU()):
        super().__init__()
        ch.insert(0, n_in)
        layers = []
        for i in range(1, len(ch)):
            chan_conv = nn.Conv2d(ch[i-1], ch[i], kernel_size=1, padding=1)
            res_layer = ResidualLayer(ch[i], kernel_size=5)
            layers.extend([chan_conv, res_layer, nn.MaxPool2d(2,2), activation])
        layers.append(nn.Conv2d(ch[-1], n_out, kernel_size=5, padding=1))
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze()

class ResidualLayer(nn.Module):

    def __init__(self, ch, kernel_size, activation=nn.ELU()):
        super().__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=kernel_size, padding=2)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=kernel_size, padding=2)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.activation(x1)
        res = self.conv2(x1)
        return x + res