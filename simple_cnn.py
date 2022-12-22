import torch
import torch.nn as nn


class SimpleCNN(nn.Module):

    def __init__(self, n_in=3, n_out=200, ch=[32,32,64,64,128,128], activation=nn.ELU()):
        super().__init__()
        ch.insert(0, n_in)
        layers = []
        for i in range(1, len(ch)):
            conv = nn.Conv2d(ch[i-1], ch[i], kernel_size=5, padding=1)
            layers.append(conv)
            if i < len(ch)-1 and (i == 0 or ch[i] > ch[i-1]):
                layers.extend([nn.MaxPool2d(2,2), activation])
        layers.append(nn.Conv2d(ch[-1], n_out, kernel_size=5, padding=1))
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze()