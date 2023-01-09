import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt

from vision_transformer import VisionTransformer, ResidualLayer, to_pair


def is_power_of_two(x):
    return (x != 0) and (x & (x - 1)) == 0

class TokenDecoder(nn.Module):

    def __init__(self, dim_in, patch_size, num_patches, channels=3):
        super().__init__()
        self.num_patches = num_patches
        layers = [nn.Conv2d(dim_in, dim_in//2, kernel_size=2, stride=1, padding=1)]
        num_layers = int(np.log2(patch_size)) - 1
        last_ch = dim_in//2
        for i in range(num_layers):
            next_ch = last_ch//2
            if i == (num_layers - 1) or next_ch <= channels:
                next_ch = channels
            #print(f"from {last_ch} channels to {next_ch}")
            ups = nn.Upsample(scale_factor=2., mode='bicubic')
            conv = nn.Conv2d(last_ch, next_ch, kernel_size=5, stride=1, padding='same')
            layers.extend([ups, conv])
            last_ch = next_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, x.shape[-1], 1, 1) # flatten num patches
        x = self.net(x)
        x = x.view(-1, self.num_patches, *x.shape[1:]) # unflatten num patches
        return x


class PatchSwapVit(VisionTransformer):
    """
    Implements a vision transformer that allows self-supervised pre-training by reconstructing swapped
    patches. This should incentivize the learning of meaningful filters and information flow between tokens.
    """

    def __init__(self, img_size, patch_size, num_classes, dim=256, depth=12, heads=12, mlp_dim=256,
                     channels=3):
        super().__init__(img_size, patch_size, num_classes, dim, depth,
                         heads, mlp_dim,channels, num_convs=0, droprate=0.)
        # only accept quadratic patches with a size to the power of two
        assert(self.patch_height == self.patch_width and is_power_of_two(self.patch_height))
        self.pretrain = False # pretraining mode

        #self.half_pos_embedding = nn.Parameter(torch.zeros(self.num_patches, dim//2, requires_grad=True))
        self.token_decoder = TokenDecoder(dim, self.patch_height, self.num_patches)

    def enable_pretrain(self, status):
        self.pretrain = status

    def _stitch_patches(self, patches):
        bs, np, ph, pw, _ = patches.shape
        patches = patches.permute(0,4,2,3,1)
        patches = patches.reshape(bs, -1, np)
        img = F.fold(patches, (self.img_height, self.img_width),
                     kernel_size=(ph, pw), stride=(ph,pw))
        return img
        
    def forward(self, x, patch_mask=None):
        if self.pretrain:
            return self._forward_decoder(x, patch_mask)
        return super().forward(x) # classify

    def _forward_decoder(self, x, patch_mask):
        patches = self._to_patches(x)      

        # patches[patch_mask] = 0.
        # imgs = self._stitch_patches(patches)
        # for img in imgs:
        #     plt.imshow(img.permute(1,2,0).cpu())
        #     plt.show()

        patches_flat = torch.flatten(patches, start_dim=2)

        if patch_mask != None: # mask out patches
            patches_flat[patch_mask] = 0.

        # encode patches
        z = self.patch_embedding(patches_flat) + self.pos_embedding
        z = self.transformer_encoders(z)

        # decode patches
        decoded_patches = self.token_decoder(z)
        decoded_patches = decoded_patches.permute(0,1,3,4,2)

        # reassemble image
        img = self._stitch_patches(decoded_patches)
        return img