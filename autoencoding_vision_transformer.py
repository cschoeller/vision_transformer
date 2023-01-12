import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt

from vision_transformer import VisionTransformer, ResidualLayer, to_pair


def is_power_of_two(x):
    return (x != 0) and (x & (x - 1)) == 0

class TokenDecoder(nn.Module):
    """ Convolutional decoder that transforms patch embeddings back to images. """

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
            ups = nn.Upsample(scale_factor=2., mode='bicubic')
            conv = nn.Conv2d(last_ch, next_ch, kernel_size=5, stride=1, padding='same')
            layers.extend([ups, conv, nn.GELU()])
            last_ch = next_ch
        layers.pop() # remove last activation
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, x.shape[-1], 1, 1) # flatten all image patches
        x = self.net(x)
        x = x.view(-1, self.num_patches, *x.shape[1:]) # unflatten per image
        return x

class AutoencodingVit(VisionTransformer):
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
        self.ps = patch_size
        self.pretrain = False # pretraining mode
        self.token_decoder = TokenDecoder(dim, self.patch_height, self.num_patches)

    def enable_pretrain(self, status):
        self.pretrain = status

    def _stitch_patches(self, patches):
        patches = patches.view(patches.shape[0], -1, self.num_patches)
        img = F.fold(patches, (self.img_height, self.img_width),
                     kernel_size=(self.ps, self.ps), stride=(self.ps, self.ps))
        return img
        
    def forward(self, x, patch_mask=None):
        if self.pretrain:
            return self._forward_decoder(x, patch_mask)
        return super().forward(x) # classify

    def _forward_decoder(self, x, patch_mask):
        patches = self._to_patches(x)      
        patches_flat = torch.flatten(patches, start_dim=2)

        if patch_mask != None: # mask out patches
            patches_flat[patch_mask] = 0.

        # encode patches
        z = self.patch_embedding(patches_flat) + self.pos_embedding
        z = self.transformer_encoders(z)

        # decode patches
        decoded_patches = self.token_decoder(z)
        decoded_patches = decoded_patches.permute(0,2,3,4,1)

        # reassemble image
        img = self._stitch_patches(decoded_patches)
        return img