import torch
import torch.nn as nn
import numpy as np

from vision_transformer import VisionTransformer, to_pair


class TokenDecoder(nn.Module):

    def __init__(self, dim_in, patch_size):
        self.dim = dim_in
        patch_size = to_pair(patch_size)


class PatchSwapVit(VisionTransformer):
    """
    Implements a vision transformer that allows self-supervised pre-training by reconstructing swapped
    patches. This should incentivize the learning of meaningful filters and information flow between tokens.
    """

    def __init__(self, img_size, patch_size, num_classes, dim=256, depth=12, heads=12, mlp_dim=256,
                     channels=3):
        super().__init__(img_size, patch_size, num_classes, dim, depth,
                         heads, mlp_dim,channels, num_convs=0, droprate=0.)
        self.pretrain = False # indicates pretraining mode

        # re-define pos_embedding with half the size to enable position swapping
        self.pos_embedding = nn.Parameter(torch.zeros(self.num_patches, dim//2, requires_grad=True))

    def train(self, pretrain=False):
        self.pretrain = pretrain
        super().train()

    def forward(self, x):
        if self.pretrain:
            return self._forward_pretrain(x)
        return super().forward(x)

    def _forward_pretrain(self, x):
        ...
