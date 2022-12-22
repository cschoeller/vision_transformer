import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """ Multi-head attention layer. """

    def __init__(self, dim, heads=8, dimheads=64):
        super().__init__()
        self.heads = heads
        self.dimheads = dimheads
        self.dim_inner = heads * 3 * dimheads
        self.scale = dimheads**-0.5

        # layers
        self.norm = nn.LayerNorm(dim)
        self.embed_qkv = nn.Linear(dim, self.dim_inner, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.to_out = nn.Linear(heads * dimheads, dim, bias=False)        

    def forward(self, x):
        x = self.norm(x)

        # querys, keys, values
        qkv = self.embed_qkv(x)
        qkv = qkv.view(x.shape[0], self.heads, x.shape[1], 3, self.dimheads) # (b, h, n, qkv, dimh)
        q, k, v = [x.squeeze() for x in torch.split(qkv, 1, dim=-2)]
 
        # compute attention per head
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)

        # stack output of heads
        out = out.transpose(-3, -2).flatten(start_dim=-2)
        out = self.to_out(out)
        return out


class TransformerEncoder(nn.Module):

    def __init__(self, dim, heads, dimheads, mlp_hdim):
        ...



x = torch.rand(3, 5, 8) # b, n, dims
att = Attention(dim=8)
y = att(x)
print(y.shape)