import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.nn.utils.parametrize as P
from einops import rearrange

class LayerNorm(nn.Module):
    def forward(self, x):
        return F.normalize(x, dim=-1)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.norm = LayerNorm()
        self.proj_in = nn.Linear(dim, hidden_dim, bias=False)
        self.proj_out = nn.Linear(hidden_dim, dim, bias=False)
        
    def forward(self, x):
        x = self.norm(x)
        x = self.proj_in(x)
        x = F.gelu(x)
        x = self.norm(x)
        x = self.proj_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = LayerNorm()
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        
    def forward(self, x, mask=None):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            dots = dots.masked_fill(~mask, -torch.finfo(dots.dtype).max)
            
        attn = F.softmax(dots, dim=-1)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.norm(out)
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, ff_mult=4):
        super().__init__()
        self.attention = Attention(dim, heads, dim_head)
        self.ff = FeedForward(dim, dim * ff_mult)
        
    def forward(self, x, mask=None):
        x = x + self.attention(x, mask)
        x = x + self.ff(x)
        return x

class NGPT(nn.Module):
    def __init__(
        self,
        *,
        vocab_size,
        dim,
        depth,
        heads=8,
        dim_head=64,
        ff_mult=4
    ):
        super().__init__()
        self.norm = LayerNorm()
        
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, 1024, dim))
        
        self.layers = nn.ModuleList([
            TransformerBlock(dim, heads, dim_head, ff_mult)
            for _ in range(depth)
        ])
        
        self.to_logits = nn.Linear(dim, vocab_size, bias=False)
        
    def forward(
        self,
        x,
        mask=None
    ):
        n = x.shape[1]
        x = self.token_emb(x)
        x = x + self.pos_emb[:, :n]
        
        for layer in self.layers:
            x = layer(x, mask)
            
        x = self.norm(x)
        return self.to_logits(x)