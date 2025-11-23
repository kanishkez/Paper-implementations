import torch
import torch.nn as nn


class RotaryPositionalEmbeddings(nn.Module):

    def __init__(self, dim, base=10000, max_seq_len=4096):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float().unsqueeze(1)
        freqs = t * inv_freq.unsqueeze(0)
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def rotate_half(self, x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).reshape_as(x)

    def forward(self, x, pos=None):
        if pos is None:
            pos = torch.arange(x.size(1), device=x.device)
        cos = self.cos_cached[pos]
        sin = self.sin_cached[pos]
        while cos.dim() < x.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        return (x * cos) + (self.rotate_half(x) * sin)
