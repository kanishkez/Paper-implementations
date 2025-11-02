import torch
import torch.nn as nn
import torch.nn.functional as F

attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
q_proj = nn.Linear(512, 512)
k_proj = nn.Linear(512, 512)
v_proj = nn.Linear(512, 512)
out_proj = nn.Linear(512, 512)

tokens = torch.randn(1, 5, 512)

cached_k, cached_v = None, None

for t in range(tokens.size(1)):
    x = tokens[:, t:t+1, :]

    q = q_proj(x)        # (B, 1, D)
    k = k_proj(x)
    v = v_proj(x)

    B, _, D = q.shape
    num_heads = attn.num_heads
    head_dim = D // num_heads

    q = q.view(B, 1, num_heads, head_dim).transpose(1, 2)
    k = k.view(B, 1, num_heads, head_dim).transpose(1, 2)   
    v = v.view(B, 1, num_heads, head_dim).transpose(1, 2)   

    if cached_k is None:
        cached_k = k
        cached_v = v
    else:
        cached_k = torch.cat([cached_k, k], dim=2)   
        cached_v = torch.cat([cached_v, v], dim=2)

    attn_scores = torch.matmul(q, cached_k.transpose(-2, -1)) / (head_dim ** 0.5) 
    attn_probs = F.softmax(attn_scores, dim=-1)
    out = torch.matmul(attn_probs, cached_v)   

    out = out.transpose(1, 2).contiguous().view(B, 1, D)  
    output = out_proj(out)
