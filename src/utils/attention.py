import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from src.utils.KVState import KVState



@dataclass
class AttentionConfigBase:
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    dropout: float = 0.1
    n_context: int = 1024
    @property
    def d_head(self):
        return self.n_embd // self.n_head



class SelfAttention(nn.Module):

    def __init__(self, config: AttentionConfigBase):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by number of heads"
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.n_layer = config.n_layer
        self.n_context = config.n_context
        self.d_head = config.n_embd // config.n_head

        self.c_attn = nn.Linear(self.n_embd, 3*self.n_embd)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)
        self.atten_scale = 1 / math.sqrt(self.d_head)

        self.dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)


        mask = torch.tril(torch.ones(self.n_context, self.n_context))
        self.register_buffer("causal_mask", mask.view(1, 1, self.n_context, self.n_context))

    def forward(self, x, padding_mask: torch.Tensor = None):
        B, T, C = x.shape

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)


        scores = torch.matmul(q, k.transpose(-2, -1)) * self.atten_scale
        neg = torch.finfo(scores.dtype).min   # finite "very negative": an all-masked row -> uniform, not NaN
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, neg)

        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask[:, None, None, :] == 0, neg)
            
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        y = torch.matmul(attn, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        return y


class SelfAttentionWithKVCache(SelfAttention):
    def __init__(self, config: AttentionConfigBase, layer: int, device=None):
        super().__init__(config)
        self.device = device
        self.layer = layer
    def forward(self, x, kv_state=None, padding_mask=None):
        B, T, C = x.shape

        if kv_state is None:
            # No cache -> behave exactly like the plain (non-cached) attention.
            return super().forward(x, padding_mask)

        # Read this layer's cached past K/V; the new K/V are concatenated in below.
        
        qkv = self.c_attn(x)
        q, k_new, v_new = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k_new = k_new.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v_new = v_new.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        k, v = kv_state.read(self.layer)
        k, v = torch.cat([k, k_new], dim=2), torch.cat([v, v_new], dim=2)

        # Total cached length (past tokens + the T new ones) and where this block starts.
        S = k.shape[2]
        past = S - T

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.atten_scale   # [B, n_head, T, S]
        neg = torch.finfo(scores.dtype).min

        # Causal mask for the new query block: query row i (absolute position past+i) may attend
        # to keys 0..past+i. Slice the precomputed lower-triangular buffer to those rows/cols.
        scores = scores.masked_fill(self.causal_mask[:, :, past:S, :S] == 0, neg)

        if padding_mask is not None:                       # padding_mask: [B, S] over all keys
            scores = scores.masked_fill(padding_mask[:, None, None, :] == 0, neg)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        y = torch.matmul(attn, v)                          # [B, n_head, T, d_head]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        # Return the new K/V as well: the model collects them from every layer and appends
        # them to the KVState in one combined update per step.
        return y, k_new, v_new
