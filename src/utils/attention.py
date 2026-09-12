import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F



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

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)


        scores = torch.matmul(q, k.transpose(-2, -1)) * self.atten_scale
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        y = torch.matmul(attn, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        return y



