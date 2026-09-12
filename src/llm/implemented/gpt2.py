
from src.utils.attention import AttentionConfigBase, SelfAttention
from src.llm.config.gpt2 import GPT2Config
import torch
import torch.nn as nn
from torch.nn import functional as F




class MLP(nn.Module):
    """Position-wise feed-forward network with GELU."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x, approximate="tanh")   # GPT-2 uses the tanh approximation of GELU
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """One transformer block: pre-LN attention + pre-LN MLP, each with a residual."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(AttentionConfigBase(
            n_embd=config.n_embd,
            n_head=config.n_head,
            dropout=config.dropout,
            n_context=config.n_ctx
        ))
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # residual around attention
        x = x + self.mlp(self.ln_2(x))    # residual around MLP
        return x

class GPT2(nn.Module):
    """Full GPT-2 model: embeddings -> N transformer blocks -> final LN -> LM head."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)   # token embeddings
        self.wpe = nn.Embedding(config.n_ctx, config.n_embd)        # positional embeddings
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight   # weight tying

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.wte(idx)          # [B, T, n_embd]
        pos_emb = self.wpe(torch.arange(T, device=idx.device))  # [T, n_embd]
        x = self.drop(tok_emb + pos_emb)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            logits = self.forward(idx)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx

    


    