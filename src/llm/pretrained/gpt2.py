"""
Minimal, from-scratch PyTorch implementation of GPT-2 

Matches OpenAI's original architecture:
- Learned token + positional embeddings (additive)
- Pre-LayerNorm transformer blocks
- Causal (masked) multi-head self-attention, fused QKV projection
- GELU MLP with 4x expansion
- Weight tying between token embedding and output head
- Final LayerNorm before the output projection

Includes a `from_pretrained` classmethod that downloads the real GPT-2
weights from HuggingFace and loads them into this custom implementation,
so you can verify it produces identical outputs to the reference model.

Requirements:
    pip install torch transformers --break-system-packages
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from src.llm.config.gpt2 import GPT2Config


class CausalSelfAttention(nn.Module):
    """Sublayer 1: masked multi-head self-attention with a fused QKV projection."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.n_embd = config.n_embd

        # Fused QKV projection: one matmul produces Q, K, V together (as in the
        # official GPT-2 code, where this is implemented via a Conv1D layer).
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection mixing information across heads.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Static causal mask, precomputed once and reused (not a learned parameter).
        mask = torch.tril(torch.ones(config.n_ctx, config.n_ctx))
        self.register_buffer("causal_mask", mask.view(1, 1, config.n_ctx, config.n_ctx))

    def forward(self, x):
        B, T, C = x.shape  # batch, seq_len, n_embd

        qkv = self.c_attn(x)                      # [B, T, 3*n_embd]
        q, k, v = qkv.split(self.n_embd, dim=2)   # each [B, T, n_embd]

        # Split into heads: [B, T, n_head, d_head] -> [B, n_head, T, d_head]
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # Scaled dot-product attention scores.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)   # [B, n_head, T, T]

        # Apply causal mask: forbid attending to future positions.
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v                                    # [B, n_head, T, d_head]
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # concat heads -> [B, T, n_embd]

        out = self.c_proj(out)
        out = self.resid_dropout(out)
        return out


class MLP(nn.Module):
    """Sublayer 2: position-wise feed-forward network with GELU."""

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
        self.attn = CausalSelfAttention(config)
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
        assert T <= self.config.n_ctx, f"sequence length {T} exceeds n_ctx {self.config.n_ctx}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        tok_emb = self.wte(idx)   # [B, T, n_embd]
        pos_emb = self.wpe(pos)   # [T, n_embd]
        x = self.drop(tok_emb + pos_emb)

        for block in self.h:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)   # [B, T, vocab_size]
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=1.0, top_k=None, eos_token_id=None):
        """Simple autoregressive sampling loop (no KV-cache, for clarity)."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.n_ctx else idx[:, -self.config.n_ctx:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)

            # Stop early once every sequence has emitted the end-of-sequence token.
            if eos_token_id is not None and (next_id == eos_token_id).all():
                break
        return idx

    @classmethod
    def from_pretrained(cls, model_type="gpt2"):
        """
        Load real GPT-2 weights from HuggingFace into this custom implementation.
        model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}.
        """
        from transformers import GPT2LMHeadModel

        config_args = {
            "gpt2":        dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl":     dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]

        config = GPT2Config(vocab_size=50257, n_ctx=1024, **config_args)
        model = cls(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith(".causal_mask")]

        print(f"Downloading pretrained weights: {model_type} ...")
        hf_model = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = hf_model.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith(".attn.bias") and not k.endswith(".attn.masked_bias")]

        # HuggingFace's GPT-2 uses Conv1D for these layers, whose weight matrix is
        # stored TRANSPOSED relative to nn.Linear. We transpose on the way in.
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} vs {len(sd_keys)}"

        with torch.no_grad():
            for k in sd_keys_hf:
                if any(k.endswith(w) for w in transposed):
                    assert sd_hf[k].shape[::-1] == sd[k].shape
                    sd[k].copy_(sd_hf[k].t())
                else:
                    assert sd_hf[k].shape == sd[k].shape
                    sd[k].copy_(sd_hf[k])

        print("Weights loaded successfully.")
        return model


if __name__ == "__main__":
    import tiktoken  # pip install tiktoken --break-system-packages

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT2.from_pretrained("gpt2").to(device)
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    prompt = "The transformer architecture works by"
    ids = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)

    out = model.generate(ids, max_new_tokens=30, temperature=0.8, top_k=40)
    print(enc.decode(out[0].tolist()))