
from src.utils.attention import AttentionConfigBase, SelfAttention, SelfAttentionWithKVCache
from src.utils.KVState import KVState
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
    def generate(self, idx, max_new_tokens=50, temperature=1.0, top_k=None, eos_token_id=None):
        for _ in range(max_new_tokens):
            logits = self.forward(idx)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

            # Stop early once every sequence has emitted the end-of-sequence token.
            if eos_token_id is not None and (next_token == eos_token_id).all():
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


class CompiledGPT2(GPT2):
    """GPT-2 whose forward pass is JIT-compiled with torch.compile.

    Identical to the eager GPT2 (same weights, state_dict and generate loop), but routes
    the forward through a torch.compile'd graph. Compilation is lazy: it happens on the
    first forward call, on whatever device the inputs are on.
    """

    def __init__(self, config: GPT2Config):
        super().__init__(config)
        # Compile the *parent* forward (not self.forward) so the compiled call does not
        # recurse back through this override. Held as a plain attribute, so torch does
        # not treat it as a submodule/parameter.
        self._compiled_forward = torch.compile(GPT2.forward)

    def forward(self, idx):
        return self._compiled_forward(self, idx)


class CachedBlock(nn.Module):
    """Transformer block for GPT2_v2: a KV-cache-aware sibling of Block.

    Returns the hidden state. When kv_state is given, the attention writes its new K/V into
    the preallocated cache in place (nothing to propagate back up to the model).
    """

    def __init__(self, config: GPT2Config, layer: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttentionWithKVCache(AttentionConfigBase(
            n_embd=config.n_embd,
            n_head=config.n_head,
            dropout=config.dropout,
            n_context=config.n_ctx
        ), layer)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, kv_state=None, padding_mask=None):
        # attn writes into the cache in place when kv_state is given; returns just y either way.
        x = x + self.attn(self.ln_1(x), kv_state, padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2_v2(GPT2):
    """GPT-2 with KV-cached generation. Same architecture/weights as v1, but its blocks are
    cache-aware and generate() reuses past K/V instead of recomputing the whole sequence."""

    def __init__(self, config: GPT2Config):
        super().__init__(config)
        # Swap in cache-aware blocks (same submodule names/params -> weight-compatible with v1).
        self.h = nn.ModuleList([CachedBlock(config, i) for i in range(config.n_layer)])
        self.lm_head.weight = self.wte.weight   # re-tie after rebuilding modules

    def forward(self, idx, kv_state=None, padding_mask=None):
        if kv_state is None:
            return super().forward(idx)   # v1 path (CachedBlock returns x when kv_state is None)

        B, T = idx.shape
        past = kv_state.seq_len()                                   # tokens already cached
        pos = torch.arange(past, past + T, device=idx.device)      # continue the position ids
        x = self.drop(self.wte(idx) + self.wpe(pos))

        for block in self.h:                # each block writes its K/V into the cache in place
            x = block(x, kv_state=kv_state, padding_mask=padding_mask)
        kv_state.advance(T)                 # all layers wrote at the same cursor; advance once

        x = self.ln_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=1.0, top_k=None, eos_token_id=None):
        B = idx.shape[0]
        kv = KVState(B, self.config.n_layer, self.config.n_head,
                     self.config.n_ctx, self.config.d_head, idx.device)

        logits = self.forward(idx, kv_state=kv)          # prefill the whole prompt at once
        for _ in range(max_new_tokens):
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
            logits = self.forward(next_token, kv_state=kv)   # decode: feed only the new token
        return idx


class CompiledGPT2_v2(GPT2_v2):
    """GPT2_v2 whose KV-cached forward is JIT-compiled with torch.compile.

    Same weights, state_dict and (inherited) cached generate loop as GPT2_v2, but the
    forward runs through a torch.compile'd graph. Compilation is lazy: it happens on the
    first forward, and the prefill vs decode shapes each trigger their own graph before the
    generation settles into steady state.
    """

    def __init__(self, config: GPT2Config):
        super().__init__(config)
        # Compile the *parent* forward (not self.forward) so the compiled call does not
        # recurse back through this override. Held as a plain attribute so torch does not
        # treat it as a submodule/parameter.
        self._compiled_forward = torch.compile(GPT2_v2.forward)

    def forward(self, idx, kv_state=None, padding_mask=None):
        return self._compiled_forward(self, idx, kv_state=kv_state, padding_mask=padding_mask)