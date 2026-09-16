"""GPT2_v2 (KV-cached) must agree with the plain GPT2 (v1).

1. No padding: v1 (recompute-everything) and v2 (KV-cached) generate the same tokens
   under greedy decoding, given identical weights.
2. Padding: v2 fed a right-padded batch with a padding_mask reproduces, at each real
   position, the logits v1 produces for each request run unpadded.
"""

import torch

from src.llm.config.gpt2 import GPT2Config
from src.llm.implemented.gpt2 import GPT2, GPT2_v2
from src.utils.KVState import KVState


def _models():
    """A small v1 and v2 sharing identical weights, both in eval mode."""
    cfg = GPT2Config(vocab_size=64, n_ctx=32, n_embd=16, n_layer=2, n_head=2, dropout=0.0)
    v1 = GPT2(cfg).eval()
    v2 = GPT2_v2(cfg).eval()
    v2.load_state_dict(v1.state_dict())   # same architecture/param names -> weight-compatible
    return cfg, v1, v2


def test_v1_v2_greedy_match_without_padding():
    torch.manual_seed(0)
    cfg, v1, v2 = _models()

    idx = torch.randint(0, cfg.vocab_size, (1, 5))
    out_v1 = v1.generate(idx, max_new_tokens=10, top_k=1)   # non-cached greedy
    out_v2 = v2.generate(idx, max_new_tokens=10, top_k=1)   # KV-cached greedy

    assert torch.equal(out_v1, out_v2)


def test_v2_padding_matches_unpadded_v1():
    torch.manual_seed(0)
    cfg, v1, v2 = _models()

    # A bunch of variable-length requests.
    lengths = [3, 5, 6]
    reqs = [torch.randint(0, cfg.vocab_size, (1, L)) for L in lengths]
    max_len = max(lengths)

    # Reference: run each request UNPADDED through v1.
    ref = [v1(r) for r in reqs]                     # each [1, L, vocab]

    # Right-pad into a batch and build the padding_mask (1 = real token, 0 = pad).
    B = len(reqs)
    padded = torch.zeros(B, max_len, dtype=torch.long)
    padding_mask = torch.zeros(B, max_len, dtype=torch.long)
    for i, r in enumerate(reqs):
        L = lengths[i]
        padded[i, :L] = r[0]
        padding_mask[i, :L] = 1

    # v2 prefill with the padding_mask (cached path threads padding_mask through attention).
    kv = KVState(B, cfg.n_layer, cfg.n_head, cfg.n_ctx, cfg.d_head, padded.device)
    logits = v2(padded, kv_state=kv, padding_mask=padding_mask)   # [B, max_len, vocab]

    # Every real position must match the unpadded reference for that request.
    for i, L in enumerate(lengths):
        assert torch.allclose(logits[i, :L], ref[i][0], atol=1e-4), f"request {i} mismatch"
