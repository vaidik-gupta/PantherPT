"""Reusable generator for random token-id requests used to drive GPT-2 style models.

Instead of hardcoding a handful of tensors, describe the requests you want as a spec
dict and let `RequestGenerator` build them. Each generated request keeps its own config
(name, batch size, sequence length) attached, so downstream analysis/benchmarks can label
results without threading shapes around by hand.

Example
-------
    gen = RequestGenerator(vocab_size=50257, seed=0)
    specs = {
        "small":      {"batch_size": 2, "seq_len": 8},
        "wide_batch": {"batch_size": 4, "seq_len": 8},
        "short":      {"batch_size": 2, "seq_len": 4},
        "long":       {"batch_size": 5, "seq_len": 16},
    }
    for req in gen.build(specs):
        logits = model(req.tokens.to(device))
        print(req.name, tuple(req.tokens.shape))
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Request:
    """A single request: random token ids plus the config that produced them."""

    name: str
    batch_size: int
    seq_len: int
    tokens: torch.Tensor  # [batch_size, seq_len], dtype=long

    @property
    def shape(self) -> tuple[int, int]:
        return self.batch_size, self.seq_len


class RequestGenerator:
    """Builds batches of random token-id requests from a spec dict.

    A spec maps a request name to its config, e.g.::

        {"small": {"batch_size": 2, "seq_len": 8}}

    Per-request keys:
        batch_size (int, required)
        seq_len    (int, required)
        vocab_size (int, optional) -- overrides the generator default for this request
    """

    def __init__(self, vocab_size: int, device: str | torch.device = "cpu", seed: int | None = None):
        self.vocab_size = vocab_size
        self.device = torch.device(device)
        # Token ids are drawn on CPU (portable, seedable everywhere) then moved to device.
        self.generator = torch.Generator().manual_seed(seed) if seed is not None else None

    def build(self, specs: dict[str, dict]) -> list[Request]:
        """Return one Request per entry in `specs`, preserving insertion order."""
        requests: list[Request] = []
        for name, cfg in specs.items():
            batch_size = cfg["batch_size"]
            seq_len = cfg["seq_len"]
            vocab_size = cfg.get("vocab_size", self.vocab_size)
            tokens = torch.randint(
                0, vocab_size, (batch_size, seq_len), generator=self.generator
            ).to(self.device)
            requests.append(Request(name, batch_size, seq_len, tokens))
        return requests
