"""torch.compile stats benchmark.

Runs the TorchDynamo compile analysis (see analysis/CompileAnalysis.py) and records the
numeric results only -- how many compiles/recompiles/graph-breaks/SymInts/guards/frames --
not the symbol names or the reasons behind them.

Two modes:
  --mode v1 (default): the non-cached analysis on the fixed 12-request spec, for every LLM
                       in `src/llm/implemented` and `src/llm/pretrained`.
  --mode v2          : the KV-cached generation analysis (one session per batch size,
                       prefill + decode steps), for every model whose forward takes a
                       `kv_state` argument.

Run:
    uv run python -m benchmarks.dynamo_bench
    uv run python -m benchmarks.dynamo_bench --mode v2 --batch-sizes 1 2 4
    uv run python -m benchmarks.dynamo_bench --backend inductor
"""

from __future__ import annotations

import argparse
import inspect

from analysis.CompileAnalysis import (
    REQUEST_SPECS,
    VOCAB_SIZE,
    analyze_torch_compile,
    analyze_v2_cached,
)
from benchmarks.tokens_per_second import discover_models
from src.llm.config.gpt2 import GPT2Config
from src.utils.requests import RequestGenerator

# Numeric columns extracted from a TorchCompileReport, in display order.
COLUMNS = [
    ("reqs", lambda r: r.num_requests),
    ("compiles", lambda r: r.total_compilations),
    ("recompiles", lambda r: len(r.recompiles)),
    ("breaks", lambda r: r.graph_break_count),
    ("symints", lambda r: len(r.symints)),
    ("guards", lambda r: r.guard_count),
    ("frames_ok", lambda r: r.frames_ok),
    ("frames_tot", lambda r: r.frames_total),
]


def _cached_models():
    """Discovered models whose forward takes a `kv_state` arg (the KV-cached path)."""
    return [
        (label, cls)
        for label, cls in discover_models()
        if "kv_state" in inspect.signature(cls.forward).parameters
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["v1", "v2"], default="v1",
                        help="v1: non-cached, 12 fixed-shape requests; "
                             "v2: KV-cached generation across batch sizes (default: v1)")
    parser.add_argument("--backend", default="eager",
                        help="torch.compile backend for the analysis (default: eager)")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4],
                        help="v2: one generation session per batch size (default: 1 2 4)")
    parser.add_argument("--prompt-len", type=int, default=8, help="v2 prefill length")
    parser.add_argument("--decode-steps", type=int, default=8, help="v2 decode steps per session")
    args = parser.parse_args()

    if args.mode == "v1":
        requests = RequestGenerator(vocab_size=VOCAB_SIZE, seed=0).build(REQUEST_SPECS)
        models = discover_models()
        print(f"spec: {len(requests)} requests  backend={args.backend}")
        print(f"models: {', '.join(label for label, _ in models)}\n")
        reports = [
            (label, analyze_torch_compile(cls(GPT2Config(vocab_size=VOCAB_SIZE)),
                                          requests, backend=args.backend))
            for label, cls in models
        ]
    else:
        cfg = GPT2Config(vocab_size=VOCAB_SIZE)
        models = _cached_models()
        print(f"spec: batch_sizes={args.batch_sizes} prefill={args.prompt_len} "
              f"decode={args.decode_steps}  backend={args.backend}")
        print(f"models: {', '.join(label for label, _ in models)}\n")
        reports = [
            (label, analyze_v2_cached(cls(cfg), cfg, batch_sizes=args.batch_sizes,
                                      prompt_len=args.prompt_len, n_decode=args.decode_steps,
                                      backend=args.backend))
            for label, cls in models
        ]

    header = f"{'model':<32}" + "".join(f"{name:>12}" for name, _ in COLUMNS)
    print(header)
    print("-" * len(header))
    for label, report in reports:
        row = f"{label:<32}" + "".join(f"{getter(report):>12}" for _, getter in COLUMNS)
        print(row)


if __name__ == "__main__":
    main()
