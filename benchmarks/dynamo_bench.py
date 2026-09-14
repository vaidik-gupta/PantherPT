"""torch.compile stats benchmark.

Runs the TorchDynamo compile analysis (see analysis/CompileAnalysis.py) on the fixed
12-request spec for every LLM in `src/llm/implemented` and `src/llm/pretrained`, and
records the numeric results only -- how many compiles/recompiles/graph-breaks/SymInts/
guards/frames -- not the symbol names or the reasons behind them.

Run:
    uv run python -m benchmarks.dynamo_bench
    uv run python -m benchmarks.dynamo_bench --backend inductor
"""

from __future__ import annotations

import argparse

from analysis.CompileAnalysis import REQUEST_SPECS, VOCAB_SIZE, analyze_torch_compile
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="eager",
                        help="torch.compile backend for the analysis (default: eager)")
    args = parser.parse_args()

    requests = RequestGenerator(vocab_size=VOCAB_SIZE, seed=0).build(REQUEST_SPECS)
    models = discover_models()
    print(f"spec: {len(requests)} requests  backend={args.backend}")
    print(f"models: {', '.join(label for label, _ in models)}\n")

    header = f"{'model':<32}" + "".join(f"{name:>12}" for name, _ in COLUMNS)
    print(header)
    print("-" * len(header))

    for label, cls in models:
        model = cls(GPT2Config(vocab_size=VOCAB_SIZE))
        report = analyze_torch_compile(model, requests, backend=args.backend)
        row = f"{label:<32}" + "".join(f"{getter(report):>12}" for _, getter in COLUMNS)
        print(row)


if __name__ == "__main__":
    main()
