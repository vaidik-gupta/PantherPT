"""Compilation analysis of the from-scratch GPT-2 (`src/llm/implemented`).

We will compile the model three ways -- torch.jit.trace, torch.jit.script (TorchScript),
and torch.compile -- and record what each mechanism does. This module starts with the
torch.compile (TorchDynamo) analysis, capturing across the 12 varying-shape requests:

  * SymInts registered   -- symbolic shape variables created for dynamic dimensions
  * graph breaks         -- where Dynamo fell back to eager, and why
  * guards               -- the runtime checks installed to validate a compiled graph
  * recompiles           -- how many times graph construction restarted, and the guard
                            failure ("the point") that triggered each one

Everything above is produced by the Dynamo *front-end*, which is identical regardless of
the compile backend, so the analysis defaults to backend="eager" (skips Inductor codegen
and runs in seconds). Use --backend inductor for the full pipeline.

Run:
    uv run python -m analysis.CompileAnalysis
    uv run python -m analysis.CompileAnalysis --backend inductor
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch._dynamo as dynamo
from torch._dynamo.utils import counters

from src.llm.config.gpt2 import GPT2Config
from src.llm.implemented.gpt2 import GPT2
from src.utils.requests import RequestGenerator

VOCAB_SIZE = 50257

# 12 request specs: a grid of batch sizes x sequence lengths.
BATCH_SIZES = [1, 2, 4, 8]
SEQ_LENS = [16, 64, 256]
REQUEST_SPECS = {
    f"b{b}_s{t}": {"batch_size": b, "seq_len": t}
    for b in BATCH_SIZES
    for t in SEQ_LENS
}

# Dynamo/dynamic-shapes emit their details through these loggers.
_CAPTURE_LOGGERS = ("torch._dynamo", "torch.fx.experimental.symbolic_shapes")


@contextmanager
def capture_dynamo_logs():
    """Enable and capture Dynamo's recompile / dynamic-shape log records.

    Records are captured into the yielded list and kept off the console by disabling
    propagation on the target loggers for the duration.
    """
    buffer: list[str] = []

    class _Handler(logging.Handler):
        def emit(self, record):
            buffer.append(self.format(record))

    handler = _Handler()
    handler.setFormatter(logging.Formatter("%(message)s"))

    torch._logging.set_logs(recompiles=True, graph_breaks=True, dynamic=logging.DEBUG)

    saved = []
    for name in _CAPTURE_LOGGERS:
        lg = logging.getLogger(name)
        saved.append((lg, lg.level, lg.propagate))
        lg.addHandler(handler)
        lg.setLevel(logging.DEBUG)
        lg.propagate = False  # keep the verbose output out of the console
    try:
        yield buffer
    finally:
        for lg, level, propagate in saved:
            lg.removeHandler(handler)
            lg.setLevel(level)
            lg.propagate = propagate
        torch._logging.set_logs()  # reset to defaults


@dataclass
class Recompile:
    header: str            # e.g. "Recompiling function forward in .../gpt2.py:61"
    causes: list[str]      # guard failures that triggered it


@dataclass
class TorchCompileReport:
    backend: str
    num_requests: int
    total_compilations: int          # distinct graphs built (initial + recompiles)
    frames_total: int
    frames_ok: int
    graph_break_count: int
    graph_break_reasons: list[str] = field(default_factory=list)
    guard_count: int = 0
    guard_types: Counter = field(default_factory=Counter)
    symints: list[str] = field(default_factory=list)
    recompiles: list[Recompile] = field(default_factory=list)


def _parse_symints(logs: list[str]) -> list[str]:
    """Extract distinct SymInts from `create_symbol ...` log lines."""
    seen: dict[str, str] = {}
    pat = re.compile(r"create_symbol (\w+) = (\S+) for (.+?) \[")
    for line in logs:
        for sym, val, source in pat.findall(line):
            seen.setdefault(sym, f"{sym}  (init={val})  from {source}")
    return list(seen.values())


def _guard_category(guard) -> str:
    """The kind of guard (e.g. TENSOR_MATCH, SHAPE_ENV) it enforces."""
    name = getattr(getattr(guard, "create_fn", None), "__name__", None)
    if name:
        return name
    match = re.search(r"Create Function:\s*(\S+)", str(guard))
    return match.group(1) if match else "UNKNOWN"


def _parse_recompiles(logs: list[str]) -> list[Recompile]:
    """Turn each `Recompiling function ...` log block into a Recompile record."""
    recompiles: list[Recompile] = []
    for line in logs:
        if not line.startswith("Recompiling function"):
            continue
        parts = line.splitlines()
        header = parts[0].strip()
        causes = [p.strip().lstrip("- ").strip() for p in parts[1:] if p.strip().startswith("-")]
        recompiles.append(Recompile(header, causes))
    return recompiles


def analyze_torch_compile(model, requests, backend="eager") -> TorchCompileReport:
    model.eval()

    # --- guards + graph breaks: structured, from explain() on one representative shape ---
    dynamo.reset()
    explanation = dynamo.explain(model)(requests[0].tokens)
    guards = explanation.out_guards or []
    break_reasons = [
        getattr(r, "reason", str(r)) for r in (explanation.break_reasons or [])
    ]

    # --- symints + recompiles: from running all 12 varying-shape requests compiled ---
    dynamo.reset()
    counters.clear()
    with capture_dynamo_logs() as logs:
        compiled = torch.compile(model, backend=backend)
        with torch.no_grad():
            for req in requests:
                compiled(req.tokens)

    return TorchCompileReport(
        backend=backend,
        num_requests=len(requests),
        total_compilations=counters["stats"].get("unique_graphs", 0),
        frames_total=counters["frames"].get("total", 0),
        frames_ok=counters["frames"].get("ok", 0),
        graph_break_count=explanation.graph_break_count,
        graph_break_reasons=break_reasons,
        guard_count=len(guards),
        guard_types=Counter(_guard_category(g) for g in guards),
        symints=_parse_symints(logs),
        recompiles=_parse_recompiles(logs),
    )


def print_report(report: TorchCompileReport) -> None:
    line = "=" * 72
    print(f"\n{line}\ntorch.compile analysis  (backend={report.backend})\n{line}")

    print(f"\nrequests processed : {report.num_requests}")
    print(f"frames traced      : {report.frames_ok}/{report.frames_total} ok")
    print(f"total compilations : {report.total_compilations} "
          f"(1 initial + {max(report.total_compilations - 1, 0)} recompiles)")

    print(f"\nSymInts registered : {len(report.symints)}")
    for s in report.symints:
        print(f"  - {s}")

    print(f"\nGraph breaks       : {report.graph_break_count}")
    for r in report.graph_break_reasons:
        print(f"  - {r}")
    if report.graph_break_count == 0:
        print("  (none -- the whole forward captured into a single graph)")

    print(f"\nGuards installed   : {report.guard_count}  (by type)")
    for kind, n in report.guard_types.most_common():
        print(f"  - {kind:<28} x {n}")

    print(f"\nRecompiles / eager-graph restarts : {len(report.recompiles)}")
    for i, rc in enumerate(report.recompiles, 1):
        print(f"  [{i}] {rc.header}")
        for cause in rc.causes:
            print(f"        cause: {cause}")
    if not report.recompiles:
        print("  (none)")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="eager",
                        help="torch.compile backend for the analysis (default: eager)")
    args = parser.parse_args()

    gen = RequestGenerator(vocab_size=VOCAB_SIZE, seed=0)
    requests = gen.build(REQUEST_SPECS)

    model = GPT2(GPT2Config(vocab_size=VOCAB_SIZE)).eval()

    # Process all 12 requests eagerly first.
    print("processing requests (eager):")
    with torch.no_grad():
        for req in requests:
            logits = model(req.tokens)
            print(f"  {req.name:<10} in={tuple(req.tokens.shape)} -> logits={tuple(logits.shape)}")

    # Then start with the torch.compile analysis.
    report = analyze_torch_compile(model, requests, backend=args.backend)
    print_report(report)


if __name__ == "__main__":
    main()
