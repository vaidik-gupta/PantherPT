"""Tokens/sec generation benchmark.

Runs every LLM found in `src/llm/implemented` and `src/llm/pretrained` across every
available device (cpu / cuda / mps), and reports generation throughput in tokens/sec.

Design choices that make the numbers meaningful:
- Devices that aren't available are skipped (with a note), not errored.
- Models are randomly initialized: generation throughput depends on architecture and
  shapes, not on weight values, so this keeps the benchmark fast and fully offline.
- Each (model, device) is warmed up (untimed) to absorb lazy device init / first-call
  overhead, then timed over several prompts and reported as mean +/- std tokens/sec.
- The device is synchronized around each timed region so async CUDA/MPS work is counted.

Run:
    uv run python -m benchmarks.tokens_per_second
    uv run python -m benchmarks.tokens_per_second --config tiny --max-new-tokens 16
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import pkgutil
import statistics
from time import perf_counter

import tiktoken
import torch
import torch.nn as nn

from src.llm.config.gpt2 import GPT2Config

# Packages scanned for runnable LLMs, keyed by the short label we report them under.
MODEL_PACKAGES = {
    "implemented": "src.llm.implemented",
    "pretrained": "src.llm.pretrained",
}

# Real prompts (tokenized with the GPT-2 encoding) so prompt lengths are realistic.
PROMPTS = [
    "The transformer architecture works by",
    "In a distributed training setup, the gradients are",
    "To reduce inference latency, a common technique is to",
    "Large language models are pretrained on large corpora of",
    "Quantization shrinks a model by representing its weights with",
]

# Model sizes. vocab_size stays 50257 so the GPT-2 tokenizer ids are always valid.
CONFIG_PRESETS = {
    "gpt2": dict(vocab_size=50257, n_ctx=1024, n_embd=768, n_layer=12, n_head=12),
    "tiny": dict(vocab_size=50257, n_ctx=256, n_embd=128, n_layer=2, n_head=2),
}


def discover_models() -> list[tuple[str, type[nn.Module]]]:
    """Find every runnable LLM class in the model packages.

    A "runnable LLM" is an `nn.Module` subclass that is *defined in* the scanned module
    (not merely imported into it) and exposes a `generate` method. This naturally
    excludes building blocks (Block, MLP, attention) and picks up new models for free.
    """
    models: list[tuple[str, type[nn.Module]]] = []
    for origin, package_name in MODEL_PACKAGES.items():
        package = importlib.import_module(package_name)
        for mod_info in pkgutil.iter_modules(package.__path__):
            module = importlib.import_module(f"{package_name}.{mod_info.name}")
            for cls_name, cls in inspect.getmembers(module, inspect.isclass):
                if cls.__module__ != module.__name__:
                    continue  # imported symbol, not defined here
                if not issubclass(cls, nn.Module):
                    continue
                if not callable(getattr(cls, "generate", None)):
                    continue
                models.append((f"{origin}.{mod_info.name}.{cls_name}", cls))
    return models


def available_devices(requested: list[str] | None) -> list[torch.device]:
    """Return the torch devices to benchmark, skipping any that aren't available."""
    candidates = requested or ["cpu", "cuda", "mps"]
    devices: list[torch.device] = []
    for name in candidates:
        if name == "cpu":
            devices.append(torch.device("cpu"))
        elif name == "cuda" and torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        elif name == "mps" and torch.backends.mps.is_available():
            devices.append(torch.device("mps"))
        else:
            print(f"  skipping {name}: not available")
    return devices


def synchronize(device: torch.device) -> None:
    """Block until all queued work on the device has finished (no-op on CPU)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def encode_prompts() -> list[torch.Tensor]:
    enc = tiktoken.get_encoding("gpt2")
    return [torch.tensor([enc.encode(p)], dtype=torch.long) for p in PROMPTS]


def benchmark_model(
    cls: type[nn.Module],
    config: GPT2Config,
    device: torch.device,
    prompts: list[torch.Tensor],
    max_new_tokens: int,
    warmup: int,
) -> tuple[float, float, int]:
    """Return (mean tok/s, std tok/s, new tokens per prompt) for one model on one device."""
    model = cls(config).to(device).eval()

    warm = prompts[0].to(device)
    for _ in range(warmup):
        model.generate(warm, max_new_tokens=max_new_tokens)
    synchronize(device)

    rates: list[float] = []
    new_tokens = 0
    for prompt in prompts:
        idx = prompt.to(device)
        synchronize(device)
        start = perf_counter()
        out = model.generate(idx, max_new_tokens=max_new_tokens)
        synchronize(device)
        elapsed = perf_counter() - start

        new_tokens = out.shape[1] - idx.shape[1]  # batch size is 1
        rates.append(new_tokens / elapsed)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    std = statistics.stdev(rates) if len(rates) > 1 else 0.0
    return statistics.mean(rates), std, new_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=CONFIG_PRESETS, default="gpt2",
                        help="model size preset (default: gpt2, i.e. 124M)")
    parser.add_argument("--max-new-tokens", type=int, default=32,
                        help="tokens generated per prompt (default: 32)")
    parser.add_argument("--warmup", type=int, default=2,
                        help="untimed warmup generations per model/device (default: 2)")
    parser.add_argument("--devices", nargs="+", choices=["cpu", "cuda", "mps"], default=None,
                        help="restrict to these devices (default: all available)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    config = GPT2Config(**CONFIG_PRESETS[args.config])
    prompts = encode_prompts()

    print(f"config={args.config} {CONFIG_PRESETS[args.config]}")
    print(f"prompts={len(prompts)}  max_new_tokens={args.max_new_tokens}  warmup={args.warmup}\n")

    print("resolving devices:")
    devices = available_devices(args.devices)
    models = discover_models()
    print(f"\ndiscovered {len(models)} model(s): {', '.join(label for label, _ in models)}\n")

    header = f"{'model':<32}{'device':<8}{'tokens/sec':>18}{'new_tok':>9}"
    print(header)
    print("-" * len(header))

    for label, cls in models:
        for device in devices:
            try:
                mean, std, new_tok = benchmark_model(
                    cls, config, device, prompts, args.max_new_tokens, args.warmup
                )
                rate = f"{mean:8.1f} +/- {std:6.1f}"
                print(f"{label:<32}{device.type:<8}{rate:>18}{new_tok:>9}")
            except Exception as exc:  # keep going if one device/model fails
                print(f"{label:<32}{device.type:<8}{'ERROR: ' + str(exc)[:40]:>18}")


if __name__ == "__main__":
    main()
