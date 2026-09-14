# PantherPT

A hands-on **AI systems optimization** project built on PyTorch.

The goal is to implement — from the ground up — the techniques that make modern deep
learning fast and scalable across the full lifecycle: **inference, training, distributed
training, and fine-tuning**, and to **benchmark** every technique so the wins (and
trade-offs) are measured, not assumed.

Each implementation follows primary sources — research papers and deep learning /
PyTorch books — with references recorded next to the code so an implementation can be
traced back to what it's based on.

## Approach

- **From scratch first.** Implement the core mechanism ourselves, then validate it
  against a trusted reference (e.g. a HuggingFace model) for numerical parity.
- **Benchmark everything.** Every optimization ships with a benchmark measuring latency,
  throughput, memory, and — where relevant — accuracy/quality, against a baseline.
- **Paper- and book-driven.** Each technique cites the source it implements.

## Focus areas

| Area | What we're implementing | Example techniques |
| --- | --- | --- |
| **Inference** | Making a trained model run faster and cheaper | KV-cache, quantization (int8/int4), speculative decoding, batching, `torch.compile`, flash attention |
| **Training** | Making a single-device training loop efficient | mixed precision (AMP), gradient checkpointing, fused optimizers, gradient accumulation, efficient data loading |
| **Distributed training** | Scaling training across GPUs/nodes | DDP, FSDP / ZeRO sharding, tensor & pipeline parallelism, communication overlap |
| **Fine-tuning** | Adapting pretrained models cheaply | LoRA / QLoRA, adapters, prefix/prompt tuning, PEFT |
| **Benchmarking** | Measuring all of the above | latency/throughput, peak memory, tokens/sec, scaling efficiency, quality deltas |

## Current status

The foundation is a GPT-2 implementation used as the workbench for these optimizations:

- `src/llm/implemented/gpt2.py` — GPT-2 built from scratch.
- `src/llm/pretrained/gpt2.py` — the same architecture with `from_pretrained` to load real
  GPT-2 weights from HuggingFace, used as the reference for parity checks.
- `src/llm/config/gpt2.py` — shared model config.
- `src/utils/attention.py` — reusable self-attention building block.

Both GPT-2 implementations expose **identical `state_dict`s**, so weights are
interchangeable between them and with HuggingFace. This parity is enforced by
`tests/test_gpt2_state_dict_parity.py`.

## Project layout

```
src/
  llm/
    config/        # model configuration dataclasses
    implemented/   # from-scratch model implementations
    pretrained/    # reference implementations that load real weights
  utils/           # reusable building blocks (attention, ...)
tests/             # correctness & parity tests
```

As the project grows, optimizations and their benchmarks will live alongside the models
(e.g. `src/inference/`, `src/training/`, `src/distributed/`, `src/finetune/`,
`benchmarks/`).

## Stack

- Python 3.12 (pinned via `.python-version`)
- `torch` — PyTorch
- `transformers`, `datasets` — HuggingFace
- `tiktoken` — GPT-2 tokenizer
- `pytest` (dev) — testing

## Setup

```bash
uv sync   # creates .venv and installs locked deps from uv.lock
```

## Usage

```bash
uv run python main.py        # environment sanity check (torch/transformers/device)
uv run pytest                # run the test suite
```

Benchmark generation throughput (tokens/sec) for every LLM in `implemented/` and
`pretrained/`, across all available devices (unavailable ones are skipped):

```bash
uv run python -m benchmarks.tokens_per_second               # real gpt2 (124M)
uv run python -m benchmarks.tokens_per_second --config tiny # fast smoke run
```

Load real GPT-2 weights into the from-scratch model and generate:

```bash
uv run python -m src.llm.pretrained.gpt2
```

## Common commands

```bash
uv add <package>           # add a dependency
uv add --dev <package>     # add a dev dependency
uv run <cmd>               # run a command inside the project env
uv run pytest -v           # verbose test run
```
