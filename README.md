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
  services/
    inference/     # FastAPI inference API shell and extensible request schema
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
- `httpx` (dev) — HTTP integration testing

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

### Inference API shell

Set a secret API key and run the FastAPI application with an ASGI server:

```bash
export INFERENCE_API_KEY="$(uv run python -c 'import secrets; print(secrets.token_urlsafe(32))')"
uv run --with uvicorn uvicorn src.services.inference.server:app --reload
```

The shell exposes `POST /infer`, `GET /health` (liveness), `GET /ready` (model
readiness), and `GET /models` (model discovery), with API documentation at `/docs`.
All four service endpoints require an `X-API-Key` header matching
`INFERENCE_API_KEY`, including health/readiness probes. Missing or invalid keys
return `401 Unauthorized`. The default application has no concrete scheduler, so
authenticated requests return `501 Not Implemented`. When a scheduler is injected,
`POST /infer` waits for its result and returns `{"request_id": "...", "output": "..."}`.
Health, readiness, and model discovery remain shells; no models are loaded by
the server itself.

After successful authentication, each request gets a new server-generated UUID4,
available to endpoint code as `request.state.request_id` and returned in the
`X-Request-ID` response header, including handled error responses such as `422`
and `501`. Client-supplied request IDs are ignored. Failed authentication and
public documentation requests do not receive a request ID.

Authentication is implemented, uses constant-time key comparison, and refuses
server startup if `INFERENCE_API_KEY` is unset or blank. The key is read at startup;
restart the service after rotating it. Supply it through your deployment's secret
manager or environment, never commit it, and use HTTPS outside local development.
The `/docs`, `/redoc`, and `/openapi.json` routes remain public; use the **Authorize**
button in `/docs` to supply the key.

```bash
curl -i http://127.0.0.1:8000/infer \
  -H "X-API-Key: $INFERENCE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "model": "gpt2"}'
```

`InferenceRequest` in `src/services/inference/schemas.py` requires a nonempty
`prompt` and accepts an optional nonempty `model` identifier. Subclass it to add
generation parameters or other service-specific fields. Invalid requests return
`422` before reaching the endpoint shell.

### Request lifecycle foundations

`src/services/inference/scheduler/` separates three independently replaceable
interfaces from their initial implementations:

| Interface | Initial implementation | Responsibility |
| --- | --- | --- |
| `RequestStorage[Payload]` | `InMemoryRequestStorage` | Bounded request registry and non-destructive pending selection; atomic all-or-nothing claims |
| `ClientDelivery[Result]` | `FutureClientDelivery` | One async waiter per request; final result or runner error delivery |
| `ResultRetention[Result]` | `UntilDeliveredResultRetention` | Keep completed outcomes only until the waiter consumes them or delivery is abandoned |

`BaseScheduler[Payload, Result]` in `scheduler/base.py` composes these foundations.
An actual scheduler extends it and implements `run()` (selection/dispatch policy)
and `stop_running()` (stop all owned runner work before returning).
These hooks must not await `close()` on their own scheduler.
`YourScheduler.with_defaults(max_requests=...)` composes the three initial
implementations, or its constructor accepts custom storage and delivery, with a
custom retention policy injected into delivery.

The shared base handles admission rollback, pending-work notification, atomic
claims, runner completion/failure, client detachment, and supervised shutdown.
Call `start()`, then `submit_and_wait(id, payload)` for a held-open client call.
Alternatively, `submit()` and `wait()` can be used separately. A scheduling loop
can await `wait_for_pending()`, select candidates, call `claim(ids)`, and dispatch
them to a runner. After execution stops, report `complete(id, output)` or
`fail(id, error)`.

The FastAPI application factory is wired to this lifecycle:

```python
from src.services.inference.server import create_app

app = create_app(YourScheduler.with_defaults(max_requests=128))
```

Here `YourScheduler` is your concrete
`BaseScheduler[InferenceRequest, str]` subclass. Application startup starts it,
and shutdown closes it. `/infer` passes the authenticated UUID to the scheduler
and overwrites the payload's `id` with that UUID. Overload or an unavailable
scheduler returns `503`; runner errors are logged and return `500` without
exposing internal error details. The exported default `app` still returns `501`
for inference until a concrete scheduler is supplied.

Storage accepts a caller-supplied UUID: HTTP integration must pass the
post-authentication `request.state.request_id`, never the payload's `id`.
`pending()` returns a snapshot without reserving requests, allowing a scheduler
to filter/reorder candidates by model, priority, or batching needs.
`claim(ids)` validates the entire selection before marking it running. Admission
requires an explicit positive `max_requests` limit, counting queued and running
requests. `finish(id, terminal_state)` removes the record and releases capacity;
running work must actually stop before it is finished, even if its client left.

Delivery registers an ID before dispatch, then `wait(id)` asynchronously awaits
`complete(id, result)` or `fail(id, error)`. Construct it with a
`ResultRetention[DeliveryResult[Result]]` policy; success and failure outcomes are
stored by that policy, not in the notification future. Completion can happen
before waiting begins. A result is released when the waiter consumes it, not
after a network acknowledgement; there is no durable delivery guarantee.
Abandonment/cancellation cleans up delivery without implicitly stopping model
execution. `BaseScheduler` removes cancelled queued requests immediately and
marks running requests for cooperative cancellation. Runners can inspect
`cancellation_requested(id)` and acknowledge stopped execution with
`cancelled(id)`. Running capacity is held until acknowledgement or completion;
late output for a cancelled request is logged and discarded. `/infer` monitors
HTTP disconnects and detaches the client waiter. Invalid IDs, duplicate
completions, and illegal transitions raise explicitly.

All three implementations are in-memory, single-process, and owned by one event
loop. Worker threads/processes must marshal completion back to that loop.
`close()` stops delivery registration, wakes active waiters with
`DeliveryClosedError`, and discards outcomes without active waiters. There is no
cross-worker sharing, restart recovery, result history, polling, or streaming.
Scheduler-loop failures stop admission, notify clients, and stop owned runner
work. Shutdown is idempotent and continues even if the caller awaiting `close()`
is cancelled; capacity is not released before `stop_running()` confirms execution
has stopped. A scheduler instance belongs to one application lifespan and cannot
be restarted after closing.

## Common commands

```bash
uv add <package>           # add a dependency
uv add --dev <package>     # add a dev dependency
uv run <cmd>               # run a command inside the project env
uv run pytest -v           # verbose test run
```
