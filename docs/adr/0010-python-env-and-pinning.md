# ADR 0010 — Isolated uv env on Python 3.12, locked

- **Status:** Accepted (carried over from the persona repo)
- **Date:** 2026-09-30

## Decision

- Project `uv` venv on uv-managed CPython 3.12 (`python-preference = "only-managed"`:
  triton needs `Python.h` at the first training step). Never the host Python.
- `uv.lock` is committed. Unsloth and unsloth-zoo are resolved together, never pinned
  apart. torch and torchvision come from the same cu128 index.
- Extras split by stage: `gen` (CPU-only generation), `train`, `eval`, `dev`.
- llama.cpp is built at a pinned commit (`LLAMA_COMMIT`).
