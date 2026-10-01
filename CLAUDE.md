# poc-finetune-cv-extraction

Fine-tune small models with **Unsloth QLoRA** to extract a **Brazilian resume PDF** into
structured JSON, export to **GGUF**, serve and evaluate with **llama.cpp**, on an
RTX 3070 Ti (8GB). One base model, **Qwen3.5-4B** (natively multimodal), two competing
modalities on the same test set (ADR 0006, 0012):

- **text** — reads `pdfplumber` text (`MOD=text`)
- **vision** — reads page images (`MOD=vision`)
- optional third arm: Qwen3-VL-4B on images (`MOD=vl`)

Sibling of `../poc-fine-tunning-llamacpp` (persona fine-tune); its lessons are carried
over below.

## Read first

- [`docs/adr/README.md`](docs/adr/README.md) — why each choice was made

## Status

Implemented and tested: `schema`, `prompt`, `normalize`, `metrics`, `splits`, `config`.
Every stage script and `generate/*` is a stub naming its phase:

0. ~~Spike~~ — done 2026-09-30, see [`docs/spike/2026-09-30-phase0.md`](docs/spike/2026-09-30-phase0.md).
   Both models fit; raw `/completion` takes images with token-exact parity. ADR 0012
   accepted: Qwen3.5-4B for both modalities.
1. ~~Generator + 3 templates + `make smoke MOD=text`~~ — done.
2. ~~20 templates~~ (difficulty probe: [`docs/probe/2026-10-01-templates.md`](docs/probe/2026-10-01-templates.md)),
   full Qwen3.5-4B run, eval + report.
3. Vision path.
4. Real test set and final report.

## Hardware and environment

| | |
|---|---|
| GPU | RTX 3070 Ti **8GB**, ~7.2GB free (GNOME holds ~0.3-0.4GB) |
| Driver / CUDA | Ampere `sm_86`; **no system CUDA toolkit** — `vendor/cuda` (12.8.1) |
| Python | always the project `uv` venv on 3.12, never the host Python |

Never `pip install` into the host Python. Use `uv run` / `uv sync`; `uv.lock` is
committed (ADR 0010).

## Pipeline

```
make setup → make data → make build → make train → make export → make eval → make report
```

`MOD=text` (default), `MOD=vision` or `MOD=vl` selects the model config. `make smoke` runs on 200
resumes; run it before any multi-hour training run. `make test` for the fast suite.

## Invariants — breaking these silently invalidates results

1. **One schema.** `src/cvx/schema.py` is the only definition of the target. Changing it
   bumps `SCHEMA_VERSION` and invalidates the dataset and adapters (ADR 0005).
2. **One prompt renderer.** Every prompt and message list comes from `src/cvx/prompt.py`.
   `tests/test_prompt_parity.py` enforces train/inference parity for both modalities.
   Changing the prompt bumps `PROMPT_VERSION`; eval checks it against the manifest.
3. **Labels are exact by construction.** Scored fields come from `generate/seed.py`,
   never from the teacher LLM (ADR 0003). A field hidden in the PDF is null in the gold.
4. **No unseen template in training.** `cvx.splits` holds out whole templates; one resume
   per persona (ADR 0004).
5. **No protected fields.** CPF, RG, date of birth and the like are neither generated nor
   in the schema (ADR 0005).
6. **Thinking mode off everywhere.** `enable_thinking=False`; any `<think>` in a
   generation is a format failure.
7. **Raw `/completion` for both modalities.** Images go through
   `prompt.to_server_prompt` with the marker from `GET /props` (ADR 0011).
8. **Base and tuned never run concurrently**, and go through the same merge → convert →
   quantise chain. Vision base and tuned share the base `mmproj` (frozen vision tower).
9. **The text extractor is part of the input.** Same extractor, same settings, in
   training and eval.
10. **Real resumes stay local.** `data/real_test/` is gitignored, never trained on, never
    sent to an external API (ADR 0009).

## Operational gotchas found in phase 0

- **llama-server's media marker is random per run.** `<__media__>` from the docs is
  rejected with "number of media markers (0) does not match number of bitmaps". Read
  `media_marker` from `GET /props`, or pin `LLAMA_MEDIA_MARKER`.
- **Qwen3.5-4B has no pre-quantised bnb checkpoint**; Unsloth quantises the bf16 weights
  at load. Linear-attention layers use the torch fallback unless `flash-linear-attention`
  and `causal-conv1d` are installed — slower, but it fits and works.
- **llama.cpp is pinned in the Makefile** (`LLAMA_COMMIT`). Bumping it means re-running
  the spike's `completion_probe.py` and `parity_grammar.py`: conversion, the media marker
  and image token parity all live in llama.cpp.
- **Grammar schema must require every key** (ADR 0007): optional keys can be skipped but
  not reordered, so an out-of-order key silently empties earlier lists.
- **A teacher outage does not fail generation by itself** — resumes just lose their prose.
  `01_generate_cvs.py` now exits non-zero above 1% teacher errors; run the teacher detached
  (`setsid nohup`), never as a tool background job with a default timeout.
- **Small-caps text extracts garbled** in pdfplumber; never set it on scored values.
- **Base Qwen3.5-4B + grammar already aces a clean resume.** The synthetic test set must
  be hard (multi-column, tables, scan noise) or the A/B will show nothing.

## Operational gotchas carried over from the persona repo

- **Cap `TORCHINDUCTOR_COMPILE_THREADS=4` before importing torch** — inductor spawns one
  compile worker per core at the first step and exhausted system RAM.
- **In-training eval OOMs the 8GB card** (full fp32 logits). `train.eval_strategy: "no"`.
- **Never pin `unsloth` without `unsloth-zoo`.** Resolve them as a unit.
- **`torchvision` from the same index as `torch`**, declared directly so
  `[tool.uv.sources]` applies.
- **`datasets` streaming aborts at interpreter shutdown** (exit 134). Exit via `os._exit`
  after flushing.
- **llama.cpp needs `nvcc`**: `scripts/00_setup_cuda.sh` installs it into `vendor/cuda`;
  `00_setup_llamacpp.sh` bakes its lib dir into the rpath.
- **triton needs `Python.h`**: `python-preference = "only-managed"`.
- **Never pipe a long-running script through `tail`** — the exit code becomes `tail`'s.
  Redirect to a file.

## Data note

Training data is synthetic, seeded from `nvidia/Nemotron-Personas-Brazil` (CC-BY-4.0,
no real PII); employers are fictitious. `data/real_test/` is the exception — real personal
data under consent.
