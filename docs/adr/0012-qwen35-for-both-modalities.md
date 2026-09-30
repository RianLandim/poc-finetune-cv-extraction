# ADR 0012 — Qwen3.5-4B for both modalities

- **Status:** Proposed
- **Date:** 2026-09-30
- **Amends:** ADR 0006 (if accepted)

## Context

ADR 0006 pairs Qwen3.5-4B (text) with Qwen3-VL-4B (vision). The phase 0 spike found that
Qwen3.5-4B is itself natively multimodal (`Qwen3_5ForConditionalGeneration`, same vision
tower shape as Qwen3-VL-4B) and, on this card, better on every axis measured:

| | Qwen3.5-4B | Qwen3-VL-4B |
|---|---|---|
| Peak, 2 pages 1000×1400 | 5.0GB | 6.2GB |
| Step, 2 pages | 3.8 s | 5.3 s |
| Peak, text 3.7k tokens | 5.2GB | 6.2GB |
| Release | 2026 | 2025 |
| llama.cpp: convert + mmproj + serve + image on `/completion` | verified | converter registered, not run |

## Decision

Use **Qwen3.5-4B as the single base model** for both modalities. The text-vs-vision
comparison then differs only in the input, not in the model — a cleaner A/B than two
different bases, and one export chain to maintain.

Qwen3-VL-4B stays available as an optional third arm (its config is kept) if Qwen3.5's
vision underperforms.

## Consequences

- One base GGUF + one mmproj serve both modalities; a text-only eval simply omits
  `--mmproj`. Two LoRAs (text-tuned, vision-tuned) are still trained separately.
- A third arm becomes possible for free: one LoRA trained on a mix of text and image
  inputs.
- Qwen3.5's linear attention runs on the torch fallback without `flash-linear-attention`
  and `causal-conv1d`. Already the faster model; installing them is optional.
- No pre-quantised 4-bit checkpoint: every load quantises 9.3GB of bf16 (~25 s).

## Alternatives considered

- **Keep Qwen3-VL-4B for vision (ADR 0006 as written).** Heavier and slower here, and the
  comparison conflates modality with model.
