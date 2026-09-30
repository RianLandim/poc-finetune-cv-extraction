# ADR 0008 — 8GB card: 4B models, and what the spike must measure

- **Status:** Accepted — measured in the phase 0 spike
- **Date:** 2026-09-30

## Context

Target is the RTX 3070 Ti 8GB (~7.2GB free with GNOME). In the persona repo, Qwen3-8B
QLoRA peaked at ~7.1GB with batch 1 and `max_seq_length` 1536, with ~0.5GB margin. A
resume needs ~4k tokens of context; a VLM adds vision tokens per page on top.

## Decision

Both modalities use **4B** models, QLoRA 4-bit, batch 1 × grad-accum 16, no in-training
eval pass. The vision tower stays frozen at first. Before phase 1, a spike measures and
records here:

- peak VRAM for Qwen3.5-4B at 4096 tokens;
- peak VRAM for Qwen3-VL-4B with 1 and 2 pages at candidate `max_pixels`;
- that the pinned Unsloth and llama.cpp support Qwen3.5 and Qwen3-VL (incl. `--mmproj`).

## Measurements (2026-09-30)

Full table in [`docs/spike/2026-09-30-phase0.md`](../spike/2026-09-30-phase0.md).

| Model | Worst case measured | Peak |
|---|---|---|
| Qwen3.5-4B | text 6.7k tokens | 5.9GB |
| Qwen3.5-4B | 2 pages at 1240×1754 | 5.6GB |
| Qwen3-VL-4B | 2 pages at 1240×1754 | 6.8GB |

Both toolchains support both models: Unsloth loads and trains them; llama.cpp converts
Qwen3.5-4B with its mmproj, quantises to Q4_K_M (2.8GB) and serves it with an image at
~4.4GB.

## Consequences

- Qwen3-VL-8B is out of scope on this card.
- The CUDA toolkit comes from `vendor/cuda` via micromamba; `00_setup_llamacpp.sh` bakes
  its lib dir into the rpath (inherited from the persona repo).
