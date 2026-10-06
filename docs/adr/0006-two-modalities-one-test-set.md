# ADR 0006 — Two modalities, one test set

- **Status:** Proposed — model choice amended by ADR 0012 (Qwen3.5-4B for both)
- **Date:** 2026-09-30

## Context

Resume PDFs are usually born-digital (text extractable), sometimes scanned or heavily
laid out. Two approaches compete: extract text and use a text LLM, or read page images
with a VLM.

## Decision

Train both, on the same generated resumes:

| Modality | Model | Input |
|---|---|---|
| text | Qwen3.5-4B | `pdfplumber` text |
| vision | Qwen3-VL-4B | page PNGs, capped at `input.max_pixels` |

Both are scored on the **same** PDFs, plus a "scanned" variant of the test set that only
the vision path can read well. The system prompt and the target are shared; only the user
turn differs (`cvx.prompt.messages`).

## Consequences

- The report is a 2×2 (base/tuned × text/vision) per split.
- The text extractor is part of the text model's input distribution: train and eval must
  use the same one, same settings.

## Alternatives considered

- **Qwen3-8B text.** Already at the VRAM limit with 1536 tokens in the persona repo; a
  resume needs ~4k. See ADR 0008.

## Addendum 2026-10-05 — the scanned variant

`test_seen_scan` / `test_unseen_scan` hold the same resumes as `test_seen` / `test_unseen`,
every page through `cvx.generate.rasterize.scan_like` (tinted paper, ~1° rotation, blur,
noise, JPEG) -- the same distribution as the train augmentation, on resumes never trained
on. The text arm reads what `pdf_to_text` extracts from an image-only PDF of those pages:
nothing, since the pipeline has no OCR. That is the honest number for a scanned resume
fed to the text path, not a straw man; an OCR step would be a third arm.
