# ADR 0007 — Evaluate with and without grammar

- **Status:** Proposed
- **Date:** 2026-09-30

## Context

llama.cpp can constrain decoding to a JSON schema, which makes output always parseable.
Using it everywhere would hide whether fine-tuning taught the format; never using it
would let format errors swamp field accuracy.

## Decision

Every model is scored twice:

1. **Free decoding** — JSON validity rate, and failure classes (`think_block`,
   `code_fence`, `invalid_json`, `schema_violation`).
2. **Grammar-constrained** (`cvx.schema.json_schema()`) — per-field accuracy, list P/R/F1,
   hallucination rate.

Metrics (`cvx.metrics`): scalars exact after `cvx.normalize`; experiences and education
matched on identity fields, then dates scored on matched pairs only; skills and languages
as set F1; hallucination = extracted values absent from the source text.

## Consequences

- Production can use the grammar; the free-decoding number still shows what the LoRA did.
- Decoding is greedy (`temperature: 0`) in both runs.
- **The grammar schema requires every key** (2026-10-01). llama.cpp keeps optional keys
  in schema order but lets them be skipped, so a model that wrote `habilidades` before
  `experiencias` was locked out of `experiencias` and scored 0 on it with no format error.
  `json_schema()` now marks every property required, matching `to_target`; null and `[]`
  stay valid. On the smoke `test_unseen` this moved tuned exp F1 75.3 -> 91.7 and base
  skills F1 80.4 -> 98.7. Numbers from before this date are not comparable.
- Patterns use `[0-9]`, never `\d`: the converter drops unsupported escapes and then
  accepts any string (`tests/test_schema.py`).
