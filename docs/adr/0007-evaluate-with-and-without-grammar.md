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
