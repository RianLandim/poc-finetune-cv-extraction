# ADR 0003 — Deterministic fields, LLM only for prose

- **Status:** Proposed
- **Date:** 2026-09-30

## Context

An LLM asked to write a whole resume produces good prose and bad structure: overlapping
jobs, degrees finished at 15, dates that disagree with the text it wrote. Those errors
would become gold labels.

## Decision

`generate/seed.py` produces every scored field without an LLM: persona attributes, Faker
`pt_BR` for contact data and fictitious employers, and a seeded timeline generator that
respects age and education. The teacher LLM (`generate/teacher.py`) writes only `resumo`
and `experiencias[].descricao`, and its output is inserted by position.

## Consequences

- A teacher failure costs one paragraph, never a wrong label.
- Generation is reproducible from the seed, except for the prose.
- Descriptions are excluded from exact scoring and from the hallucination check, since
  they may be paraphrased.

## Alternatives considered

- **LLM writes everything, validate afterwards.** Validation catches malformed output,
  not plausible-but-inconsistent output.
