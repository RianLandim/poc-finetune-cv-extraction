# ADR 0005 — One schema; protected fields excluded

- **Status:** Proposed
- **Date:** 2026-09-30

## Context

The schema is the task. It must be identical in generation, training targets, grammar-
constrained decoding and scoring, and it decides what the model learns to pull out of a
real person's resume.

## Decision

`src/cvx/schema.py` (Pydantic, `extra="forbid"`) is the only definition. Dates are
`YYYY-MM` or `YYYY`; `fim: null` means current. The target serialisation is compact JSON
in schema key order (`to_target`).

CPF, RG, date of birth, photo, marital status, gender and similar are **not in the
schema**, and the generator never renders them. A generation containing such a key fails
validation (`schema_violation`).

## Consequences

- Changing a field bumps `SCHEMA_VERSION`, invalidates the dataset and the adapter.
- Downstream systems that need those fields must get them elsewhere, with consent.
