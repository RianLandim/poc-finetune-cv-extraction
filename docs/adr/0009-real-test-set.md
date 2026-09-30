# ADR 0009 — A small real test set, with consent

- **Status:** Proposed
- **Date:** 2026-09-30

## Context

Every training and most test data is synthetic (ADR 0002). Only real resumes can tell
whether the model transfers.

## Decision

Collect 30–50 real resumes **with written consent**, annotate the gold JSON by hand
against `cvx.schema`, keep them in `data/real_test/` (gitignored). They are never used
for training, prompt tuning or model selection — only for the final report, where
`real_test` is the headline number.

## Consequences

- Personal data (LGPD): stored locally only, never committed, never sent to an external
  API, deleted on request. Generated outputs on them are not shared.
- 30–50 is enough to detect a large gap, not to rank close models; say so in the report.
