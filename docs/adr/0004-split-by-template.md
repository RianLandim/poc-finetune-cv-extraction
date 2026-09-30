# ADR 0004 — Split by template as well as by persona

- **Status:** Proposed
- **Date:** 2026-09-30

## Context

With N templates, a model can learn "the email is in the top-right of layout 7" rather
than "find the email". A random row split would score that as success.

## Decision

`cvx.splits`: a fixed fraction of templates (`templates.holdout_frac`, 0.2) is held out
entirely; every resume rendered with one is `test_unseen`. Among seen templates, a hash of
the persona id picks train / val / `test_seen`. One resume per persona, so no person is
in two splits. Hashing is SHA-256 over a seed — stable across runs and Python versions.

## Consequences

- The gap between `test_seen` and `test_unseen` measures layout memorisation directly.
- Needs enough templates for 20% to be meaningful: target 20–30.
- `tests/test_splits.py` and `02_build_dataset.py` both refuse any unseen template in train.
