# Architecture Decision Records

| # | Decision | Status |
|---|---|---|
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions | Accepted |
| [0002](0002-synthetic-data-from-json.md) | Synthetic resumes, generated from the JSON outward | Proposed |
| [0003](0003-deterministic-fields-llm-prose.md) | Deterministic fields, LLM only for prose | Proposed |
| [0004](0004-split-by-template.md) | Split by template as well as by persona | Proposed |
| [0005](0005-schema-and-excluded-fields.md) | One schema; protected fields excluded | Proposed |
| [0006](0006-two-modalities-one-test-set.md) | Two modalities, one test set | Proposed; amended by 0012 |
| [0007](0007-evaluate-with-and-without-grammar.md) | Evaluate with and without grammar | Proposed |
| [0008](0008-hardware-8gb-4b-models.md) | 8GB card: 4B models, spike measurements | Accepted |
| [0009](0009-real-test-set.md) | A small real test set, with consent | Proposed |
| [0010](0010-python-env-and-pinning.md) | Isolated uv env on Python 3.12, locked | Accepted |
| [0011](0011-prompt-parity-and-endpoint.md) | One prompt renderer; raw /completion for both modalities | Accepted |
| [0012](0012-qwen35-for-both-modalities.md) | Qwen3.5-4B for both modalities | Accepted |

New decisions get the next number. ADRs are immutable once Accepted — a reversal is a new
ADR that supersedes the old one.
