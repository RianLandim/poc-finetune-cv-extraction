# ADR 0001 — Record architecture decisions

- **Status:** Accepted
- **Date:** 2026-09-30

## Context

This PoC inherits the structure of `poc-fine-tunning-llamacpp` (persona fine-tune), where
numbered ADRs proved their worth: most choices here (data source, split design, schema,
modalities) are expensive to reverse once a dataset is generated and a GPU run is spent.

## Decision

Every consequential decision is a numbered ADR in `docs/adr/`: Context, Decision,
Consequences, Alternatives considered. ADRs are immutable once Accepted; a reversal is a
new ADR that supersedes the old one, which gains a `Superseded by ADR-XXXX` line.

ADRs start as **Proposed**. They become Accepted once reviewed, or once the phase 0 spike
has measured whatever they depend on.

## Consequences

- `CLAUDE.md` stays an index of invariants and gotchas, not a changelog.
- Decisions carried over from the persona repo are restated here, not linked, so this
  repo reads on its own.
