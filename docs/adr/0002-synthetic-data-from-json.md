# ADR 0002 — Synthetic resumes, generated from the JSON outward

- **Status:** Proposed
- **Date:** 2026-09-30

## Context

A survey on 2026-09-30 found no public dataset that is simultaneously resume PDFs,
Brazilian Portuguese, labelled with structured fields, and openly licensed:

| Candidate | Why not |
|---|---|
| Kaggle Resume Dataset (~2.4k) | English; only a job-category label, no fields |
| amnamine/ResumeDataset (HF) | 100 files, unlabelled |
| JobHop / JobHop v2 | Belgian career trajectories, not documents; licence unverified |
| SynthResume / ResumeBench | Availability and licence unverified; useful as OOD eval at best |
| Lattes CVs | Real personal data (LGPD) and an academic-only format |

## Decision

Generate the corpus. Direction matters: start from the **gold JSON** and render it to
PDF, never the reverse. The label is then exact by construction and needs no annotation.

Seeds come from `nvidia/Nemotron-Personas-Brazil` (CC-BY-4.0, fully synthetic): occupation,
education, city and age give plausible Brazilian career shapes.

## Consequences

- Unlimited, exactly-labelled, PII-free training data.
- Synthetic data is cleaner than reality. The mitigation is layout diversity (ADR 0004)
  and a small real test set that is the headline number (ADR 0009).
- Label quality now depends on the renderer: `tests/test_render_roundtrip.py` checks
  every gold value is visible in the PDF.

## Alternatives considered

- **Label real resumes with a large LLM, train on that.** Labels inherit the labeller's
  errors, and real resumes carry PII we should not train on.
- **Use a public English corpus and translate.** Layouts and conventions (CEP, UF,
  "Ensino Médio", phone formats) would be wrong.
