"""Persona -> resume skeleton, with NO LLM involved (ADR 0003).

Everything the model will be scored on exactly -- name, contact, city/UF, employers,
titles, dates, institutions -- is produced here deterministically from the persona row,
Faker ``pt_BR`` and a seeded RNG. Invariants the generator must hold:

- the career timeline fits the persona's age and education (no job before 14, no
  degree before 17, no overlapping full-time jobs);
- employers are fictitious (Faker), never real companies;
- no CPF, RG or date of birth is ever generated (ADR 0005).

Phase 1.
"""

from __future__ import annotations

from cvx.schema import Curriculo


def skeleton(persona: dict, seed: int) -> Curriculo:
    """Resume with every exact field filled; ``resumo`` and ``descricao`` left None."""
    raise NotImplementedError("phase 1: timeline generator + Faker pt_BR")
