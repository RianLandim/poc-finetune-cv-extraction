"""Fill the free-text fields of a skeleton with an LLM (ADR 0003).

The teacher writes ONLY ``resumo`` and ``experiencias[].descricao``. It never sees a
field it could corrupt: its output is dropped into the skeleton by position, so a bad
generation costs one paragraph of prose, never a wrong label.

Served through llama-server's raw ``/completion`` like everything else; the teacher's
own prompt lives here because it is a data-generation prompt, not a model prompt
(``cvx.prompt`` stays the single source for the model under training).

Phase 1.
"""

from __future__ import annotations

from cvx.schema import Curriculo


def fill_prose(cv: Curriculo, persona: dict, base_url: str) -> Curriculo:
    raise NotImplementedError("phase 1: teacher prompt + /completion client")
