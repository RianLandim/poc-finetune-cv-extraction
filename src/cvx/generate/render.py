"""Gold JSON -> HTML (Jinja) -> PDF (WeasyPrint).

Per resume the renderer samples a template plus surface variation: font family, section
order, date format ("03/2021", "mar 2021", "março de 2021"), phone format, and whether
optional fields are shown. A field hidden in the PDF must also be None in the gold JSON
-- ``tests/test_render_roundtrip.py`` checks every gold value is present in the PDF text.

Phase 1.
"""

from __future__ import annotations

from pathlib import Path

from cvx.schema import Curriculo

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"


def available_templates() -> list[str]:
    return sorted(p.stem for p in TEMPLATES_DIR.glob("*.html.j2"))


def render_pdf(cv: Curriculo, template_id: str, out_pdf: Path, seed: int) -> None:
    raise NotImplementedError("phase 1: Jinja env + WeasyPrint")
