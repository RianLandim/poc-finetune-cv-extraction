"""PDF -> plain text for the text modality.

The extractor is part of the text model's input distribution: training, evaluation and
serving MUST use this function with these settings (CLAUDE.md invariant 9).

pdfplumber's default ``extract_text`` groups characters into lines by vertical position
across the whole page, so a two-column layout comes out interleaved line by line. That is
deliberate: it is what naive extraction does to real resumes, and the model has to cope.
"""

from __future__ import annotations

from pathlib import Path

PAGE_SEPARATOR = "\n\n"


def pdf_to_text(pdf: Path) -> str:
    import pdfplumber

    with pdfplumber.open(pdf) as doc:
        pages = [(page.extract_text() or "").strip() for page in doc.pages]
    return PAGE_SEPARATOR.join(p for p in pages if p)
