"""PDF -> plain text for the text modality.

The extractor is part of the model's input distribution: train and eval MUST use the
same one with the same settings. pdfplumber by default; docling is an option recorded
in an ADR if it is ever switched on.

Phase 1.
"""

from __future__ import annotations

from pathlib import Path


def pdf_to_text(pdf: Path) -> str:
    raise NotImplementedError("phase 1: pdfplumber, layout-aware, page separator")
