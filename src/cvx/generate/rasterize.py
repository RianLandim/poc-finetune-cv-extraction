"""PDF -> page PNGs for the vision modality, plus optional scan augmentation.

Resolution is capped by ``input.max_pixels`` in the vision config: vision tokens grow
with pixel count and are the dominant VRAM cost on the 8GB card (ADR 0008).
Augmentation (slight rotation, blur, JPEG artefacts, grey background) applies to TRAIN
only; the test set is rendered clean plus a separate "scanned" variant.

Phase 3.
"""

from __future__ import annotations

from pathlib import Path


def pdf_to_pages(pdf: Path, out_dir: Path, dpi: int, augment: bool, seed: int) -> list[Path]:
    raise NotImplementedError("phase 3: pypdfium2 + PIL augmentation")
