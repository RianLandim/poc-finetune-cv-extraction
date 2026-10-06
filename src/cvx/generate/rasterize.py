"""PDF -> page images for the vision modality, plus optional scan augmentation.

Resolution is capped by ``input.max_pixels`` in the vision config: vision tokens grow
with pixel count and are the dominant VRAM cost on the 8GB card (ADR 0008).

Every page is resized here to a multiple of ``ALIGN`` px (patch 16 x merge 2) within
``max_pixels``, so neither the HF processor (smart_resize) nor llama.cpp's mtmd resizes
it again: both see the same pixels and produce the same number of image tokens.

Augmentation (slight rotation, blur, JPEG artefacts, tinted paper, noise) is deterministic
per (seed, resume, page). It applies to a share of TRAIN rows, and to every resume of the
scanned test variant (``*_scan`` splits, ADR 0006); the plain test splits stay clean.
``scan_pdf`` wraps scan pages in an image-only PDF -- what the text path sees when a
resume was printed and scanned.
"""

from __future__ import annotations

import io
import math
from pathlib import Path

from cvx.generate.seed import rng_for

# Qwen3.5 / Qwen3-VL: 16px patches merged 2x2, so one image token covers 32x32 px.
ALIGN = 32


def aligned_size(width: int, height: int, max_pixels: int) -> tuple[int, int]:
    """Nearest multiple-of-ALIGN size, scaled down (aspect kept) to fit max_pixels."""
    scale = min(1.0, math.sqrt(max_pixels / (width * height)))
    w = max(ALIGN, round(width * scale / ALIGN) * ALIGN)
    h = max(ALIGN, round(height * scale / ALIGN) * ALIGN)
    while w * h > max_pixels:  # rounding up can overshoot by a row/column of tokens
        if w >= h:
            w -= ALIGN
        else:
            h -= ALIGN
    return w, h


def image_tokens(path: Path) -> int:
    """<|image_pad|> tokens one aligned page expands into (HF processor and llama.cpp)."""
    from PIL import Image

    w, h = Image.open(path).size
    if w % ALIGN or h % ALIGN:
        raise AssertionError(f"{path}: {w}x{h} is not aligned to {ALIGN}px")
    return (w // ALIGN) * (h // ALIGN)


def render_pages(pdf: Path, dpi: int, max_pixels: int) -> list:
    """Clean PIL RGB images, one per page, at an aligned size."""
    import pypdfium2 as pdfium
    from PIL import Image

    doc = pdfium.PdfDocument(str(pdf))
    try:
        pages = []
        for page in doc:
            img = page.render(scale=dpi / 72).to_pil().convert("RGB")
            size = aligned_size(*img.size, max_pixels)
            if size != img.size:
                img = img.resize(size, Image.Resampling.LANCZOS)
            pages.append(img)
        return pages
    finally:
        doc.close()


def scan_like(img, seed: int, key: str):
    """A deterministic 'printed and scanned' version of a page, same size."""
    import numpy as np
    from PIL import Image, ImageFilter

    rng = rng_for(seed, f"scan:{key}")
    w, h = img.size
    tint = tuple(255 - rng.randint(0, 22) for _ in range(3))
    paper = Image.new("RGB", img.size, tint)
    # multiply: white becomes the paper tint, ink stays dark
    out = Image.fromarray((np.asarray(img, dtype=np.uint16) * np.asarray(paper) // 255)
                          .astype(np.uint8))
    angle = rng.uniform(-1.2, 1.2)
    out = out.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=tint)
    if rng.random() < 0.6:
        out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.9)))
    if rng.random() < 0.5:
        noise = np.random.default_rng(rng.getrandbits(32)).normal(0, rng.uniform(3, 8), (h, w, 1))
        out = Image.fromarray(np.clip(np.asarray(out, dtype=np.float32) + noise, 0, 255)
                              .astype(np.uint8))
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=rng.randint(35, 80))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def page_paths(stem: Path, n_pages: int, augmented: bool) -> list[Path]:
    """cv_000123-p1.png (clean) or cv_000123-p1-scan.png, per page."""
    suffix = "-scan" if augmented else ""
    return [stem.with_name(f"{stem.name}-p{i}{suffix}.png") for i in range(1, n_pages + 1)]


def pdf_to_pages(pdf: Path, n_pages: int, dpi: int, max_pixels: int, augment: bool,
                 seed: int) -> list[Path]:
    """Write the page images next to the PDF (idempotent) and return their paths."""
    stem = pdf.with_suffix("")
    paths = page_paths(stem, n_pages, augment)
    if all(p.exists() for p in paths):
        return paths
    pages = render_pages(pdf, dpi, max_pixels)
    if len(pages) != n_pages:
        raise AssertionError(f"{pdf}: {len(pages)} pages rendered, meta says {n_pages}")
    for i, (img, path) in enumerate(zip(pages, paths), 1):
        if augment:
            img = scan_like(img, seed, f"{stem.name}:p{i}")
        tmp = path.with_suffix(".tmp.png")
        img.save(tmp, format="PNG")
        tmp.replace(path)  # a crash never leaves a truncated page behind
    return paths


def scan_pdf(pages: list[Path], out: Path, dpi: int) -> Path:
    """An image-only PDF of the given (scan) pages, no text layer (idempotent)."""
    from PIL import Image

    if out.exists():
        return out
    images = [Image.open(p).convert("RGB") for p in pages]
    tmp = out.with_suffix(".tmp.pdf")
    images[0].save(tmp, format="PDF", resolution=float(dpi), save_all=True,
                   append_images=images[1:])
    tmp.replace(out)
    return out
