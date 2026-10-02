"""cvx.generate.rasterize: aligned page sizes (token parity) and deterministic augmentation."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from cvx.generate.rasterize import ALIGN, aligned_size, image_tokens, pdf_to_pages, scan_like
from cvx.generate.render import render_pdf, sample_style


@pytest.mark.parametrize("size", [(992, 1403), (1240, 1754), (595, 842), (3000, 4000)])
def test_aligned_size_is_a_multiple_of_32_within_the_cap(size):
    w, h = aligned_size(*size, max_pixels=1_400_000)
    assert w % ALIGN == 0 and h % ALIGN == 0
    assert w * h <= 1_400_000
    assert abs(w / h - size[0] / size[1]) < 0.05


def test_a4_at_120_dpi_is_992x1408():
    assert aligned_size(992, 1403, 1_400_000) == (992, 1408)


def _page():
    img = Image.new("RGB", (320, 448), "white")
    ImageDraw.Draw(img).text((20, 20), "Joana Ribeiro Lopes", fill="black")
    return img


def test_scan_like_is_deterministic_and_keeps_the_size():
    a, b = scan_like(_page(), 3407, "cv_1:p1"), scan_like(_page(), 3407, "cv_1:p1")
    assert a.size == (320, 448)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert not np.array_equal(np.asarray(a), np.asarray(scan_like(_page(), 3407, "cv_2:p1")))


def test_pdf_to_pages_writes_aligned_pngs_once(tmp_path, cv):
    pdf = tmp_path / "cv_000001.pdf"
    n = render_pdf(cv, sample_style("classico", 3407, "x"), pdf)
    pages = pdf_to_pages(pdf, n, dpi=120, max_pixels=1_400_000, augment=False, seed=3407)
    assert [p.name for p in pages] == [f"cv_000001-p{i}.png" for i in range(1, n + 1)]
    assert image_tokens(pages[0]) == (992 // ALIGN) * (1408 // ALIGN)
    mtime = pages[0].stat().st_mtime_ns
    assert pdf_to_pages(pdf, n, 120, 1_400_000, False, 3407) == pages
    assert pages[0].stat().st_mtime_ns == mtime  # idempotent: not re-rendered
    scan = pdf_to_pages(pdf, n, 120, 1_400_000, True, 3407)
    assert scan[0].name == "cv_000001-p1-scan.png" and Image.open(scan[0]).size == (992, 1408)
