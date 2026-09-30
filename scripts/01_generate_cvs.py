#!/usr/bin/env python
"""Stage 1 -- personas -> gold JSON -> PDF, text and page images.

For each sampled persona: seed.skeleton() -> teacher.fill_prose() -> render.render_pdf()
-> extract_text.pdf_to_text() -> rasterize.pdf_to_pages(). Writes one directory per run
under data/cvs/ plus a manifest (prompt/schema version, template ids, unseen set).

Resumable: a resume whose .json and .pdf both exist is skipped, so a teacher crash
costs only the resume in flight.

Datasets streaming aborts at interpreter shutdown (lesson from the persona repo): exit via
os._exit after flushing if streaming is used.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvx.config import load_data_config  # noqa: E402,F401


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--smoke", action="store_true", help="200 resumes, 3 templates")
    args = parser.parse_args()
    raise NotImplementedError("phase 1")


if __name__ == "__main__":
    main()
