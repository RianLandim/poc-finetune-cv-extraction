#!/usr/bin/env python
"""Stage 7 -- the 2x2 table (base/tuned x text/vision), split by seen/unseen/real.

Reads every outputs/*/eval-*.jsonl and writes docs/RESULTS.md. The headline number is
real_test; synthetic numbers are reported beside it, never instead of it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="docs/RESULTS.md")
    args = parser.parse_args()
    raise NotImplementedError("phase 2")


if __name__ == "__main__":
    main()
