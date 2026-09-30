#!/usr/bin/env python
"""Stage 6 -- score base and tuned on test_seen, test_unseen and real_test.

Sequential: the Makefile starts one server, this script scores it, the Makefile stops it
(base and tuned never share the GPU). Raw /completion only (ADR 0011).

Each model is scored twice (ADR 0007):
- free decoding -- measures whether the model learnt the format (JSON validity);
- grammar-constrained (json_schema from cvx.schema) -- isolates field accuracy.

Refuses to run when the dataset manifest's prompt/schema version differs from cvx.prompt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvx.config import load_model_config  # noqa: E402,F401
from cvx.metrics import score  # noqa: E402,F401
from cvx.prompt import manifest, parse_generation  # noqa: E402,F401


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--which", choices=("base", "tuned"), required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    raise NotImplementedError("phase 2")


if __name__ == "__main__":
    main()
