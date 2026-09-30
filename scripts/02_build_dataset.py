#!/usr/bin/env python
"""Stage 2 -- data/cvs -> train/val/test jsonl for ONE modality.

Assigns splits with cvx.splits (template holdout + persona hash), builds messages with
cvx.prompt.messages(), and writes the prompt manifest next to the jsonl files. Text rows
carry the extracted text; vision rows carry page image paths.

Fails loudly if any train row's template is in the unseen set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvx.config import load_data_config, load_model_config  # noqa: E402,F401


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="model config (sets modality)")
    parser.add_argument("--data-config", default="configs/data.yaml")
    args = parser.parse_args()
    raise NotImplementedError("phase 1 (text) / phase 3 (vision)")


if __name__ == "__main__":
    main()
