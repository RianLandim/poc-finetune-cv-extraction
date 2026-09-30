#!/usr/bin/env python
"""Stage 3 -- QLoRA with Unsloth; FastLanguageModel or FastVisionModel by modality.

TORCHINDUCTOR_COMPILE_THREADS must be capped BEFORE torch is imported: inductor spawns
one compile worker per core at the first step and exhausted system RAM in the persona
repo. Keep the os.environ line above every torch/unsloth import.

Loss is computed on the assistant turn only (train.train_on_responses_only).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import os  # noqa: E402

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")

from cvx.config import load_model_config  # noqa: E402,F401


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    raise NotImplementedError("phase 2 (text) / phase 3 (vision)")


if __name__ == "__main__":
    main()
