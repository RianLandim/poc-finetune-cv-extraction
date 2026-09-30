#!/usr/bin/env python
"""Stage 4 -- merge, convert and quantise; base and tuned through the SAME chain.

Explicit steps (merge -> convert_hf_to_gguf.py -> llama-quantize), each re-runnable.
Vision additionally emits the projector with convert_hf_to_gguf.py --mmproj. The mmproj
comes from the BASE model and is shared by base and tuned: the vision tower is frozen
(lora.finetune_vision_layers: false), so this is exact, and it keeps the A/B about the
LoRA only. If vision layers are ever tuned, the tuned mmproj must be exported too.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvx.config import load_model_config  # noqa: E402,F401


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--which", choices=("base", "tuned"), required=True)
    args = parser.parse_args()
    raise NotImplementedError("phase 2 (text) / phase 3 (vision)")


if __name__ == "__main__":
    main()
