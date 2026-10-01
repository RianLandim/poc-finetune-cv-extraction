"""Loads a YAML config into a dotted-access object.

Configs are the source of truth for every hyperparameter and path; scripts must never
hardcode a value that lives in one. There are two kinds:

- ``configs/data.yaml`` -- synthetic generation and splits, shared by both modalities.
- ``configs/<model>.yaml`` -- one per modality: model, LoRA, training, decode, paths.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA_CONFIG = ROOT / "configs" / "data.yaml"

DATA_SECTIONS = ("source", "generate", "templates", "splits", "paths")
MODEL_SECTIONS = ("model", "input", "lora", "train", "decode", "paths")


def _namespacify(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _namespacify(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_namespacify(v) for v in value]
    return value


def _load(path: str | Path, required: tuple[str, ...]) -> SimpleNamespace:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"config at {path} must be a mapping, got {type(raw).__name__}")
    for section in required:
        if section not in raw:
            raise ValueError(f"config at {path} is missing the '{section}' section")
    return _namespacify(raw)


def load_data_config(path: str | Path = DATA_CONFIG) -> SimpleNamespace:
    return _load(path, DATA_SECTIONS)


def load_model_config(path: str | Path) -> SimpleNamespace:
    cfg = _load(path, MODEL_SECTIONS)
    if cfg.input.modality not in ("text", "vision"):
        raise ValueError(f"input.modality must be 'text' or 'vision', got {cfg.input.modality!r}")
    return cfg


def gguf_path(cfg: SimpleNamespace, which: str, run: str) -> Path:
    """The quantised GGUF that stage 4 writes and stage 6 serves.

    The base does not depend on the training run; the tuned model does.
    """
    if which not in ("base", "tuned"):
        raise ValueError(f"which must be 'base' or 'tuned', got {which!r}")
    tag = "base" if which == "base" else f"{run}-tuned"
    quant = cfg.export.quant.lower()
    return Path(cfg.paths.gguf_dir) / f"{cfg.paths.gguf_prefix}-{tag}-{quant}.gguf"


def mmproj_path(cfg: SimpleNamespace) -> Path | None:
    """The vision projector, exported once from the base (frozen tower, ADR 0012)."""
    if cfg.input.modality != "vision":
        return None
    return Path(cfg.paths.gguf_dir) / f"{cfg.paths.gguf_prefix}-mmproj-f16.gguf"
