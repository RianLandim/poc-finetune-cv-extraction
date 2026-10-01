from pathlib import Path

import pytest

from cvx.config import gguf_path, load_data_config, load_model_config, mmproj_path

ROOT = Path(__file__).resolve().parents[1]


def test_data_config_loads():
    cfg = load_data_config()
    assert 0 < cfg.templates.holdout_frac < 1


@pytest.mark.parametrize("name, modality", [
    ("qwen35-4b-text.yaml", "text"),
    ("qwen35-4b-vision.yaml", "vision"),
    ("qwen3vl-4b-vision.yaml", "vision"),
])
def test_model_configs_load(name, modality):
    cfg = load_model_config(ROOT / "configs" / name)
    assert cfg.input.modality == modality
    # On the 8GB card an in-training eval pass OOMs (inherited from the persona repo).
    assert cfg.train.eval_strategy == "no"
    assert cfg.decode.temperature == 0.0
    # Both modalities load through FastVisionModel; the tower stays frozen so base and
    # tuned can share one mmproj (ADR 0012).
    assert cfg.lora.finetune_vision_layers is False


def test_gguf_paths_separate_base_from_each_tuned_run():
    cfg = load_model_config(ROOT / "configs" / "qwen35-4b-text.yaml")
    base = gguf_path(cfg, "base", "smoke")
    assert base == gguf_path(cfg, "base", "full")  # the base does not depend on the run
    assert gguf_path(cfg, "tuned", "smoke") != gguf_path(cfg, "tuned", "full")
    assert base.name == "cvx-text-base-q4_k_m.gguf"
    with pytest.raises(ValueError):
        gguf_path(cfg, "merged", "smoke")


def test_mmproj_only_for_vision():
    assert mmproj_path(load_model_config(ROOT / "configs" / "qwen35-4b-text.yaml")) is None
    vision = load_model_config(ROOT / "configs" / "qwen35-4b-vision.yaml")
    assert mmproj_path(vision).name == "cvx-vision-mmproj-f16.gguf"
