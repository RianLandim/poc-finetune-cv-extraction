from pathlib import Path

import pytest

from cvx.config import load_data_config, load_model_config

ROOT = Path(__file__).resolve().parents[1]


def test_data_config_loads():
    cfg = load_data_config()
    assert 0 < cfg.templates.holdout_frac < 1


@pytest.mark.parametrize("name, modality", [
    ("qwen35-4b-text.yaml", "text"),
    ("qwen3vl-4b-vision.yaml", "vision"),
])
def test_model_configs_load(name, modality):
    cfg = load_model_config(ROOT / "configs" / name)
    assert cfg.input.modality == modality
    # On the 8GB card an in-training eval pass OOMs (inherited from the persona repo).
    assert cfg.train.eval_strategy == "no"
    assert cfg.decode.temperature == 0.0
