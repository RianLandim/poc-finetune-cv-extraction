# MOD selects the modality: text (Qwen3.5-4B) or vision (Qwen3-VL-4B).
MOD      ?= text
UV       ?= uv
DATA_CFG ?= configs/data.yaml
# llama.cpp commit verified in the phase 0 spike (docs/spike/2026-09-30-phase0.md).
# Changing it means re-checking conversion, the media marker and image token parity.
LLAMA_COMMIT ?= bdeb855b30dfe7f6e695cba98445a7ba09e6416e
export LLAMA_COMMIT

ifeq ($(MOD),text)
CONFIG ?= configs/qwen35-4b-text.yaml
else ifeq ($(MOD),vision)
CONFIG ?= configs/qwen3vl-4b-vision.yaml
else
$(error MOD must be 'text' or 'vision', got '$(MOD)')
endif

.PHONY: setup data build train export serve eval report smoke test clean

setup:
	$(UV) sync --extra gen --extra train --extra eval --extra dev
	$(UV) run python -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible to torch'; print('torch', torch.__version__, '|', torch.cuda.get_device_name(0))"
	bash scripts/00_setup_cuda.sh
	bash scripts/00_setup_llamacpp.sh

data:
	$(UV) run python scripts/01_generate_cvs.py --config $(DATA_CFG)

build:
	$(UV) run python scripts/02_build_dataset.py --config $(CONFIG) --data-config $(DATA_CFG)

train:
	$(UV) run python scripts/03_train.py --config $(CONFIG)

export:
	$(UV) run python scripts/04_export_gguf.py --config $(CONFIG) --which tuned
	$(UV) run python scripts/04_export_gguf.py --config $(CONFIG) --which base

# make serve MODEL=outputs/gguf/cvx-vision-tuned-q4_k_m.gguf MMPROJ=outputs/gguf/cvx-vision-mmproj-f16.gguf
serve:
	bash scripts/05_serve.sh $(MODEL) $(MMPROJ)

# TODO(phase 2): own the server lifecycle here -- start base, eval, stop; start tuned,
# eval, stop -- as the persona repo does. Never both at once.
eval:
	$(UV) run python scripts/06_evaluate.py --config $(CONFIG) --which base
	$(UV) run python scripts/06_evaluate.py --config $(CONFIG) --which tuned

report:
	$(UV) run python scripts/07_report.py

test:
	$(UV) run --extra dev pytest -v

smoke:
	$(UV) run python scripts/01_generate_cvs.py --config $(DATA_CFG) --smoke
	$(UV) run python scripts/02_build_dataset.py --config $(CONFIG) --data-config $(DATA_CFG)
	@echo "Smoke data ready. Run: make train export eval report MOD=$(MOD)"

clean:
	rm -rf outputs/*/merged-16bit-* outputs/gguf/*-f16.gguf
