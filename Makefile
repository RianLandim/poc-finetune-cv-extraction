# MOD selects the arm: text or vision (both Qwen3.5-4B, ADR 0012), or vl (Qwen3-VL-4B,
# optional third arm).
MOD      ?= text
UV       ?= uv
DATA_CFG ?= configs/data.yaml
# llama.cpp commit verified in the phase 0 spike (docs/spike/2026-09-30-phase0.md).
# Changing it means re-checking conversion, the media marker and image token parity.
LLAMA_COMMIT ?= bdeb855b30dfe7f6e695cba98445a7ba09e6416e
export LLAMA_COMMIT
# SMOKE=1 points train/export/eval/report at the smoke run (configs/data.yaml smoke.run).
SMOKE    ?=
RUNFLAG  := $(if $(SMOKE),--smoke)
# RESUME=1 makes `make eval` keep finished rows from the same GGUF and run only the rest.
RESUME   ?=
EVALFLAG := $(RUNFLAG) $(if $(RESUME),--resume)

ifeq ($(MOD),text)
CONFIG ?= configs/qwen35-4b-text.yaml
else ifeq ($(MOD),vision)
CONFIG ?= configs/qwen35-4b-vision.yaml
else ifeq ($(MOD),vl)
CONFIG ?= configs/qwen3vl-4b-vision.yaml
else
$(error MOD must be 'text', 'vision' or 'vl', got '$(MOD)')
endif

.PHONY: setup teacher data build train export serve eval report smoke test clean

setup:
	$(UV) sync --extra gen --extra train --extra eval --extra dev
	$(UV) run python -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible to torch'; print('torch', torch.__version__, '|', torch.cuda.get_device_name(0))"
	bash scripts/00_setup_cuda.sh
	bash scripts/00_setup_llamacpp.sh

# Serves the prose teacher for `make data` (ADR 0003). Blocks; run it in another terminal.
teacher:
	bash scripts/01_serve_teacher.sh $(DATA_CFG)

data:
	$(UV) run python scripts/01_generate_cvs.py --config $(DATA_CFG)

build:
	$(UV) run python scripts/02_build_dataset.py --config $(CONFIG) --data-config $(DATA_CFG)

train:
	$(UV) run python scripts/03_train.py --config $(CONFIG) --data-config $(DATA_CFG) $(RUNFLAG)

export:
	$(UV) run python scripts/04_export_gguf.py --config $(CONFIG) --data-config $(DATA_CFG) --which tuned $(RUNFLAG)
	$(UV) run python scripts/04_export_gguf.py --config $(CONFIG) --data-config $(DATA_CFG) --which base $(RUNFLAG)

# make serve MODEL=outputs/gguf/cvx-vision-tuned-q4_k_m.gguf MMPROJ=outputs/gguf/cvx-mmproj-f16.gguf
serve:
	bash scripts/05_serve.sh $(MODEL) $(MMPROJ)

# Each invocation starts its own llama-server, scores and stops it: base, then tuned,
# never both at once. Stop `make teacher` / `make serve` first.
eval:
	$(UV) run python scripts/06_evaluate.py --config $(CONFIG) --data-config $(DATA_CFG) --which base $(EVALFLAG)
	$(UV) run python scripts/06_evaluate.py --config $(CONFIG) --data-config $(DATA_CFG) --which tuned $(EVALFLAG)

report:
	$(UV) run python scripts/07_report.py --data-config $(DATA_CFG) $(RUNFLAG)

test:
	$(UV) run --extra dev pytest -v

smoke:
	$(UV) run python scripts/01_generate_cvs.py --config $(DATA_CFG) --smoke
	$(UV) run python scripts/02_build_dataset.py --config $(CONFIG) --data-config $(DATA_CFG) --smoke
	@echo "Smoke data ready. Run: make train export eval report MOD=$(MOD) SMOKE=1"

clean:
	rm -rf outputs/*/merged-16bit-* outputs/gguf/*-f16.gguf
