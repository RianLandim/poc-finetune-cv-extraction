#!/usr/bin/env python
"""Stage 4 -- merge, convert and quantise; base and tuned through the SAME chain.

Explicit steps (16-bit weights -> convert_hf_to_gguf.py -> llama-quantize), each
re-runnable. Step 1 is the only one that differs, and only by the LoRA: both load the
bf16 base on CPU with transformers; tuned applies the adapter with PEFT and merges it,
base does not; both save with save_pretrained. Quantisation lineage, converter and
tokenizer files are identical.

Vision additionally emits the projector with convert_hf_to_gguf.py --mmproj. The mmproj
comes from the BASE model and is shared by base and tuned: the vision tower is frozen
(lora.finetune_vision_layers: false), so this is exact, and it keeps the A/B about the
LoRA only. If vision layers are ever tuned, the tuned mmproj must be exported too.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvx.config import gguf_path, load_data_config, load_model_config, mmproj_path  # noqa: E402


def run(command: list[str]) -> None:
    print(">>", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, check=True)


def materialise(cfg, adapter_dir: Path | None, merged_dir: Path) -> None:
    """bf16 weights on disk; with an adapter, merged into them. Peak RAM ~10GB."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        cfg.model.base_id, dtype=torch.bfloat16, device_map="cpu")
    if adapter_dir is not None:
        from peft import PeftModel

        if json.loads((adapter_dir / "adapter_config.json").read_text())["base_model_name_or_path"] \
                != cfg.model.train_id:
            raise SystemExit(f"FATAL: {adapter_dir} was not trained from {cfg.model.train_id}")
        model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    model.save_pretrained(str(merged_dir))
    AutoProcessor.from_pretrained(cfg.model.base_id).save_pretrained(str(merged_dir))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--which", choices=("base", "tuned"), required=True)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--smoke", action="store_true", help="export the smoke run's adapter")
    parser.add_argument("--llama-dir", default="vendor/llama.cpp")
    parser.add_argument("--keep-intermediates", action="store_true")
    args = parser.parse_args()
    cfg = load_model_config(args.config)
    data = load_data_config(args.data_config)
    run_name = data.smoke.run if args.smoke else data.generate.run

    llama = Path(args.llama_dir)
    out = gguf_path(cfg, args.which, run_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(cfg.paths.outputs_dir) / (run_name if args.which == "tuned" else "base")
    merged_dir = work / f"merged-16bit-{args.which}"
    f16 = out.with_name(out.name.replace(f"-{cfg.export.quant.lower()}.gguf", "-f16.gguf"))
    adapter_dir = Path(cfg.paths.outputs_dir) / run_name / "adapter" if args.which == "tuned" else None

    # A merge or GGUF older than the adapter is STALE: re-exporting after retraining would
    # silently ship the previous model and invalidate every number in the evaluation.
    if adapter_dir is not None:
        weights = adapter_dir / "adapter_model.safetensors"
        if not weights.exists():
            print(f"FATAL: no adapter at {adapter_dir}; run `make train` first", file=sys.stderr)
            return 1
        newest = weights.stat().st_mtime
        for stale in (merged_dir / "config.json", f16, out):
            if stale.exists() and stale.stat().st_mtime < newest:
                print(f">> {stale} is older than the adapter -- discarding")
                shutil.rmtree(merged_dir, ignore_errors=True)
                f16.unlink(missing_ok=True)
                out.unlink(missing_ok=True)
                break

    if out.exists():
        print(f">> {out} is up to date")
    else:
        if not (merged_dir / "config.json").exists():
            print(f">> materialising bf16 weights for '{args.which}' in {merged_dir}", flush=True)
            materialise(cfg, adapter_dir, merged_dir)
        if not f16.exists():
            # --no-mtp: save_pretrained drops Qwen3.5's multi-token-prediction weights but
            # config.json still declares the layer, and llama-server then fails to load
            # ("blk.32.attn_norm.weight not found"). MTP is unused at inference anyway.
            run([sys.executable, str(llama / "convert_hf_to_gguf.py"), str(merged_dir),
                 "--outfile", str(f16), "--outtype", "f16", "--no-mtp"])
        run([str(llama / "build" / "bin" / "llama-quantize"), str(f16), str(out), cfg.export.quant])
        print(f">> {out.name}: {out.stat().st_size / 1024**3:.2f} GB")

    mmproj = mmproj_path(cfg)
    if mmproj is not None and args.which == "base" and not mmproj.exists():
        if not (merged_dir / "config.json").exists():
            materialise(cfg, None, merged_dir)
        run([sys.executable, str(llama / "convert_hf_to_gguf.py"), str(merged_dir),
             "--mmproj", "--outfile", str(mmproj), "--outtype", "f16"])

    if not args.keep_intermediates:
        print(">> removing intermediates (pass --keep-intermediates to keep them)")
        f16.unlink(missing_ok=True)
        shutil.rmtree(merged_dir, ignore_errors=True)
    print(">> export complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
