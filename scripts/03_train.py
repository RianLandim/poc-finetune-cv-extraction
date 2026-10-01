#!/usr/bin/env python
"""Stage 3 -- QLoRA with Unsloth on one modality's dataset.

Qwen3.5-4B is natively multimodal (Qwen3_5ForConditionalGeneration), so BOTH modalities
load through FastVisionModel -- the phase 0 spike measured text training on that path.
The vision tower stays frozen (lora.finetune_vision_layers: false).

Labels are built here, not by a response-template search: the prompt prefix is
cvx.prompt.render_prompt() of the row's messages, the full string is
render_training_text(), and every prompt token is masked to -100. The prefix ids are
checked against the full string's ids, so a tokenisation drift at the boundary fails the
run instead of silently training on a shifted target.

TORCHINDUCTOR_COMPILE_THREADS must be capped BEFORE torch is imported: inductor spawns
one compile worker per core at the first step and exhausted system RAM in the persona
repo. Keep the os.environ lines above every torch/unsloth import.
"""

from __future__ import annotations

import os

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import unsloth  # noqa: E402,F401  # must precede transformers -- it patches on import
from unsloth import FastVisionModel  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvx.config import load_data_config, load_model_config  # noqa: E402
from cvx.prompt import manifest as prompt_manifest  # noqa: E402
from cvx.prompt import render_prompt, render_training_text  # noqa: E402
from cvx.schema import Curriculo  # noqa: E402

WARMUP_STEPS_FOR_ETA = 5


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def tokenize_text_row(row: dict, tokenizer, max_len: int) -> dict:
    """input_ids + labels with the prompt masked; raises on a prefix mismatch."""
    msgs = row["messages"]
    prompt = render_prompt(msgs, tokenizer)
    full = render_training_text(msgs, Curriculo.model_validate(row["gold"]), tokenizer)
    if not full.startswith(prompt):
        raise AssertionError(f"{row['id']}: training text does not start with the prompt")
    ids = tokenizer(full, add_special_tokens=False).input_ids
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    if ids[: len(prompt_ids)] != prompt_ids:
        raise AssertionError(f"{row['id']}: prompt tokens differ inside the full string")
    if len(ids) > max_len:
        raise AssertionError(f"{row['id']}: {len(ids)} tokens > {max_len}; stage 2 drops these")
    labels = [-100] * len(prompt_ids) + ids[len(prompt_ids):]
    return {"input_ids": ids, "labels": labels, "length": len(ids)}


def collate(pad_id: int):
    import torch

    def fn(batch: list[dict]) -> dict:
        width = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), width), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"])
            labels[i, :n] = torch.tensor(b["labels"])
            mask[i, :n] = 1
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}

    return fn


def make_eta_callback(total_steps: int):
    from transformers import TrainerCallback

    class EtaCallback(TrainerCallback):
        def __init__(self) -> None:
            self.start: float | None = None
            self.reported = False

        def on_step_end(self, args, state, control, **kwargs):
            step = state.global_step
            if step == 1:
                self.start = time.time()
            elif step >= WARMUP_STEPS_FOR_ETA and self.start and not self.reported:
                per_step = (time.time() - self.start) / (step - 1)
                print(f"\n>> measured {per_step:.1f}s/step -> ETA "
                      f"{(total_steps - step) * per_step / 3600:.2f}h for {total_steps} steps\n",
                      flush=True)
                self.reported = True

    return EtaCallback()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--smoke", action="store_true", help="train on smoke.run")
    args = parser.parse_args()
    cfg = load_model_config(args.config)
    data = load_data_config(args.data_config)
    modality = cfg.input.modality
    if modality != "text":
        raise NotImplementedError("phase 3: vision rows need page images and a processor collator")

    run = data.smoke.run if args.smoke else data.generate.run
    data_dir = Path(data.paths.datasets_dir) / run / modality
    ds_manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if {k: ds_manifest[k] for k in prompt_manifest()} != prompt_manifest():
        print(f"FATAL: {data_dir} was built under {ds_manifest}, code is at "
              f"{prompt_manifest()}. Re-run `make data build`.", file=sys.stderr)
        return 1

    out_dir = Path(cfg.paths.outputs_dir) / run
    adapter_dir = out_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    model, processor = FastVisionModel.from_pretrained(
        cfg.model.train_id,
        load_in_4bit=cfg.model.load_in_4bit,
        use_gradient_checkpointing=cfg.lora.use_gradient_checkpointing,
    )
    tokenizer = processor.tokenizer
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=cfg.lora.finetune_vision_layers,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        random_state=cfg.lora.random_state,
        use_gradient_checkpointing=cfg.lora.use_gradient_checkpointing,
    )
    FastVisionModel.for_training(model)
    lora_modules = sorted({n.split(".lora_A")[0].rsplit(".", 1)[-1]
                           for n, _ in model.named_parameters() if ".lora_A" in n})
    if not cfg.lora.finetune_vision_layers and any(
            "visual" in n and ".lora_" in n for n, _ in model.named_parameters()):
        raise AssertionError("LoRA reached the vision tower; base and tuned could not share mmproj")
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f">> LoRA on {lora_modules}, {n_trainable / 1e6:.1f}M trainable params", flush=True)

    import torch
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments

    max_len = cfg.model.max_seq_length
    train_rows = load_rows(data_dir / "train.jsonl")
    train = Dataset.from_list([tokenize_text_row(r, tokenizer, max_len) for r in train_rows])
    n_target = sum(sum(t != -100 for t in r["labels"]) for r in train)
    print(f">> {len(train)} train rows, {sum(train['length'])} tokens "
          f"({n_target} in the loss)", flush=True)

    t = cfg.train
    effective_batch = t.per_device_train_batch_size * t.gradient_accumulation_steps
    total_steps = max(1, -(-len(train) // effective_batch) * t.num_train_epochs)

    trainer = Trainer(
        model=model,
        train_dataset=train,
        data_collator=collate(tokenizer.pad_token_id),
        args=TrainingArguments(
            output_dir=str(out_dir / "checkpoints"),
            per_device_train_batch_size=t.per_device_train_batch_size,
            gradient_accumulation_steps=t.gradient_accumulation_steps,
            train_sampling_strategy="group_by_length" if t.group_by_length else "random",
            length_column_name="length",
            num_train_epochs=t.num_train_epochs,
            learning_rate=t.learning_rate,
            lr_scheduler_type=t.lr_scheduler_type,
            warmup_ratio=t.warmup_ratio,
            optim=t.optim,
            weight_decay=t.weight_decay,
            bf16=True,
            logging_steps=min(t.logging_steps, max(1, total_steps // 5)),
            eval_strategy=t.eval_strategy,
            save_steps=t.save_steps,
            save_total_limit=t.save_total_limit,
            seed=t.seed,
            remove_unused_columns=False,
            # Pinned host memory is non-swappable; it got the persona run killed.
            dataloader_pin_memory=False,
            report_to="none",
        ),
    )
    trainer.add_callback(make_eta_callback(total_steps))

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    result = trainer.train()
    elapsed = time.time() - started
    peak_gb = torch.cuda.max_memory_reserved() / 1024**3
    print(f">> finished in {elapsed / 60:.1f} min, peak reserved VRAM {peak_gb:.2f} GB", flush=True)

    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    (out_dir / "train_log.json").write_text(json.dumps({
        **prompt_manifest(),
        "run": run,
        "modality": modality,
        "train_id": cfg.model.train_id,
        "lora_modules": lora_modules,
        "trainable_params": n_trainable,
        "train_rows": len(train),
        "train_tokens": sum(train["length"]),
        "loss_tokens": n_target,
        "total_steps": total_steps,
        "train_runtime_min": elapsed / 60,
        "peak_reserved_vram_gb": peak_gb,
        "final_train_loss": result.training_loss,
        "log_history": trainer.state.log_history,
    }, indent=1), encoding="utf-8")
    print(f">> adapter saved to {adapter_dir}", flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # datasets/unsloth threads can abort at interpreter shutdown
