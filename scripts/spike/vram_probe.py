#!/usr/bin/env python
"""Phase 0 spike -- peak VRAM of a few QLoRA training steps at a given input size.

Not a training script: random-ish text, a synthetic dense A4 page image, a manual loop.
It exercises the same memory path as the real run (Unsloth 4-bit load, LoRA, Unsloth
gradient checkpointing, loss on the assistant turn only, adamw_8bit) and prints one JSON
line with the measured peaks. Run one configuration per process so peaks do not mix.
"""

from __future__ import annotations

import os

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

LOREM = (
    "Experiência profissional: analista administrativo na Empresa Horizonte Ltda, de 03/2019 "
    "a 08/2023, responsável por conciliação bancária, contas a pagar e relatórios gerenciais. "
    "Formação: bacharelado em Administração pela Universidade Federal, concluído em 2018. "
)


def nvidia_smi_used() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip().splitlines()[0])


class SmiSampler(threading.Thread):
    """Whole-device peak as nvidia-smi sees it: includes the CUDA context and allocator
    slack that torch.cuda.max_memory_reserved() does not."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.peak = 0
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            self.peak = max(self.peak, nvidia_smi_used())
            time.sleep(0.2)

    def stop(self) -> int:
        self._halt.set()
        self.join()
        return self.peak


def page_image(width: int, height: int):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    y, line = 20, 0
    while y < height - 20:
        draw.text((30, y), (LOREM * 3)[line % 50: line % 50 + 140], fill="black")
        y += 18
        line += 7
    return img


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--modality", choices=("text", "vision"), required=True)
    p.add_argument("--prompt-tokens", type=int, default=3000, help="text: resume text length")
    p.add_argument("--target-tokens", type=int, default=700)
    p.add_argument("--pages", type=int, default=1)
    p.add_argument("--page-px", default="1000x1400", help="vision: WxH per page")
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--finetune-vision", action="store_true")
    args = p.parse_args()

    baseline = nvidia_smi_used()
    sampler = SmiSampler()
    sampler.start()
    t0 = time.time()
    result = {"model": args.model, "modality": args.modality, "baseline_mib": baseline}

    import torch
    from unsloth import FastVisionModel

    model, processor = FastVisionModel.from_pretrained(
        args.model, load_in_4bit=True, use_gradient_checkpointing="unsloth",
    )
    result["load_s"] = round(time.time() - t0, 1)
    result["after_load_alloc_mib"] = torch.cuda.memory_allocated() // 2**20

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=args.finetune_vision,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.r, lora_alpha=args.r, lora_dropout=0.0, bias="none", random_state=3407,
    )
    FastVisionModel.for_training(model)
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    target = ('{"nome":"Fulano de Tal","experiencias":[' + '{"empresa":"Horizonte","cargo":"Analista"},' * 200)
    target_ids = tok(target, add_special_tokens=False).input_ids[: args.target_tokens]
    target_text = tok.decode(target_ids)

    if args.modality == "text":
        body_ids = tok(LOREM * 200, add_special_tokens=False).input_ids[: args.prompt_tokens]
        user = [{"type": "text", "text": "Extraia o currículo para JSON.\n\n" + tok.decode(body_ids)}]
        images = None
    else:
        w, h = (int(x) for x in args.page_px.split("x"))
        images = [page_image(w, h) for _ in range(args.pages)]
        user = [{"type": "image"} for _ in images] + [{"type": "text", "text": "Extraia o currículo para JSON."}]
    msgs = [{"role": "user", "content": user}]
    prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full = prompt + target_text + tok.eos_token

    kw = {"text": [full], "return_tensors": "pt"}
    if images:
        kw["images"] = [images]
    batch = processor(**kw).to("cuda")
    n_prompt = len(processor(**{**kw, "text": [prompt]}).input_ids[0])
    labels = batch["input_ids"].clone()
    labels[:, :n_prompt] = -100
    result["seq_len"] = int(batch["input_ids"].shape[1])
    result["n_prompt"] = n_prompt
    result["n_target"] = result["seq_len"] - n_prompt

    import bitsandbytes as bnb

    opt = bnb.optim.AdamW8bit([q for q in model.parameters() if q.requires_grad], lr=2e-4)
    torch.cuda.reset_peak_memory_stats()
    step_s = []
    for _ in range(args.steps):
        s = time.time()
        out = model(**batch, labels=labels)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_s.append(round(time.time() - s, 2))
    result["loss"] = round(out.loss.item(), 3)
    result["step_s"] = step_s
    result["torch_peak_alloc_mib"] = torch.cuda.max_memory_allocated() // 2**20
    result["torch_peak_reserved_mib"] = torch.cuda.max_memory_reserved() // 2**20
    result["smi_peak_mib"] = sampler.stop()
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
