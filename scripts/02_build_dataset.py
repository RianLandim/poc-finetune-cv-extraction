#!/usr/bin/env python
"""Stage 2 -- data/cvs -> train/val/test jsonl for ONE modality.

Assigns splits with cvx.splits (template holdout + persona hash), builds messages with
cvx.prompt.messages(), and writes the prompt manifest next to the jsonl files. Text rows
carry the extracted text; vision rows carry page image paths (``images``), rasterised
here by cvx.generate.rasterize -- clean for every split, and for a share
(``input.augment_frac``) of TRAIN rows a scan-like version instead (``input.augment_train``).

Fails loudly if any train row's template is in the unseen set.

Writes data/datasets/<run>/<modality>/{train,val,test_seen,test_unseen}.jsonl, one row:

    {"id", "persona_uuid", "template", "split", "messages", "gold", "n_tokens"[, "images"]}

``messages`` is the prompt only; training rebuilds the full string with
cvx.prompt.render_training_text(messages, Curriculo(gold), tokenizer), so the target is
never serialised twice. ``n_tokens`` counts that full training string with the model's
own tokenizer. Train/val rows longer than model.max_seq_length are dropped (truncating
would cut the target); test rows are kept -- eval runs at serve.ctx, and the manifest
says how many test rows exceed the training length.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvx.config import load_data_config, load_model_config  # noqa: E402
from cvx.generate.rasterize import image_tokens, pdf_to_pages  # noqa: E402
from cvx.prompt import manifest as prompt_manifest  # noqa: E402
from cvx.prompt import messages, render_training_text  # noqa: E402
from cvx.schema import Curriculo  # noqa: E402
from cvx.splits import SPLITS, assign, unseen_templates  # noqa: E402

# Vision rows whose analytic token count is checked against the real processor.
PROCESSOR_CHECKS = 8



def _rasterize(job: tuple) -> list[str]:
    pdf, n_pages, dpi, max_pixels, augment, seed = job
    return [str(p) for p in pdf_to_pages(Path(pdf), n_pages, dpi, max_pixels, augment, seed)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="model config (sets modality)")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--smoke", action="store_true", help="read and write smoke.run")
    args = parser.parse_args()

    cfg = load_model_config(args.config)
    data = load_data_config(args.data_config)
    modality = cfg.input.modality

    run = data.smoke.run if args.smoke else data.generate.run
    cvs_dir = Path(data.paths.cvs_dir) / run
    gen_manifest = json.loads((cvs_dir / "manifest.json").read_text(encoding="utf-8"))
    if {k: gen_manifest[k] for k in prompt_manifest()} != prompt_manifest():
        sys.exit(f"FATAL: {cvs_dir} was generated under {gen_manifest['schema_version']=} / "
                 f"{gen_manifest['prompt_version']=}; current is {prompt_manifest()}. "
                 "Regenerate the data.")

    templates = gen_manifest["templates"]
    unseen = unseen_templates(templates, data.templates.holdout_frac, data.splits.seed)

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(cfg.model.base_id)
    tokenizer = processor.tokenizer
    max_len = cfg.model.max_seq_length

    metas = [json.loads(p.read_text(encoding="utf-8"))
             for p in sorted(cvs_dir.glob("cv_*.meta.json"))]
    splits = {m["id"]: assign(m["persona_uuid"], m["template"], unseen, data.splits.val_frac,
                              data.splits.test_frac, data.splits.seed) for m in metas}
    images: dict[str, list[str]] = {}
    if modality == "vision":
        inp = cfg.input
        jobs = []
        for m in metas:
            # Deterministic per resume: rebuilding never flips a row between clean and scan.
            augment = (splits[m["id"]] == "train" and inp.augment_train
                       and random.Random(f"{data.generate.seed}:aug:{m['id']}").random()
                       < inp.augment_frac)
            jobs.append((str(cvs_dir / f"{m['id']}.pdf"), m["n_pages"], inp.dpi,
                         inp.max_pixels, augment, data.generate.seed))
        print(f">> rasterising {len(jobs)} resumes "
              f"({sum(j[4] for j in jobs)} scan-augmented)", flush=True)
        with ProcessPoolExecutor(max_workers=data.generate.workers) as pool:
            for m, paths in zip(metas, pool.map(_rasterize, jobs, chunksize=16)):
                images[m["id"]] = paths

    rows: dict[str, list[dict]] = {s: [] for s in SPLITS}
    dropped: collections.Counter = collections.Counter()
    checked = 0
    for meta in metas:
        stem = cvs_dir / meta["id"]
        cv = Curriculo.model_validate_json(stem.with_suffix(".json").read_text(encoding="utf-8"))
        split = splits[meta["id"]]
        if split == "train" and meta["template"] in unseen:  # cvx.splits guarantees it
            raise AssertionError(f"{meta['id']}: unseen template {meta['template']} in train")
        extra = {}
        if modality == "text":
            msgs = messages(cv_text=stem.with_suffix(".txt").read_text(encoding="utf-8"))
            n_tokens = len(tokenizer(render_training_text(msgs, cv, tokenizer)).input_ids)
        else:
            pages = images[meta["id"]]
            msgs = messages(n_pages=len(pages))
            full = render_training_text(msgs, cv, processor)
            # One <|image_pad|> per page in the string; the processor expands each into
            # its grid. Checked against the real processor on the first few rows.
            n_tokens = len(tokenizer(full).input_ids) + sum(
                image_tokens(Path(p)) - 1 for p in pages)
            if checked < PROCESSOR_CHECKS:
                from PIL import Image

                real = processor(text=[full], images=[Image.open(p) for p in pages],
                                 return_tensors="pt").input_ids.shape[1]
                if real != n_tokens:
                    raise AssertionError(f"{meta['id']}: {n_tokens} tokens counted, "
                                         f"processor says {real}")
                checked += 1
            extra = {"images": pages}
        if split in ("train", "val") and n_tokens > max_len:
            dropped[split] += 1
            continue
        rows[split].append({"id": meta["id"], "persona_uuid": meta["persona_uuid"],
                            "template": meta["template"], "split": split, "messages": msgs,
                            "gold": cv.model_dump(), "n_tokens": n_tokens, **extra})

    seen_personas: dict[str, str] = {}
    for split, split_rows in rows.items():
        for r in split_rows:
            other = seen_personas.setdefault(r["persona_uuid"], split)
            if other != split:
                raise AssertionError(f"persona {r['persona_uuid']} in {other} and {split}")

    out_dir = Path(data.paths.datasets_dir) / run / modality
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, split_rows in rows.items():
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in split_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    all_tokens = sorted(r["n_tokens"] for rs in rows.values() for r in rs)

    def pct(q: float) -> int:
        return all_tokens[min(len(all_tokens) - 1, int(q * len(all_tokens)))] if all_tokens else 0

    manifest = {
        **prompt_manifest(),
        "run": run,
        "modality": modality,
        "tokenizer": cfg.model.base_id,
        "templates": templates,
        "unseen_templates": sorted(unseen),
        "counts": {s: len(rs) for s, rs in rows.items()},
        "dropped_over_max_seq_length": dict(dropped),
        "max_seq_length": max_len,
        "test_rows_over_max_seq_length": sum(
            r["n_tokens"] > max_len for s in ("test_seen", "test_unseen") for r in rows[s]),
        "n_tokens": {"p50": pct(0.5), "p95": pct(0.95), "max": pct(1.0)},
        **({"input": vars(cfg.input),
            "scan_augmented_train": sum("-scan" in r["images"][0] for r in rows["train"])}
           if modality == "vision" else {}),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    if not rows["train"]:
        sys.exit("FATAL: no train rows")


if __name__ == "__main__":
    main()
