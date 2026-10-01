#!/usr/bin/env python
"""Stage 2 -- data/cvs -> train/val/test jsonl for ONE modality.

Assigns splits with cvx.splits (template holdout + persona hash), builds messages with
cvx.prompt.messages(), and writes the prompt manifest next to the jsonl files. Text rows
carry the extracted text; vision rows carry page image paths.

Fails loudly if any train row's template is in the unseen set.

Writes data/datasets/<run>/<modality>/{train,val,test_seen,test_unseen}.jsonl, one row:

    {"id", "persona_uuid", "template", "split", "messages", "gold", "n_tokens"}

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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvx.config import load_data_config, load_model_config  # noqa: E402
from cvx.prompt import manifest as prompt_manifest  # noqa: E402
from cvx.prompt import messages, render_training_text  # noqa: E402
from cvx.schema import Curriculo  # noqa: E402
from cvx.splits import SPLITS, assign, unseen_templates  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="model config (sets modality)")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--smoke", action="store_true", help="read and write smoke.run")
    args = parser.parse_args()

    cfg = load_model_config(args.config)
    data = load_data_config(args.data_config)
    modality = cfg.input.modality
    if modality != "text":
        raise NotImplementedError("phase 3: vision rows need page images from the generator")

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

    tokenizer = AutoProcessor.from_pretrained(cfg.model.base_id).tokenizer
    max_len = cfg.model.max_seq_length

    rows: dict[str, list[dict]] = {s: [] for s in SPLITS}
    dropped: collections.Counter = collections.Counter()
    for meta_path in sorted(cvs_dir.glob("cv_*.meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        stem = meta_path.with_name(meta["id"])
        cv = Curriculo.model_validate_json(stem.with_suffix(".json").read_text(encoding="utf-8"))
        msgs = messages(cv_text=stem.with_suffix(".txt").read_text(encoding="utf-8"))
        split = assign(meta["persona_uuid"], meta["template"], unseen,
                       data.splits.val_frac, data.splits.test_frac, data.splits.seed)
        if split == "train" and meta["template"] in unseen:  # cvx.splits guarantees it
            raise AssertionError(f"{meta['id']}: unseen template {meta['template']} in train")
        n_tokens = len(tokenizer(render_training_text(msgs, cv, tokenizer)).input_ids)
        if split in ("train", "val") and n_tokens > max_len:
            dropped[split] += 1
            continue
        rows[split].append({"id": meta["id"], "persona_uuid": meta["persona_uuid"],
                            "template": meta["template"], "split": split, "messages": msgs,
                            "gold": cv.model_dump(), "n_tokens": n_tokens})

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
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    if not rows["train"]:
        sys.exit("FATAL: no train rows")


if __name__ == "__main__":
    main()
