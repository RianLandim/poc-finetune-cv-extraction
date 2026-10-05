#!/usr/bin/env python
"""Stage 7 -- report: training loss, base vs tuned accuracy and processing time per modality.

Reads every outputs/*/<run>/eval-{base,tuned}.jsonl and writes a Markdown report: docs/RESULTS.md
for the full run, outputs/report-<run>.md for smoke. The headline number is real_test;
synthetic numbers are reported beside it, never instead of it.

Lists are micro-averaged (summed tp/fp/fn over the split), scalars are mean exact-match.
Format failures are already scored against an empty resume by stage 6.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvx.config import load_data_config  # noqa: E402
from cvx.metrics import SCALARS  # noqa: E402

LISTS = ("experiencias", "experiencia_datas", "formacao", "habilidades", "idiomas")
SPLIT_ORDER = ("real_test", "test_unseen", "test_seen")


def f1(tp: int, fp: int, fn: int) -> float:
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return 2 * p * r / (p + r) if p + r else 0.0


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    out = {"n": n, "json_ok": sum(r["error"] is None for r in rows) / n}
    for name in SCALARS:
        out[name] = sum(r["score"]["scalars"][name] for r in rows) / n
    out["scalars"] = sum(out[name] for name in SCALARS) / len(SCALARS)
    for name in LISTS:
        tot = collections.Counter()
        for r in rows:
            tot.update(r["score"][name])
        out[name] = f1(tot["tp"], tot["fp"], tot["fn"])
    extracted = sum(r["score"]["extracted_values"] for r in rows)
    out["halluc"] = sum(r["score"]["hallucinated"] for r in rows) / extracted if extracted else 0.0
    out["errors"] = dict(collections.Counter(r["error"] for r in rows if r["error"]))
    out["latency_s"] = sum(r["latency_s"] for r in rows) / n
    return out


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def delta(x: float, base: float, lower_is_better: bool = False) -> str:
    d = 100 * (x - base)
    good = d < 0 if lower_is_better else d > 0
    s = f"{d:+.1f}"
    return f"**{s}**" if good and abs(d) >= 0.05 else s


def quantile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


METRICS = (  # (column, aggregate key, lower is better)
    ("JSON ok", "json_ok", False), ("scalars", "scalars", False),
    ("exp F1", "experiencias", False), ("exp dates", "experiencia_datas", False),
    ("edu F1", "formacao", False), ("skills F1", "habilidades", False),
    ("lang F1", "idiomas", False), ("halluc.", "halluc", True),
)
LEGEND = [
    "Percentages. *JSON ok*: parsed and schema-valid, not truncated. *Scalars*: mean",
    "exact-match over " + ", ".join(SCALARS) + ". List columns: micro F1; *exp dates* is",
    "scored on matched experiences only. *Halluc.*: grounded values absent from the",
    "source text (lower is better). *Δ*: tuned minus base in points, bold when tuned wins.",
]


def training_section(run: str) -> list[str]:
    logs = {}
    for path in sorted(Path("outputs").glob(f"*/{run}/train_log.json")):
        log = json.loads(path.read_text(encoding="utf-8"))
        logs[log["modality"]] = (path, log)
    if not logs:
        return []
    mods = list(logs)
    lines = ["", "## Training", "",
             "QLoRA (4-bit base, r=16) for one epoch, loss on the assistant turn only. Read from",
             *[f"`{path}`" + ("," if i < len(mods) - 1 else ".") for i, (path, _) in
               enumerate(logs.values())],
             "",
             "| modality | rows | train tokens | loss tokens | steps | trainable params "
             "| runtime | s/step | peak VRAM | first loss | last loss | mean loss |",
             "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    curves = {}
    for mod, (_, log) in logs.items():
        steps = [h for h in log["log_history"] if "loss" in h]
        curves[mod] = {h["step"]: h for h in steps}
        lines.append("| " + " | ".join([
            mod, f"{log['train_rows']:,}", f"{log['train_tokens'] / 1e6:.2f}M",
            f"{log['loss_tokens'] / 1e6:.2f}M", str(log["total_steps"]),
            f"{log['trainable_params'] / 1e6:.1f}M", f"{log['train_runtime_min']:.0f} min",
            f"{60 * log['train_runtime_min'] / log['total_steps']:.1f}",
            f"{log['peak_reserved_vram_gb']:.2f} GB", f"{steps[0]['loss']:.4f}",
            f"{steps[-1]['loss']:.4f}", f"{log['final_train_loss']:.4f}"]) + " |")
    lines += ["", "### Loss curve", "",
              f"Training loss every {next(iter(curves[mods[0]]))} optimiser steps "
              "(no eval loss: in-training eval OOMs the 8GB card).", "",
              "| step | epoch | " + " | ".join(mods) + " |",
              "|--:|--:|" + "--:|" * len(mods)]
    for step in sorted({s for c in curves.values() for s in c}):
        hs = [curves[m].get(step) for m in mods]
        epoch = next(h["epoch"] for h in hs if h)
        lines.append(f"| {step} | {epoch:.2f} | "
                     + " | ".join(f"{h['loss']:.4f}" if h else "--" for h in hs) + " |")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--smoke", action="store_true", help="report the smoke run")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    data = load_data_config(args.data_config)
    run = data.smoke.run if args.smoke else data.generate.run
    out = Path(args.out or ("docs/RESULTS.md" if not args.smoke else f"outputs/report-{run}.md"))

    groups: dict[tuple, list[dict]] = collections.defaultdict(list)
    # Exactly eval-base / eval-tuned: partial runs and backups beside them would double-count.
    sources = sorted(p for which in ("base", "tuned")
                     for p in Path("outputs").glob(f"*/{run}/eval-{which}.jsonl"))
    for path in sources:
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            groups[(r["split"], r["modality"], r["which"], r["mode"])].append(r)
    if not groups:
        print(f"FATAL: no outputs/*/{run}/eval-{{base,tuned}}.jsonl", file=sys.stderr)
        return 1

    def order(key: tuple) -> tuple:
        split, modality, which, mode = key
        return (SPLIT_ORDER.index(split) if split in SPLIT_ORDER else 9, modality,
                which != "base", mode)

    keys = sorted(groups, key=order)
    aggs = {k: aggregate(groups[k]) for k in keys}
    splits = sorted({k[0] for k in keys},
                    key=lambda s: SPLIT_ORDER.index(s) if s in SPLIT_ORDER else 9)
    modalities = sorted({k[1] for k in keys})
    headline = splits[0]

    lines = [
        f"# Results -- run `{run}`",
        "",
        "Generated by `scripts/07_report.py`; do not edit by hand. Sources:",
        "",
        *[f"- `{p}`" for p in sources],
        "",
        "Contents: [Summary](#summary) · [Training](#training) · "
        "[Accuracy](#accuracy-base-vs-tuned) · [Processing time](#processing-time) · "
        "[Scalar fields](#scalar-fields) · [Format failures](#format-failures) · "
        "[Per template](#per-template-grammar-mode)",
        "",
        "## Summary",
        "",
        f"`{headline}`, grammar-constrained decoding (the production setting); "
        "`real_test` is the headline when present. Base → tuned.",
        "",
        "| modality | JSON ok | scalars | exp F1 | edu F1 | skills F1 | halluc. | s/req |",
        "|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for mod in modalities:
        b, t = aggs.get((headline, mod, "base", "grammar")), aggs.get((headline, mod, "tuned", "grammar"))
        if not (b and t):
            continue
        def cell(old: str, new: str, better: bool) -> str:
            return f"{old} → **{new}**" if better else f"{old} → {new}"

        cells = [cell(pct(b[k]), pct(t[k]), round(100 * t[k], 1) > round(100 * b[k], 1))
                 for k in ("json_ok", "scalars", "experiencias", "formacao", "habilidades")]
        cells.append(cell(pct(b["halluc"]), pct(t["halluc"]),
                          round(100 * t["halluc"], 1) < round(100 * b["halluc"], 1)))
        cells.append(cell(f"{b['latency_s']:.1f}", f"{t['latency_s']:.1f}",
                          t["latency_s"] < b["latency_s"]))
        lines.append(f"| {mod} | " + " | ".join(cells) + " |")

    lines += training_section(run)

    lines += ["", "## Accuracy: base vs tuned", "", *LEGEND]
    for split in splits:
        n = next(aggs[k]["n"] for k in keys if k[0] == split)
        lines += ["", f"### `{split}` (n={n})", "",
                  "| modality | mode | model | " + " | ".join(c for c, _, _ in METRICS) + " |",
                  "|---|---|---|" + "--:|" * len(METRICS)]
        for mod in modalities:
            for mode in ("grammar", "free"):
                b, t = aggs.get((split, mod, "base", mode)), aggs.get((split, mod, "tuned", mode))
                for which, a in (("base", b), ("tuned", t)):
                    if a:
                        lines.append(f"| {mod} | {mode} | {which} | "
                                     + " | ".join(pct(a[k]) for _, k, _ in METRICS) + " |")
                if b and t:
                    lines.append(f"| {mod} | {mode} | *Δ* | "
                                 + " | ".join(delta(t[k], b[k], low) for _, k, low in METRICS)
                                 + " |")

    lines += ["", "## Processing time", "",
              "Wall-clock per request against llama-server (Q4_K_M, RTX 3070 Ti, 4 slots kept",
              "busy, so requests share the GPU), all splits pooled. *Prefill tok*: prompt tokens",
              "the server evaluated, excluding any prefix reused from its cache (vision includes",
              "the page images). *Out tok/s*: output tokens over",
              "the whole request latency, prefill included. *Speedup*: base mean / tuned mean.",
              "",
              "| modality | mode | model | n | mean s | p50 s | p95 s | prefill tok | out tok "
              "| out tok/s | speedup |",
              "|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for mod in modalities:
        for mode in ("grammar", "free"):
            pooled = {w: [r for k in keys if k[1:] == (mod, w, mode) for r in groups[k]]
                      for w in ("base", "tuned")}
            mean = {w: sum(r["latency_s"] for r in rs) / len(rs) for w, rs in pooled.items() if rs}
            for which, rows in pooled.items():
                if not rows:
                    continue
                lat = [r["latency_s"] for r in rows]
                prefill = sum(r["prompt_n"] for r in rows) / len(rows)
                outn = sum(r["predicted_n"] for r in rows) / len(rows)
                speed = (f"{mean['base'] / mean['tuned']:.2f}x"
                         if which == "tuned" and "base" in mean else "--")
                lines.append("| " + " | ".join([
                    mod, mode, which, str(len(rows)), f"{mean[which]:.2f}",
                    f"{quantile(lat, 0.5):.2f}", f"{quantile(lat, 0.95):.2f}", f"{prefill:.0f}",
                    f"{outn:.0f}", f"{sum(r['predicted_n'] for r in rows) / sum(lat):.1f}",
                    speed]) + " |")

    lines += ["", "## Scalar fields", "",
              "| split | modality | model | mode | " + " | ".join(SCALARS) + " |",
              "|---|---|---|---|" + "--:|" * len(SCALARS)]
    for k in keys:
        lines.append("| " + " | ".join([*k, *(pct(aggs[k][n]) for n in SCALARS)]) + " |")

    lines += ["", "## Format failures", "", "| split | modality | model | mode | errors |",
              "|---|---|---|---|---|"]
    for k in keys:
        errs = ", ".join(f"{e}: {c}" for e, c in sorted(aggs[k]["errors"].items())) or "--"
        lines.append("| " + " | ".join([*k, errs]) + " |")

    by_template: dict[tuple, list[dict]] = collections.defaultdict(list)
    for (split, modality, which, mode), rows in groups.items():
        if mode == "grammar":
            for r in rows:
                by_template[(r["template"], modality, which)].append(r)
    lines += ["", "## Per template (grammar mode)", "",
              "| template | modality | model | n | scalars | exp F1 | exp dates | edu F1 |",
              "|---|---|---|--:|--:|--:|--:|--:|"]
    for k in sorted(by_template, key=lambda k: (k[0], k[1], k[2] != "base")):
        a = aggregate(by_template[k])
        lines.append("| " + " | ".join([*k, str(a["n"]), pct(a["scalars"]),
                                         pct(a["experiencias"]), pct(a["experiencia_datas"]),
                                         pct(a["formacao"])]) + " |")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f">> wrote {out}")
    summary = lines.index("## Summary")
    print("\n".join(lines[summary: lines.index("", summary + 6)]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
