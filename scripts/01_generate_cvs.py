#!/usr/bin/env python
"""Stage 1 -- personas -> gold JSON -> PDF + extracted text.

For each sampled persona: seed.skeleton() -> teacher.fill_prose() -> render.render_pdf()
-> extract_text.pdf_to_text(). Writes, per resume, under data/cvs/<run>/:

    cv_000123.json       gold label (cvx.schema.Curriculo)
    cv_000123.pdf        the rendered resume
    cv_000123.txt        pdfplumber text -- the text modality's input
    cv_000123.meta.json  persona uuid, template, sampled style, pages, teacher stats

plus manifest.json (prompt/schema versions, templates, counts).

Resumable: a resume whose .meta.json exists is skipped, so a crash or a teacher outage
costs only the resumes in flight. The meta file is written last for exactly that reason.

`datasets` streaming aborts the process at interpreter shutdown (lesson from the persona
repo): the script flushes and exits via os._exit.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvx.config import load_data_config  # noqa: E402
from cvx.generate.render import available_templates  # noqa: E402
from cvx.prompt import manifest as prompt_manifest  # noqa: E402

_client = None  # one httpx client per worker process


def _work(i: int, persona: dict, out_dir: str, gen, templates: list[str],
          teacher_url: str | None) -> dict:
    global _client
    import httpx

    from cvx.generate.extract_text import pdf_to_text
    from cvx.generate.render import pick_template, render_pdf, sample_style, style_dict
    from cvx.generate.seed import skeleton
    from cvx.generate.teacher import fill_prose

    stem = Path(out_dir) / f"cv_{i:06d}"
    meta_path = stem.with_suffix(".meta.json")
    if meta_path.exists():
        return {**json.loads(meta_path.read_text(encoding="utf-8")), "skipped": True}

    seeded = skeleton(persona, gen.seed, gen.p_show)
    cv, teacher_stats = seeded.cv, {"called": False}
    if teacher_url:
        if _client is None:
            _client = httpx.Client(base_url=teacher_url, timeout=300)
        cv, teacher_stats = fill_prose(cv, persona, seeded.plan, gen.teacher, _client)

    template = pick_template(templates, gen.seed, persona["uuid"])
    style = sample_style(template, gen.seed, persona["uuid"])
    n_pages = render_pdf(cv, style, stem.with_suffix(".pdf"))
    stem.with_suffix(".txt").write_text(pdf_to_text(stem.with_suffix(".pdf")), encoding="utf-8")
    stem.with_suffix(".json").write_text(
        json.dumps(cv.model_dump(), ensure_ascii=False, indent=1), encoding="utf-8")
    meta = {
        "id": stem.name,
        "persona_uuid": persona["uuid"],
        "template": template,
        "n_pages": n_pages,
        "style": style_dict(style),
        "teacher": teacher_stats,
        "persona": {k: persona[k] for k in ("sex", "age", "education_level", "occupation",
                                            "municipality", "state")},
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


def _personas(src, n: int):
    from datasets import load_dataset

    stream = load_dataset(src.hf_dataset, split="train", streaming=True)
    out = []
    for row in stream:
        if src.min_age <= int(row["age"]) <= src.max_age:
            out.append(row)
            if len(out) == n:
                break
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--smoke", action="store_true", help="smoke.n_resumes into smoke.run")
    parser.add_argument("--teacher-url", default=None,
                        help='override generate.teacher.base_url; "none" skips the teacher')
    parser.add_argument("--n", type=int, default=None, help="override the resume count")
    parser.add_argument("--run", default=None, help="override the output run name")
    parser.add_argument("--templates", default=None,
                        help="comma-separated subset of templates (difficulty probes)")
    args = parser.parse_args()

    cfg = load_data_config(args.config)
    gen = cfg.generate
    n = args.n or (cfg.smoke.n_resumes if args.smoke else gen.n_resumes)
    run = args.run or (cfg.smoke.run if args.smoke else gen.run)
    teacher_url = args.teacher_url or gen.teacher.base_url
    teacher_url = None if teacher_url == "none" else teacher_url
    templates = available_templates()
    if args.templates:
        unknown = set(args.templates.split(",")) - set(templates)
        if unknown:
            sys.exit(f"FATAL: unknown templates {sorted(unknown)}")
        templates = sorted(args.templates.split(","))
    out_dir = Path(cfg.paths.cvs_dir) / run
    out_dir.mkdir(parents=True, exist_ok=True)

    if teacher_url:
        import httpx

        try:
            httpx.get(f"{teacher_url}/health", timeout=5).raise_for_status()
        except httpx.HTTPError as exc:
            sys.exit(f"FATAL: teacher at {teacher_url} is not up ({exc}). Start one with "
                     "`make teacher`, or pass --teacher-url none.")

    print(f">> {n} resumes -> {out_dir} | templates={templates} | teacher={teacher_url}",
          flush=True)
    personas = _personas(cfg.source, n)
    print(f">> {len(personas)} personas loaded", flush=True)

    t0 = time.time()
    metas, failures = [], 0
    with ProcessPoolExecutor(max_workers=gen.workers) as pool:
        futures = {pool.submit(_work, i, p, str(out_dir), gen, templates, teacher_url): i
                   for i, p in enumerate(personas)}
        for k, fut in enumerate(as_completed(futures), 1):
            try:
                metas.append(fut.result())
            except Exception as exc:  # one broken resume must not kill a 6000-row run
                failures += 1
                print(f"!! cv_{futures[fut]:06d}: {type(exc).__name__}: {exc}", flush=True)
            if k % 50 == 0 or k == len(futures):
                rate = k / (time.time() - t0)
                print(f">> {k}/{len(futures)} ({rate:.2f}/s, failures={failures})", flush=True)

    teacher_calls = [m["teacher"] for m in metas if m["teacher"].get("called")]
    manifest = {
        **prompt_manifest(),
        "run": run,
        "n_requested": n,
        "n_written": len(metas),
        "n_failed": failures,
        "seed": gen.seed,
        "templates": templates,
        "per_template": dict(collections.Counter(m["template"] for m in metas)),
        "pages": dict(collections.Counter(str(m["n_pages"]) for m in metas)),
        "teacher": {
            "url": teacher_url,
            "calls": len(teacher_calls),
            "errors": sum("error" in t for t in teacher_calls),
            "desc_asked": sum(t.get("desc_asked", 0) for t in teacher_calls),
            "desc_kept": sum(t.get("desc_kept", 0) for t in teacher_calls),
            "resumo_kept": sum(bool(t.get("resumo_kept")) for t in teacher_calls),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1 if failures else 0)


if __name__ == "__main__":
    main()
