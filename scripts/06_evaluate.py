#!/usr/bin/env python
"""Stage 6 -- score base or tuned on test_seen, test_unseen, their scanned variants (and, in
phase 4, real_test).

Owns the server lifecycle: starts scripts/05_serve.sh with the model's GGUF, checks via
GET /props that the server loaded exactly that file, scores, stops it. Base and tuned are
separate invocations, so they never share the GPU. It refuses to start if something
already answers on the port. Raw /completion only (ADR 0011).

Each model is scored twice (ADR 0007):
- free decoding -- measures whether the model learnt the format (JSON validity);
- grammar-constrained (json_schema from cvx.schema) -- isolates field accuracy.

A generation that does not parse is scored against an empty resume, so format failures
count as wrong fields rather than vanishing from the averages.

Hallucination is always checked against the clean PDF's extracted text, also on the
``*_scan`` splits: a scan does not change what the resume says, only what the model sees.
A split whose jsonl is missing (dataset built before ``scan_test``) is skipped with a note.

Vision requests send the page PNGs as base64 ``multimodal_data`` and the prompt through
cvx.prompt.to_server_prompt with the server's media marker from GET /props. The pages are
aligned to 32px at stage 2, so server and HF tokenise them identically; each row records
the HF prompt length (``n_prompt``) next to the server's ``prompt_n`` + ``cache_n``, and
the run reports how many differ.

Refuses to run when the dataset manifest's prompt/schema version differs from cvx.prompt.
Writes <outputs_dir>/<run>/eval-<which>.jsonl, one row per (resume, mode).
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx  # noqa: E402

from cvx.config import gguf_path, load_data_config, load_model_config, mmproj_path  # noqa: E402
from cvx.metrics import score  # noqa: E402
from cvx.generate.rasterize import image_tokens  # noqa: E402
from cvx.prompt import manifest, parse_generation, render_prompt, to_server_prompt  # noqa: E402
from cvx.schema import Curriculo, json_schema  # noqa: E402

EMPTY = Curriculo(nome="")
MODES = ("free", "grammar")


class Server:
    """One llama-server for the duration of a `with` block."""

    def __init__(self, model: Path, mmproj: Path | None, port: int, ctx: int, parallel: int,
                 log: Path) -> None:
        self.model, self.mmproj, self.port = model, mmproj, port
        self.ctx, self.parallel, self.log = ctx, parallel, log
        self.url = f"http://127.0.0.1:{port}"
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> "Server":
        try:
            httpx.get(f"{self.url}/health", timeout=2)
            raise SystemExit(f"FATAL: something already serves {self.url}; stop it first "
                             "(base and tuned must never share the GPU)")
        except httpx.TransportError:
            pass
        cmd = ["bash", "scripts/05_serve.sh", str(self.model), str(self.mmproj or ""),
               str(self.port), str(self.ctx), str(self.parallel)]
        print(">>", " ".join(cmd), flush=True)
        self.proc = subprocess.Popen(cmd, stdout=self.log.open("w"), stderr=subprocess.STDOUT)
        deadline = time.time() + 600
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise SystemExit(f"FATAL: llama-server exited with {self.proc.returncode}; "
                                 f"see {self.log}")
            try:
                if httpx.get(f"{self.url}/health", timeout=2).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            time.sleep(1)
        else:
            raise SystemExit(f"FATAL: llama-server not healthy after 600s; see {self.log}")
        loaded = Path(httpx.get(f"{self.url}/props", timeout=10).json()["model_path"])
        if loaded.resolve() != self.model.resolve():
            raise SystemExit(f"FATAL: server loaded {loaded}, expected {self.model}")
        return self

    def __exit__(self, *exc) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def resumable_rows(path: Path, model: Path, model_mtime: float) -> list[dict]:
    """Rows of a previous run of THIS GGUF, deduplicated; rewrites the file with only them.

    A row counts only if it names the same GGUF and was written for this exact file: rows
    carry ``model_mtime``; older rows without it are accepted only when the jsonl is newer
    than the GGUF (a re-export after retraining replaces the file and its mtime). A torn
    last line from a crash is dropped.
    """
    if not path.exists():
        return []
    legacy_ok = path.stat().st_mtime >= model_mtime
    rows: dict[tuple, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        same = r.get("model") == model.name and (
            r.get("model_mtime") == model_mtime if "model_mtime" in r else legacy_ok)
        if same:
            rows[(r["id"], r["split"], r["mode"])] = r
    kept = list(rows.values())
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept),
                    encoding="utf-8")
    return kept


def hf_prompt_tokens(prompt: str, images: list[str], tokenizer) -> int:
    """Prompt length as the HF processor sees it: each page expands to its 32px grid."""
    n = len(tokenizer(prompt, add_special_tokens=False).input_ids)
    return n + sum(image_tokens(Path(p)) - 1 for p in images)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--which", choices=("base", "tuned"), required=True)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--smoke", action="store_true", help="evaluate the smoke run")
    parser.add_argument("--splits",
                        default="test_seen,test_unseen,test_seen_scan,test_unseen_scan")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--limit", type=int, default=None, help="first N rows per split")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--resume", action="store_true",
                        help="keep rows already in eval-<which>.jsonl from this same GGUF and "
                             "run only the missing (resume, split, mode) jobs")
    args = parser.parse_args()
    cfg = load_model_config(args.config)
    data = load_data_config(args.data_config)
    modality = cfg.input.modality
    modes = [m for m in args.modes.split(",") if m]
    if set(modes) - set(MODES):
        parser.error(f"--modes must be a subset of {MODES}")

    run = data.smoke.run if args.smoke else data.generate.run
    data_dir = Path(data.paths.datasets_dir) / run / modality
    ds_manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if {k: ds_manifest[k] for k in manifest()} != manifest():
        print(f"FATAL: {data_dir} was built under {ds_manifest}, code is at {manifest()}",
              file=sys.stderr)
        return 1

    model = gguf_path(cfg, args.which, run)
    if not model.exists():
        print(f"FATAL: {model} not found; run `make export` first", file=sys.stderr)
        return 1
    out_dir = Path(cfg.paths.outputs_dir) / run
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval-{args.which}.jsonl"

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(cfg.model.base_id)
    # Text renders with the tokenizer, vision with the processor -- as in stages 2 and 3.
    template_owner = processor.tokenizer if modality == "text" else processor
    cvs_dir = Path(data.paths.cvs_dir) / run
    jobs = []
    for split in args.splits.split(","):
        if not (data_dir / f"{split}.jsonl").exists():
            print(f">> skipping {split}: no {data_dir / split}.jsonl (rebuild with `make build`)",
                  flush=True)
            continue
        rows = load_rows(data_dir / f"{split}.jsonl")[: args.limit]
        for row in rows:
            # Text prompts carry no image block; vision swaps it for the media marker below.
            prompt = render_prompt(row["messages"], template_owner)
            row["n_prompt"] = hf_prompt_tokens(prompt, row.get("images", []),
                                               processor.tokenizer)
            source = (cvs_dir / f"{row['id']}.txt").read_text(encoding="utf-8")
            for mode in modes:
                jobs.append((row, split, mode, prompt, source))
    model_mtime = model.stat().st_mtime
    kept = resumable_rows(out_path, model, model_mtime) if args.resume else []
    done_keys = {(r["id"], r["split"], r["mode"]) for r in kept}
    jobs = [j for j in jobs if (j[0]["id"], j[1], j[2]) not in done_keys]
    if args.resume:
        print(f">> resume: keeping {len(kept)} rows from {out_path}", flush=True)
    print(f">> {len(jobs)} requests ({args.which}, {run}/{modality}, modes={modes})", flush=True)

    schema = json_schema()
    media_marker: str | None = None  # set once the server is up

    def request(client: httpx.Client, job) -> dict:
        row, split, mode, prompt, source = job
        if modality == "vision":
            pages = [base64.b64encode(Path(p).read_bytes()).decode() for p in row["images"]]
            payload = {"prompt_string": to_server_prompt(prompt, media_marker),
                       "multimodal_data": pages}
        else:
            payload = prompt
        body = {"prompt": payload, "temperature": cfg.decode.temperature,
                "n_predict": cfg.decode.max_tokens, "seed": 3407}
        if mode == "grammar":
            body["json_schema"] = schema
        started = time.time()
        r = client.post("/completion", json=body)
        r.raise_for_status()
        res = r.json()
        pred, error = parse_generation(res["content"])
        if error is None and res.get("stop_type") == "limit":
            error = "truncated"  # parsed by luck at the token limit -- still not trustworthy
        gold = Curriculo.model_validate(row["gold"])
        s = score(gold, pred if pred is not None else EMPTY, source_text=source)
        return {
            "id": row["id"], "split": split, "template": row["template"], "mode": mode,
            "which": args.which, "run": run, "modality": modality, "model": model.name,
            "model_mtime": model_mtime,
            "error": error, "raw": res["content"] if error else None,
            "pred": pred.model_dump() if pred is not None else None,
            "score": dataclasses.asdict(s),
            "n_prompt": row["n_prompt"],
            "prompt_n": res["timings"]["prompt_n"], "cache_n": res["timings"].get("cache_n"),
            "predicted_n": res["timings"]["predicted_n"],
            "latency_s": round(time.time() - started, 2),
        }

    started = time.time()
    with Server(model, mmproj_path(cfg), args.port, cfg.serve.ctx, cfg.serve.parallel,
                out_dir / f"serve-{args.which}.log") as server, \
            httpx.Client(base_url=server.url, timeout=900) as client, \
            out_path.open("a" if args.resume else "w", encoding="utf-8") as f, \
            cf.ThreadPoolExecutor(cfg.serve.parallel) as pool:
        if modality == "vision":
            media_marker = client.get("/props").json()["media_marker"]
        done, errors, drift = 0, 0, 0
        for result in pool.map(lambda j: request(client, j), jobs):
            drift += result["prompt_n"] + (result["cache_n"] or 0) != result["n_prompt"]
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()  # a crash keeps every finished row; --resume picks up from there
            done += 1
            errors += result["error"] is not None
            if done % 20 == 0 or done == len(jobs):
                print(f">> {done}/{len(jobs)} ({done / (time.time() - started):.2f}/s, "
                      f"format errors={errors}, prompt-length drift={drift})", flush=True)
    print(f">> wrote {out_path} in {(time.time() - started) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
