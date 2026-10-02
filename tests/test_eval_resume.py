"""scripts/06_evaluate.py --resume keeps only rows from the same GGUF (no server needed)."""

import importlib.util
import json
import os
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "evaluate", Path(__file__).resolve().parents[1] / "scripts" / "06_evaluate.py")
evaluate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate)


def _row(i, mode="grammar", **kw):
    return {"id": f"cv_{i:06d}", "split": "test_unseen", "mode": mode, "model": "m.gguf", **kw}


def test_keeps_same_model_drops_torn_line_and_dedupes(tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    mt = model.stat().st_mtime
    out = tmp_path / "eval-tuned.jsonl"
    lines = [_row(1, model_mtime=mt), _row(1, model_mtime=mt), _row(2, model_mtime=mt - 1),
             _row(3, model_mtime=mt) | {"model": "other.gguf"}]
    out.write_text("".join(json.dumps(r) + "\n" for r in lines) + '{"id": "cv_0000', "utf-8")
    kept = evaluate.resumable_rows(out, model, mt)
    assert [r["id"] for r in kept] == ["cv_000001"]
    assert out.read_text("utf-8").count("\n") == 1  # file rewritten without the junk


def test_legacy_rows_need_a_jsonl_newer_than_the_gguf(tmp_path):
    model = tmp_path / "m.gguf"
    out = tmp_path / "eval-base.jsonl"
    out.write_text(json.dumps(_row(1)) + "\n", "utf-8")
    model.write_bytes(b"x")
    os.utime(out, (1, 1))  # jsonl older than the GGUF: a re-export happened since
    assert evaluate.resumable_rows(out, model, model.stat().st_mtime) == []
    out.write_text(json.dumps(_row(1)) + "\n", "utf-8")  # now newer
    assert len(evaluate.resumable_rows(out, model, model.stat().st_mtime)) == 1
