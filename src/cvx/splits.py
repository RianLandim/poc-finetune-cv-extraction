"""Deterministic train/val/test assignment (ADR 0004).

Two axes, both must hold:
- **Template:** a fixed fraction of templates is held out entirely. Every resume rendered
  with one goes to ``test_unseen``. Without this the score measures memorised layouts.
- **Persona:** among seen templates, a persona lands in exactly one split, by hash, so the
  same person never appears in train and test under two layouts.

Generation renders exactly ONE resume per persona, so a persona sent to ``test_unseen``
by its template cannot also appear in train.
"""

from __future__ import annotations

import hashlib

SPLITS = ("train", "val", "test_seen", "test_unseen")
# The same test resumes, printed and scanned (ADR 0006): derived rows, not a new assignment.
SCAN_SPLITS = {"test_seen": "test_seen_scan", "test_unseen": "test_unseen_scan"}


def _unit(key: str, seed: int) -> float:
    """Stable value in [0, 1) -- independent of PYTHONHASHSEED and of row order."""
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def unseen_templates(template_ids: list[str], holdout_frac: float, seed: int) -> set[str]:
    if not 0 < holdout_frac < 1:
        raise ValueError("holdout_frac must be in (0, 1)")
    ranked = sorted(template_ids, key=lambda t: _unit(f"tpl:{t}", seed))
    k = max(1, round(len(template_ids) * holdout_frac))
    return set(ranked[:k])


def assign(persona_id: str, template_id: str, unseen: set[str],
           val_frac: float, test_frac: float, seed: int) -> str:
    if template_id in unseen:
        return "test_unseen"
    u = _unit(f"persona:{persona_id}", seed)
    if u < test_frac:
        return "test_seen"
    if u < test_frac + val_frac:
        return "val"
    return "train"
