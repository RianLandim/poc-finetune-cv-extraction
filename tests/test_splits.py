import pytest

from cvx.splits import SPLITS, assign, unseen_templates

TEMPLATES = [f"tpl{i:02d}" for i in range(20)]


def _assign_all(n=3000, seed=3407):
    unseen = unseen_templates(TEMPLATES, 0.2, seed)
    rows = [(f"p{i}", TEMPLATES[i % len(TEMPLATES)]) for i in range(n)]
    return unseen, [(p, t, assign(p, t, unseen, 0.05, 0.05, seed)) for p, t in rows]


def test_holdout_size_and_determinism():
    a = unseen_templates(TEMPLATES, 0.2, 3407)
    assert len(a) == 4
    assert a == unseen_templates(list(reversed(TEMPLATES)), 0.2, 3407)


def test_no_unseen_template_reaches_training():
    unseen, rows = _assign_all()
    for _, template, split in rows:
        assert (split == "test_unseen") == (template in unseen)


def test_every_split_is_populated():
    _, rows = _assign_all()
    assert {split for _, _, split in rows} == set(SPLITS)


def test_persona_split_is_stable():
    unseen = unseen_templates(TEMPLATES, 0.2, 1)
    seen = next(t for t in TEMPLATES if t not in unseen)
    assert assign("p42", seen, unseen, 0.05, 0.05, 1) == assign("p42", seen, unseen, 0.05, 0.05, 1)


def test_bad_holdout_rejected():
    with pytest.raises(ValueError):
        unseen_templates(TEMPLATES, 0.0, 1)
