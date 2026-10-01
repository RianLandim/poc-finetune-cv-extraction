import json

import pytest
from pydantic import ValidationError

from cvx.schema import Curriculo, json_schema, to_target


def test_target_roundtrips(cv):
    assert Curriculo.model_validate(json.loads(to_target(cv))) == cv


def test_target_is_compact_and_keeps_accents(cv):
    target = to_target(cv)
    assert ", " not in target.split('"resumo"')[0]  # compact separators
    assert "Florianópolis" in target                # ensure_ascii=False


@pytest.mark.parametrize("field", ["cpf", "rg", "data_nascimento"])
def test_protected_fields_are_rejected(cv, field):
    raw = cv.model_dump() | {field: "x"}
    with pytest.raises(ValidationError):
        Curriculo.model_validate(raw)


@pytest.mark.parametrize("bad", ["03/2021", "2021-13", "2021-3", "março 2021"])
def test_dates_must_be_canonical(cv, bad):
    raw = cv.model_dump()
    raw["experiencias"][0]["inicio"] = bad
    with pytest.raises(ValidationError):
        Curriculo.model_validate(raw)


def test_json_schema_has_no_protected_fields():
    props = json_schema()["properties"]
    assert not {"cpf", "rg", "data_nascimento"} & set(props)


def _patterns(node):
    if isinstance(node, dict):
        if "pattern" in node:
            yield node["pattern"]
        for v in node.values():
            yield from _patterns(v)
    elif isinstance(node, list):
        for v in node:
            yield from _patterns(v)


def test_patterns_are_grammar_convertible():
    """llama.cpp's json_schema -> grammar converter knows only a few escapes; on \\d, \\w,
    \\s it drops the pattern and accepts ANY string, silently disabling grammar-mode date
    checks (found in the phase 2 smoke eval)."""
    patterns = list(_patterns(json_schema()))
    assert patterns
    for p in patterns:
        assert p.startswith("^") and p.endswith("$"), p
        assert not any(f"\\{c}" in p for c in "dDwWsSb"), p


def test_grammar_requires_every_key_in_target_order(cv):
    """Optional keys in a llama.cpp grammar can be skipped but never reordered: a model
    that writes "habilidades" first is then locked out of "experiencias" (found in the
    phase 2 smoke: two-column resumes scored 0 on experiences under grammar). The grammar
    must demand exactly the target's keys, in the target's order."""
    schema = json_schema()
    for obj in [schema, *schema["$defs"].values()]:
        assert obj["required"] == list(obj["properties"]), obj["title"]
    target = json.loads(to_target(cv))
    assert list(target) == list(schema["properties"])
    exp = schema["$defs"]["Experiencia"]["properties"]
    assert list(target["experiencias"][0]) == list(exp)
