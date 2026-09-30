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
