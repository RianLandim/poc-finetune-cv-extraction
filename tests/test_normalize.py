import pytest

from cvx import normalize as n


@pytest.mark.parametrize("raw, expected", [
    ("2021-03", "2021-03"),
    ("2021-3", "2021-03"),
    ("03/2021", "2021-03"),
    ("3.2021", "2021-03"),
    ("mar 2021", "2021-03"),
    ("Março de 2021", "2021-03"),
    ("mar/2021", "2021-03"),
    ("2017", "2017"),
    ("Atual", None),
    ("o momento", None),
    (None, None),
])
def test_date(raw, expected):
    assert n.date(raw) == expected


def test_phone_drops_formatting_and_country_code():
    assert n.phone("(48) 99876-5432") == n.phone("+55 48 99876 5432") == "48998765432"


def test_text_ignores_case_accents_and_spacing():
    assert n.text("  São   Paulo. ") == n.text("sao paulo")
