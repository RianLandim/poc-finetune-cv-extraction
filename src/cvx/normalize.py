"""Canonical forms for comparison. Used by metrics only -- never to rewrite a target.

A prediction is right when it matches the gold value *after* normalisation, so that
"São Paulo" vs "sao paulo" or "(11) 98765-4321" vs "11987654321" are not scored as errors.
"""

from __future__ import annotations

import re
import unicodedata

MONTHS = {
    "jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6,
    "jul": 7, "ago": 8, "set": 9, "out": 10, "nov": 11, "dez": 12,
}
_CURRENT = ("atual", "o momento", "presente", "hoje", "atualmente", "em andamento", "cursando")

_YM_ISO = re.compile(r"^(\d{4})-(\d{1,2})$")
_MY_SLASH = re.compile(r"^(\d{1,2})[/.-](\d{4})$")
_MONTH_NAME = re.compile(r"^([a-z]{3})[a-z]*\.?(?:\s+de)?\s*[/ ]\s*(\d{4})$")
_YEAR = re.compile(r"^(\d{4})$")


def text(value: str | None) -> str:
    """Casefold, strip accents and punctuation at the edges, collapse whitespace."""
    if value is None:
        return ""
    decomposed = unicodedata.normalize("NFKD", value)
    no_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", no_accents).strip().strip(".,;:").casefold()


def phone(value: str | None) -> str:
    """Digits only, without the +55 country code."""
    digits = re.sub(r"\D", "", value or "")
    return digits[2:] if digits.startswith("55") and len(digits) > 11 else digits


def email(value: str | None) -> str:
    return (value or "").strip().casefold()


def date(value: str | None) -> str | None:
    """'03/2021', 'mar 2021', 'março de 2021', '2021-3' -> '2021-03'; '2021' -> '2021'.

    Words meaning "current" become None, matching the schema's open-ended convention.
    Unparseable input is returned normalised as text so it still compares, and fails.
    """
    if value is None:
        return None
    v = text(value)
    if not v or v in _CURRENT:
        return None
    if m := _YM_ISO.match(v):
        return f"{m[1]}-{int(m[2]):02d}"
    if m := _MY_SLASH.match(v):
        return f"{m[2]}-{int(m[1]):02d}"
    if (m := _MONTH_NAME.match(v)) and m[1] in MONTHS:
        return f"{m[2]}-{MONTHS[m[1]]:02d}"
    if m := _YEAR.match(v):
        return m[1]
    return v
