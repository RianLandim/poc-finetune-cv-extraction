"""The ONLY definition of the extraction target (ADR 0005).

Generation writes it, training serialises it, evaluation parses back into it. Changing a
field here changes the task: bump ``SCHEMA_VERSION`` and regenerate the dataset.

Deliberately absent: CPF, RG, date of birth, photo, marital status. The model must never
learn to pull government IDs or protected attributes out of a resume.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0.0"

# "YYYY-MM", or "YYYY" when the resume only gives a year. [0-9], not \d: llama.cpp's
# json_schema -> grammar converter rejects \d and then silently accepts ANY string.
DATE_PATTERN = r"^[0-9]{4}(-(0[1-9]|1[0-2]))?$"

NivelFormacao = Literal[
    "fundamental", "medio", "tecnico", "graduacao", "pos", "mestrado", "doutorado"
]
NivelIdioma = Literal["basico", "intermediario", "avancado", "fluente", "nativo"]


class _Strict(BaseModel):
    # A generation with an unknown key is a format failure, not something to ignore.
    model_config = ConfigDict(extra="forbid")


class Experiencia(_Strict):
    empresa: str
    cargo: str
    inicio: str = Field(pattern=DATE_PATTERN)
    fim: str | None = Field(default=None, pattern=DATE_PATTERN)  # None = emprego atual
    cidade: str | None = None
    descricao: str | None = None


class Formacao(_Strict):
    instituicao: str
    curso: str
    nivel: NivelFormacao
    conclusao: str | None = Field(default=None, pattern=DATE_PATTERN)  # None = em andamento


class Idioma(_Strict):
    idioma: str
    nivel: NivelIdioma


class Curriculo(_Strict):
    nome: str
    email: str | None = None
    telefone: str | None = None
    cidade: str | None = None
    uf: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    linkedin: str | None = None
    resumo: str | None = None
    experiencias: list[Experiencia] = []
    formacao: list[Formacao] = []
    habilidades: list[str] = []
    idiomas: list[Idioma] = []


def to_target(cv: Curriculo) -> str:
    """Canonical serialisation used as the training target.

    Compact separators: every token of the target is paid for in VRAM at train time and
    in latency at inference. Key order follows the schema, so it is stable.
    """
    return json.dumps(cv.model_dump(), ensure_ascii=False, separators=(",", ":"))


def json_schema() -> dict:
    """For llama.cpp's grammar-constrained decoding (ADR 0007).

    Every key is made required, matching ``to_target`` (which always emits every key, in
    schema order). llama.cpp's grammar keeps optional keys in schema order but lets them
    be skipped, so a model that writes "habilidades" before "experiencias" silently loses
    every experience: the grammar then forbids the earlier key. Nullability and the empty
    list stay allowed, so the model can still say "absent".
    """
    schema = Curriculo.model_json_schema()
    for obj in [schema, *schema["$defs"].values()]:
        obj["required"] = list(obj["properties"])
        for prop in obj["properties"].values():
            prop.pop("default", None)
    return schema
