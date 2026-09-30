"""Per-field scoring of one extraction against its gold resume (ADR 0007).

Scalars score exact-match after normalisation. Lists score P/R/F1: experiences and
education are matched on their identity fields first, then their dates are scored on the
matched pairs only -- so one missing job does not also count as N wrong dates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cvx import normalize as n
from cvx.schema import Curriculo

SCALARS = {
    "nome": n.text,
    "email": n.email,
    "telefone": n.phone,
    "cidade": n.text,
    "uf": n.text,
    "linkedin": n.text,
}


@dataclass
class PRF:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: "PRF") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


@dataclass
class Score:
    scalars: dict[str, bool] = field(default_factory=dict)
    experiencias: PRF = field(default_factory=PRF)
    experiencia_datas: PRF = field(default_factory=PRF)
    formacao: PRF = field(default_factory=PRF)
    habilidades: PRF = field(default_factory=PRF)
    idiomas: PRF = field(default_factory=PRF)
    # Extracted values that do not occur anywhere in the source text.
    hallucinated: int = 0
    extracted_values: int = 0


def _set_prf(gold: set, pred: set) -> PRF:
    return PRF(tp=len(gold & pred), fp=len(pred - gold), fn=len(gold - pred))


def _exp_key(e) -> tuple[str, str]:
    return n.text(e.empresa), n.text(e.cargo)


def _edu_key(f) -> tuple[str, str]:
    return n.text(f.instituicao), n.text(f.curso)


def score(gold: Curriculo, pred: Curriculo, source_text: str | None = None) -> Score:
    s = Score()
    for name, norm in SCALARS.items():
        s.scalars[name] = norm(getattr(gold, name)) == norm(getattr(pred, name))

    gold_exp = {_exp_key(e): e for e in gold.experiencias}
    pred_exp = {_exp_key(e): e for e in pred.experiencias}
    s.experiencias = _set_prf(set(gold_exp), set(pred_exp))
    for key in set(gold_exp) & set(pred_exp):
        g, p = gold_exp[key], pred_exp[key]
        for attr in ("inicio", "fim"):
            ok = n.date(getattr(g, attr)) == n.date(getattr(p, attr))
            s.experiencia_datas.add(PRF(tp=1) if ok else PRF(fp=1, fn=1))

    s.formacao = _set_prf({_edu_key(f) for f in gold.formacao}, {_edu_key(f) for f in pred.formacao})
    s.habilidades = _set_prf({n.text(h) for h in gold.habilidades}, {n.text(h) for h in pred.habilidades})
    s.idiomas = _set_prf(
        {(n.text(i.idioma), i.nivel) for i in gold.idiomas},
        {(n.text(i.idioma), i.nivel) for i in pred.idiomas},
    )

    if source_text is not None:
        haystack = n.text(source_text)
        for value in _grounded_values(pred):
            s.extracted_values += 1
            if n.text(value) not in haystack:
                s.hallucinated += 1
    return s


def _grounded_values(cv: Curriculo) -> list[str]:
    """Values that must appear verbatim in the resume. Dates and free text are excluded:
    dates are reformatted by design, and descriptions may be paraphrased."""
    values = [v for v in (cv.nome, cv.email, cv.cidade) if v]
    for e in cv.experiencias:
        values += [e.empresa, e.cargo]
    for f in cv.formacao:
        values += [f.instituicao, f.curso]
    values += cv.habilidades
    return values
