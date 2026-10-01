"""Fill the free-text fields of a skeleton with an LLM (ADR 0003).

The teacher writes ONLY ``resumo`` and ``experiencias[].descricao``, and only where the
``ProsePlan`` asks for them. Its output is dropped in by position, so a bad generation
costs a paragraph of prose, never a wrong label. Anything suspicious -- the wrong number
of descriptions, a name, a date, a company that is not the job's own -- discards that
piece of prose (the field stays null in both the PDF and the gold).

Endpoint: llama-server's ``/v1/chat/completions`` with a JSON-schema response format. The
raw-``/completion`` rule (ADR 0011) protects train/inference parity of the model under
training; the teacher is a data source, not that model, so its prompt lives here and
not in ``cvx.prompt``.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import httpx

from cvx.generate.seed import ProsePlan
from cvx.schema import Curriculo

SYSTEM = (
    "Você escreve trechos de currículos brasileiros, em português do Brasil, com tom "
    "profissional e objetivo. Nunca invente nomes de pessoas, empresas, datas, números "
    "de telefone ou instituições; use apenas o que for fornecido."
)

_DATE_RE = re.compile(r"\b(19|20)\d{2}\b")


def _schema(n_desc: int, want_resumo: bool) -> dict:
    return {
        "type": "object",
        "properties": {
            "resumo": {"type": "string"} if want_resumo else {"type": "null"},
            "descricoes": {"type": "array", "items": {"type": "string"},
                           "minItems": n_desc, "maxItems": n_desc},
        },
        "required": ["resumo", "descricoes"],
        "additionalProperties": False,
    }


def _user_prompt(cv: Curriculo, persona: dict, plan: ProsePlan) -> str:
    jobs = [e for e, want in zip(cv.experiencias, plan.descricoes) if want]
    # The resume's own career goes first and wins: the persona narrative may describe a
    # different role ("coordenadora de equipe") than the titles the seed drew.
    cargos = ", ".join(dict.fromkeys(e.cargo for e in cv.experiencias)) or "nenhum (sem experiência)"
    lines = [
        "Perfil da pessoa (use só como inspiração, sem copiar nomes):",
        (persona.get("professional_persona") or "").strip(),
        "",
        "Cargos no currículo, do mais recente ao mais antigo: " + cargos,
        "Habilidades: " + ", ".join(cv.habilidades),
        "",
    ]
    if plan.resumo:
        lines.append("Escreva \"resumo\": 2 a 3 frases de resumo profissional em terceira "
                     "pessoa impessoal (ex.: \"Profissional com experiência em...\"), sem "
                     "nome, sem datas e sem citar empresas. O resumo deve ser coerente com os "
                     "cargos acima: não atribua à pessoa um cargo ou nível que não esteja na "
                     "lista, mesmo que o perfil o mencione.")
    else:
        lines.append("\"resumo\" deve ser null.")
    if jobs:
        lines.append(f"Escreva \"descricoes\": exatamente {len(jobs)} descrições, uma por "
                     "cargo abaixo e na mesma ordem, cada uma com 1 a 2 frases começando "
                     "por um verbo ou por \"Responsável por\", sem repetir o cargo, sem "
                     "datas e sem nomes:")
        lines += [f"{i + 1}. {e.cargo}" for i, e in enumerate(jobs)]
    else:
        lines.append("\"descricoes\" deve ser uma lista vazia.")
    return "\n".join(lines)


def _clean(text: str | None, forbidden: list[str]) -> str | None:
    """None when the prose leaks something that could contradict a scored field."""
    if not text:
        return None
    text = " ".join(text.split())
    if _DATE_RE.search(text) or "@" in text:
        return None
    lowered = text.casefold()
    if any(f and f.casefold() in lowered for f in forbidden):
        return None
    return text


def _strip_title(text: str | None, cargos: list[str]) -> str | None:
    """Drop a leading "Cargo:" the teacher sometimes prepends -- often the WRONG job's
    title, which would teach the model to read titles off the prose."""
    if not text or ":" not in text[:80]:
        return text
    head, rest = text.split(":", 1)
    if head.strip().casefold() in {c.casefold() for c in cargos} and rest.strip():
        rest = rest.strip()
        return rest[0].upper() + rest[1:]
    return text


def fill_prose(cv: Curriculo, persona: dict, plan: ProsePlan, cfg: SimpleNamespace,
               client: httpx.Client) -> tuple[Curriculo, dict]:
    """(cv with prose, stats). Never raises on a bad generation -- it drops the prose."""
    n_desc = sum(plan.descricoes)
    if not plan.resumo and n_desc == 0:
        return cv, {"called": False}
    body = {
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": _user_prompt(cv, persona, plan)}],
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "max_tokens": cfg.max_tokens,
        "response_format": {"type": "json_schema",
                            "json_schema": {"schema": _schema(n_desc, plan.resumo)}},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    stats: dict = {"called": True}
    try:
        r = client.post("/v1/chat/completions", json=body)
        r.raise_for_status()
        out = json.loads(r.json()["choices"][0]["message"]["content"])
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        stats["error"] = type(exc).__name__
        return cv, stats

    # Words that must not appear in prose: the persona's narrative name (its first word),
    # every person/employer/institution name on the resume.
    narrative_name = (persona.get("professional_persona") or "").split(",")[0].split()[:1]
    forbidden = [*narrative_name, *cv.nome.split(),
                 *(e.empresa for e in cv.experiencias),
                 *(f.instituicao for f in cv.formacao)]
    # Particles in names ("da", "de") are ordinary words; do not forbid them.
    forbidden = [f for f in forbidden if len(f) > 3]

    resumo = _clean(out.get("resumo"), forbidden) if plan.resumo else None
    cargos = [e.cargo for e in cv.experiencias]
    descs = out.get("descricoes") or []
    it = iter(descs if len(descs) == n_desc else [None] * n_desc)
    jobs = []
    for exp, want in zip(cv.experiencias, plan.descricoes):
        desc = _strip_title(_clean(next(it), forbidden), cargos) if want else None
        jobs.append(exp.model_copy(update={"descricao": desc}))
    stats.update(resumo_kept=resumo is not None if plan.resumo else None,
                 desc_asked=n_desc, desc_kept=sum(j.descricao is not None for j in jobs))
    return cv.model_copy(update={"resumo": resumo, "experiencias": jobs}), stats
