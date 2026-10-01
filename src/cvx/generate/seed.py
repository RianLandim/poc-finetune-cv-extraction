"""Persona -> resume skeleton, with NO LLM involved (ADR 0003).

Everything the model is scored on exactly -- name, contact, city/UF, employers, titles,
dates, institutions, skills -- is produced here, deterministically, from the persona row,
Faker ``pt_BR``, the curated ``catalog`` and an RNG seeded by (seed, persona uuid).

Invariants (tests/test_seed.py):
- the career timeline fits the persona's age and education: no job before 15, jobs do not
  overlap, nothing ends after ``NOW``, education completes at plausible ages;
- employers and institutions are composed from name parts -- never real ones;
- no CPF, RG or date of birth is generated (ADR 0005);
- a field the renderer will not show is None in the skeleton already, so the skeleton IS
  the gold label. The only fields left for the teacher are listed in ``ProsePlan``.
"""

from __future__ import annotations

import ast
import hashlib
import random
import re
import unicodedata
from dataclasses import dataclass, field
from types import SimpleNamespace

from faker import Faker

from cvx.generate import catalog as cat
from cvx.schema import Curriculo, Experiencia, Formacao, Idioma

# The generator's "today". Fixed, so a dataset regenerates identically on another day.
NOW = (2026, 9)

EMAIL_DOMAINS = ["gmail.com", "hotmail.com", "outlook.com", "yahoo.com.br", "uol.com.br",
                 "bol.com.br", "icloud.com"]

LANGUAGE_LEVELS = {
    0: ["basico"],
    1: ["basico", "basico", "intermediario"],
    2: ["basico", "intermediario", "intermediario", "avancado"],
    3: ["intermediario", "avancado", "avancado", "fluente"],
}


@dataclass
class ProsePlan:
    """Which free-text fields the teacher should write (ADR 0003)."""

    resumo: bool = False
    descricoes: list[bool] = field(default_factory=list)  # one per experiencias entry


@dataclass
class Seeded:
    cv: Curriculo
    plan: ProsePlan
    persona_uuid: str


def rng_for(seed: int, key: str) -> random.Random:
    """Stable RNG per (seed, key) -- independent of PYTHONHASHSEED and call order."""
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _slug(text: str) -> str:
    ascii_ = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", ".", ascii_.lower()).strip(".")


def _months(y: int, m: int) -> int:
    return y * 12 + (m - 1)


def _ym(total: int) -> str:
    return f"{total // 12:04d}-{total % 12 + 1:02d}"


def _pick_tier(table: dict[int, list[str]], tier: int) -> list[str]:
    for t in range(tier, -1, -1):
        if t in table:
            return table[t]
    return table[min(table)]


# Persona skills that describe private life, not work ("Organização de finanças pessoais",
# "Jardinagem doméstica"); a resume would not list them.
# Word-bounded on purpose: "Departamento pessoal", "Relações interpessoais", "Conserto de
# eletrodomésticos" and "Agricultura familiar" are work skills.
_PRIVATE_SKILL = re.compile(r"\bpessoais\b|\bdom[eé]stic|\bhobb|\blazer\b", re.IGNORECASE)

# Words too generic to say two job titles or a title and a persona are related.
_STOP = {"auxiliar", "ajudante", "assistente", "analista", "tecnico", "geral", "gerais",
         "servicos", "trabalhador", "profissional", "agente"}


def _persona_skills(persona: dict) -> list[str]:
    raw = persona.get("skills_and_expertise_list") or "[]"
    try:
        return [s.strip() for s in ast.literal_eval(raw) if isinstance(s, str) and s.strip()]
    except (ValueError, SyntaxError):
        return []


def _stems(text: str) -> set[str]:
    """Crude Portuguese stems: 'vendas'/'vendedor' -> 'vend', 'operação'/'operador' -> 'oper'."""
    words = re.findall(r"[a-z]{4,}", _slug(text).replace(".", " "))
    return {w[:4] for w in words if w not in _STOP}


def _skills(persona: dict, rng: random.Random) -> list[str]:
    # Dataset skills are sometimes full clauses; resumes list short items.
    skills = list(dict.fromkeys(s for s in _persona_skills(persona)
                                if len(s) <= 60 and not _PRIVATE_SKILL.search(s)))
    rng.shuffle(skills)
    return skills[: rng.randint(3, 8)]


def _name(fake: Faker, sex: str, rng: random.Random) -> str:
    first = fake.first_name_female() if sex == "Feminino" else fake.first_name_male()
    surnames = [fake.last_name() for _ in range(rng.choice([1, 2, 2, 3]))]
    # Faker occasionally repeats a surname; a real resume would not.
    return " ".join([first, *dict.fromkeys(surnames)])


def _education(tier: int, occupation: str, birth_year: int, age: int,
               rng: random.Random) -> list[Formacao]:
    out: list[Formacao] = []

    def inst(nivel: str) -> str:
        return f"{rng.choice(cat.INSTITUTION_PREFIX[nivel])} {rng.choice(cat.INSTITUTION_NAME)}"

    def done(at_age: int) -> str | None:
        year = birth_year + at_age
        return str(year) if year < NOW[0] else None

    if tier == 1:
        if rng.random() < 0.6:
            out.append(Formacao(instituicao=inst("fundamental"), curso="Ensino Fundamental",
                                nivel="fundamental", conclusao=done(rng.randint(14, 16))))
        if age <= 20 and rng.random() < 0.5:
            out.insert(0, Formacao(instituicao=inst("medio"), curso="Ensino Médio",
                                   nivel="medio", conclusao=None))
    elif tier == 2:
        medio_age = rng.randint(17, 19)
        out.append(Formacao(instituicao=inst("medio"), curso="Ensino Médio",
                            nivel="medio", conclusao=done(medio_age)))
        if age <= 30 and rng.random() < 0.35:
            course = rng.choice(cat.GRADUACAO.get(occupation, cat.GRADUACAO_DEFAULT))
            out.insert(0, Formacao(instituicao=inst("graduacao"), curso=course,
                                   nivel="graduacao", conclusao=None))
        elif rng.random() < 0.25:
            out.insert(0, Formacao(instituicao=inst("tecnico"), curso=rng.choice(cat.TECNICO),
                                   nivel="tecnico", conclusao=done(medio_age + rng.randint(1, 3))))
    elif tier == 3:
        grad_age = rng.randint(21, 26)
        course = rng.choice(cat.GRADUACAO.get(occupation, cat.GRADUACAO_DEFAULT))
        if rng.random() < 0.4:
            out.append(Formacao(instituicao=inst("medio"), curso="Ensino Médio",
                                nivel="medio", conclusao=done(rng.randint(17, 18))))
        out.insert(0, Formacao(instituicao=inst("graduacao"), curso=course,
                               nivel="graduacao", conclusao=done(grad_age)))
        last = grad_age
        if age > last + 2 and rng.random() < 0.25:
            last += rng.randint(1, 5)
            out.insert(0, Formacao(instituicao=inst("pos"), curso=rng.choice(cat.POS),
                                   nivel="pos", conclusao=done(last)))
        if (occupation == "Profissional das ciências ou intelectual"
                and age > last + 3 and rng.random() < 0.15):
            last += rng.randint(2, 4)
            out.insert(0, Formacao(instituicao=inst("mestrado"), curso=rng.choice(cat.MESTRADO),
                                   nivel="mestrado", conclusao=done(last)))
            if age > last + 5 and rng.random() < 0.3:
                last += rng.randint(4, 5)
                out.insert(0, Formacao(instituicao=inst("doutorado"),
                                       curso=rng.choice(cat.DOUTORADO),
                                       nivel="doutorado", conclusao=done(last)))
    # A resume states either a year or a month; occasionally a month.
    for i, f in enumerate(out):
        if f.conclusao and rng.random() < 0.2:
            month = rng.choice([6, 7, 11, 12])
            out[i] = f.model_copy(update={"conclusao": f"{f.conclusao}-{month:02d}"})
    return out


def _company(occupation: str, fake: Faker, rng: random.Random) -> str:
    if occupation.startswith("Membro das forças armadas") and rng.random() < 0.6:
        return rng.choice(cat.PUBLIC_EMPLOYERS).format(n=rng.randint(1, 40))
    segment = rng.choice(cat.COMPANY_SEGMENTS[occupation])
    # Faker pt_BR surnames carry particles ("da Rosa"); an employer name drops them.
    name = fake.last_name().split()[-1]
    if rng.random() < 0.3:
        name = f"{name} & {fake.last_name().split()[-1]}"
    elif rng.random() < 0.3:
        name = rng.choice(cat.INSTITUTION_NAME)
    suffix = rng.choice(cat.COMPANY_SUFFIXES)
    return " ".join(p for p in (segment, name, suffix) if p)


def _pick_title(pool: list[str], persona_stems: set[str], held: list[str],
                rng: random.Random) -> str:
    """A title from the pool, weighted towards the persona's own trade and the career so far.

    The occupation group is coarse (a biologist and a lawyer share one), so a uniform pick
    gives "Advogado -> Nutricionista" careers and summaries that contradict them. Titles that
    share stems with the persona text weigh more, a title already held more still; every
    title keeps a little weight, so the catalogue is still covered.
    """
    held_stems = set().union(*(_stems(t) for t in held)) if held else set()
    weights = []
    for title in pool:
        stems = _stems(title)
        w = 1.0 + 6.0 * len(stems & persona_stems) + 4.0 * len(stems & held_stems)
        weights.append(w * (3.0 if title in held else 1.0))
    return rng.choices(pool, weights=weights)[0]


def _career(tier: int, occupation: str, birth_year: int, city: str, persona_stems: set[str],
            fake: Faker, rng: random.Random, p_show: SimpleNamespace) -> list[Experiencia]:
    start_age = {0: (15, 18), 1: (16, 19), 2: (18, 21), 3: (21, 24)}[tier]
    earliest = _months(birth_year + rng.randint(*start_age), rng.randint(1, 12))
    now = _months(*NOW)
    if earliest >= now - 3:
        return []
    years = (now - earliest) // 12
    n_jobs = rng.randint(1, max(1, min(5, 1 + years // 3)))
    titles = cat.JOB_TITLES[occupation]

    jobs: list[Experiencia] = []
    job_tier = tier
    current = rng.random() < 0.7
    cursor = now if current else now - rng.randint(1, 12)
    for i in range(n_jobs):
        start = max(earliest, cursor - rng.randint(6, 72))
        if cursor - start < 3:
            break
        # Walking back in time, the tier may drop but never rises again: a career ladder.
        if i > 0 and rng.random() < 0.4:
            job_tier = max(0, job_tier - 1)
        exp_city = city if rng.random() < 0.8 else fake.city()
        jobs.append(Experiencia(
            empresa=_company(occupation, fake, rng),
            cargo=_pick_title(_pick_tier(titles, job_tier), persona_stems,
                              [j.cargo for j in jobs], rng),
            inicio=_ym(start),
            fim=None if (i == 0 and current) else _ym(cursor),
            cidade=exp_city if rng.random() < p_show.exp_cidade else None,
        ))
        cursor = start - rng.randint(1, 8)
        if cursor <= earliest:
            break
    return jobs


def _languages(tier: int, rng: random.Random) -> list[Idioma]:
    pool = cat.LANGUAGES[:4] if tier >= 2 else cat.LANGUAGES[:2]
    n = rng.choice([1, 1, 2]) if tier >= 2 else 1
    return [Idioma(idioma=lang, nivel=rng.choice(LANGUAGE_LEVELS[tier]))
            for lang in rng.sample(pool, n)]


def skeleton(persona: dict, seed: int, p_show: SimpleNamespace) -> Seeded:
    """Gold resume with every exact field decided; prose left for the teacher."""
    uuid = persona["uuid"]
    rng = rng_for(seed, uuid)
    fake = Faker("pt_BR")
    fake.seed_instance(rng.getrandbits(32))

    tier = cat.EDUCATION_TIER[persona["education_level"]]
    age = int(persona["age"])
    birth_year = NOW[0] - age
    uf = cat.UF.get(persona["state"])
    city = persona["municipality"]

    name = _name(fake, persona["sex"], rng)
    slug = _slug(name)
    parts = slug.split(".")
    local = rng.choice([slug, f"{parts[0]}.{parts[-1]}", f"{parts[0]}{parts[-1]}",
                        f"{parts[0]}.{parts[-1]}{rng.randint(1, 99)}"])
    ddd = rng.choice(cat.DDD[uf]) if uf else 11
    phone = f"({ddd}) 9{rng.randint(1000, 9999)}-{rng.randint(0, 9999):04d}"

    persona_stems = _stems(" ".join([persona.get("professional_persona") or "",
                                     *_persona_skills(persona)]))
    experiencias = _career(tier, persona["occupation"], birth_year, city, persona_stems,
                           fake, rng, p_show)
    cv = Curriculo(
        nome=name,
        email=f"{local}@{rng.choice(EMAIL_DOMAINS)}" if rng.random() < p_show.email else None,
        telefone=phone if rng.random() < p_show.telefone else None,
        cidade=city,
        uf=uf,
        linkedin=(f"linkedin.com/in/{slug.replace('.', '-')}"
                  if rng.random() < p_show.linkedin else None),
        resumo=None,
        experiencias=experiencias,
        formacao=_education(tier, persona["occupation"], birth_year, age, rng),
        habilidades=_skills(persona, rng),
        idiomas=_languages(tier, rng) if rng.random() < p_show.idiomas else [],
    )
    plan = ProsePlan(
        resumo=rng.random() < p_show.resumo,
        descricoes=[rng.random() < p_show.descricao for _ in experiencias],
    )
    return Seeded(cv=cv, plan=plan, persona_uuid=uuid)
