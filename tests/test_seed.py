"""Invariants of the persona -> gold skeleton step (cvx.generate.seed, ADR 0003/0005)."""

import itertools

import pytest

from cvx.config import load_data_config
from cvx.generate import catalog as cat
from cvx.generate.seed import NOW, skeleton

P_SHOW = load_data_config().generate.p_show

RETAIL = {
    "occupation": "Trabalhador dos serviços, vendedor do comércio ou mercado",
    "professional_persona": "Luci, coordenadora de equipe em comércio de varejo, com "
                            "experiência em atendimento, vendas e operação de caixa.",
    "skills_and_expertise_list": "['atendimento ao cliente', 'vendas', 'operador de caixa', "
                                 "'Organização de finanças pessoais', 'Jardinagem doméstica']",
}
SCIENCE = {
    "occupation": "Profissional das ciências ou intelectual",
    "professional_persona": "Caroline, pesquisadora em biologia molecular na universidade.",
    "skills_and_expertise_list": "['Redação de artigos científicos', 'Gestão de projetos de "
                                 "pesquisa']",
}


def persona(uuid: str, **over) -> dict:
    base = {"uuid": uuid, "age": 40, "sex": "Feminino", "state": "São Paulo",
            "municipality": "Campinas", "education_level": "Superior completo", **RETAIL}
    return {**base, **over}


def _months(ym: str) -> int:
    y, m = ym.split("-")
    return int(y) * 12 + int(m) - 1


GRID = [persona(f"u{i}", age=age, education_level=edu, occupation=occ, sex=sex)
        for i, (age, edu, occ, sex) in enumerate(itertools.product(
            [18, 24, 37, 64], list(cat.EDUCATION_TIER), list(cat.JOB_TITLES),
            ["Feminino", "Masculino"]))]


def test_deterministic():
    p = persona("abc")
    assert skeleton(p, 1, P_SHOW).cv == skeleton(p, 1, P_SHOW).cv
    assert skeleton(p, 1, P_SHOW).cv != skeleton(p, 2, P_SHOW).cv


@pytest.mark.parametrize("p", GRID[::7], ids=lambda p: p["uuid"])
def test_timeline_is_plausible(p):
    cv = skeleton(p, 3407, P_SHOW).cv
    birth = (NOW[0] - p["age"]) * 12
    now = NOW[0] * 12 + NOW[1] - 1
    prev_start = None
    for e in cv.experiencias:  # most recent first
        start = _months(e.inicio)
        end = now if e.fim is None else _months(e.fim)
        assert start - birth >= 15 * 12
        assert start < end <= now
        if prev_start is not None:
            assert end < prev_start  # no overlap
        prev_start = start
    for f in cv.formacao:
        if f.conclusao:
            assert int(f.conclusao[:4]) < NOW[0] + 1


def test_no_protected_fields():
    fields = set(skeleton(persona("x"), 1, P_SHOW).cv.model_dump())
    assert not fields & {"cpf", "rg", "data_nascimento", "nascimento", "idade"}


def test_titles_follow_the_persona():
    """A retail persona mostly gets counter/shop titles; uniform picking gave ~1/3."""
    titles = [e.cargo for i in range(200)
              for e in skeleton(persona(f"r{i}", education_level="Fundamental completo e "
                                        "médio incompleto"), 1, P_SHOW).cv.experiencias]
    on_track = sum(t.startswith(("Operador de caixa", "Atendente")) for t in titles)
    assert on_track / len(titles) > 0.6


def test_science_persona_reaches_research_titles():
    titles = [e.cargo for i in range(200)
              for e in skeleton(persona(f"s{i}", **SCIENCE), 1, P_SHOW).cv.experiencias]
    research = ("Pesquisador", "Biólogo", "Auxiliar de pesquisa", "Professor universitário")
    assert sum(t in research for t in titles) / len(titles) > 0.5


def test_private_skills_dropped():
    for i in range(30):
        skills = skeleton(persona(f"k{i}"), 1, P_SHOW).cv.habilidades
        assert not any("pessoais" in s or "doméstica" in s for s in skills)
