"""Gold resume -> HTML (Jinja) -> PDF (WeasyPrint).

The gold JSON is canonical ("2021-03", "(48) 99876-5432"); the PDF is not. Per resume
the renderer samples one surface ``Style`` -- date, phone and city formats, section
headings and order, font, skills layout, distractor lines -- so the model learns to read
and normalise, not to copy one layout.

Contract with the gold (tests/test_render_roundtrip.py): every non-null gold value is
visible in the PDF, and nothing the schema covers is shown without being in the gold.
Distractors (driving licence, availability) are outside the schema on purpose: the model
must learn to leave them out.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from cvx.generate.seed import rng_for
from cvx.schema import Curriculo

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"

MONTHS = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto",
          "setembro", "outubro", "novembro", "dezembro"]
DATE_STYLES = ["mm/yyyy", "mon/yyyy", "Mon yyyy", "month de yyyy", "mm.yyyy"]
CURRENT_WORDS = ["atual", "Atual", "o momento", "presente", "Presente", "até o momento"]
RANGE_SEPS = [" – ", " - ", " a ", " até "]
PHONE_STYLES = ["(dd) nnnnn-nnnn", "dd nnnnn-nnnn", "+55 (dd) nnnnn-nnnn", "(dd) nnnnnnnnn",
                "+55 dd nnnnnnnnn"]
CITY_STYLES = ["{c}, {uf}", "{c} - {uf}", "{c}/{uf}", "{c} ({uf})"]
SKILL_STYLES = ["comma", "bullets", "dots"]
HEADINGS = {
    "resumo": ["Resumo", "Resumo profissional", "Perfil", "Sobre mim", "Perfil profissional"],
    "experiencias": ["Experiência profissional", "Experiência", "Histórico profissional",
                     "Experiências"],
    "formacao": ["Formação", "Formação acadêmica", "Escolaridade", "Educação"],
    "habilidades": ["Habilidades", "Competências", "Conhecimentos", "Habilidades técnicas"],
    "idiomas": ["Idiomas", "Línguas"],
    "extras": ["Informações adicionais", "Outras informações", "Adicionais"],
}
LEVEL_LABEL = {"basico": "Básico", "intermediario": "Intermediário", "avancado": "Avançado",
               "fluente": "Fluente", "nativo": "Nativo"}
NIVEL_LABEL = {"fundamental": "Ensino Fundamental", "medio": "Ensino Médio",
               "tecnico": "Curso Técnico", "graduacao": "Graduação", "pos": "Pós-graduação",
               "mestrado": "Mestrado", "doutorado": "Doutorado"}
FONTS = ["DejaVu Sans", "Liberation Sans", "Liberation Serif", "Noto Sans", "Noto Serif",
         "Carlito", "Caladea", "Roboto"]
ACCENTS = ["#1f3a5f", "#2e5e4e", "#6b2d2d", "#333333", "#4a3b6b", "#0b5c8a"]
DISTRACTORS = ["CNH categoria B", "CNH categorias A e B", "Disponibilidade para viagens",
               "Disponibilidade de horário", "Disponibilidade para início imediato",
               "Veículo próprio", "Referências disponíveis mediante solicitação",
               "Pacote Office (certificado)", "Trabalho voluntário em ONG local"]


@dataclass
class Style:
    template: str
    font: str
    font_size: float
    accent: str
    date_style: str
    current_word: str
    range_sep: str
    phone_style: str
    city_style: str
    skill_style: str
    show_nivel_label: bool
    headings: dict[str, str]
    section_order: list[str]
    extras: list[str]


# Layouts whose regions sit side by side, so pdfplumber interleaves them line by line
# (cvx.generate.extract_text). There a gold value is only guaranteed to appear word by
# word, in order -- not as one contiguous string (tests/test_render_roundtrip.py).
INTERLEAVED = {
    "lateral", "lateral_direita", "duas_colunas", "colunas_fluidas", "perfil_lateral",
    "cartoes", "tabela_grade", "linha_do_tempo", "secoes_margem", "etiquetas",
    "cabecalho_faixa", "caixa_resumo",
}


def available_templates() -> list[str]:
    """Template ids; files starting with "_" are shared partials, not layouts."""
    return sorted(p.name.removesuffix(".html.j2") for p in TEMPLATES_DIR.glob("*.html.j2")
                  if not p.name.startswith("_"))


def sample_style(template: str, seed: int, key: str) -> Style:
    rng = rng_for(seed, f"style:{key}")
    body = ["experiencias", "formacao"] if rng.random() < 0.75 else ["formacao", "experiencias"]
    tail = ["habilidades", "idiomas"]
    rng.shuffle(tail)
    return Style(
        template=template,
        font=rng.choice(FONTS),
        font_size=rng.choice([9.5, 10.0, 10.5, 11.0]),
        accent=rng.choice(ACCENTS),
        date_style=rng.choice(DATE_STYLES),
        current_word=rng.choice(CURRENT_WORDS),
        range_sep=rng.choice(RANGE_SEPS),
        phone_style=rng.choice(PHONE_STYLES),
        city_style=rng.choice(CITY_STYLES),
        skill_style=rng.choice(SKILL_STYLES),
        show_nivel_label=rng.random() < 0.6,
        headings={k: rng.choice(v) for k, v in HEADINGS.items()},
        section_order=["resumo", *body, *tail, "extras"],
        extras=rng.sample(DISTRACTORS, rng.randint(1, 3)) if rng.random() < 0.35 else [],
    )


class Fmt:
    """Display formatting for one resume. Templates call these; they never format."""

    def __init__(self, style: Style) -> None:
        self.s = style

    def date(self, value: str | None) -> str:
        if value is None:
            return self.s.current_word
        if len(value) == 4:
            return value
        year, month = value.split("-")
        m = int(month)
        return {
            "mm/yyyy": f"{month}/{year}",
            "mon/yyyy": f"{MONTHS[m - 1][:3]}/{year}",
            "Mon yyyy": f"{MONTHS[m - 1][:3].capitalize()} {year}",
            "month de yyyy": f"{MONTHS[m - 1]} de {year}",
            "mm.yyyy": f"{month}.{year}",
        }[self.s.date_style]

    def period(self, start: str, end: str | None) -> str:
        return f"{self.date(start)}{self.s.range_sep}{self.date(end)}"

    def completion(self, value: str | None) -> str:
        return self.date(value) if value else "em andamento"

    def phone(self, value: str | None) -> str | None:
        if value is None:
            return None
        ddd, num = value[1:3], value[5:].replace("-", "")
        return {
            "(dd) nnnnn-nnnn": value,
            "dd nnnnn-nnnn": f"{ddd} {num[:5]}-{num[5:]}",
            "+55 (dd) nnnnn-nnnn": f"+55 {value}",
            "(dd) nnnnnnnnn": f"({ddd}) {num}",
            "+55 dd nnnnnnnnn": f"+55 {ddd} {num}",
        }[self.s.phone_style]

    def city(self, city: str | None, uf: str | None) -> str | None:
        if not city:
            return None
        return self.s.city_style.format(c=city, uf=uf) if uf else city

    def language_level(self, nivel: str) -> str:
        return LEVEL_LABEL[nivel]

    def nivel(self, nivel: str) -> str:
        return NIVEL_LABEL[nivel]


_env = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    autoescape=select_autoescape(["j2", "html"]),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)
# Contact items too wide for the lateral template's sidebar at 8.5pt in the widest font.
_env.tests["long_contact"] = lambda s: len(s) > 26


def render_html(cv: Curriculo, style: Style) -> str:
    template = _env.get_template(f"{style.template}.html.j2")
    return template.render(cv=cv, fmt=Fmt(style), style=style)


def render_pdf(cv: Curriculo, style: Style, out_pdf: Path) -> int:
    """Write the PDF; return its page count."""
    from weasyprint import HTML

    document = HTML(string=render_html(cv, style), base_url=str(TEMPLATES_DIR)).render()
    document.write_pdf(out_pdf)
    return len(document.pages)


def style_dict(style: Style) -> dict:
    return asdict(style)


def pick_template(templates: list[str], seed: int, key: str) -> str:
    return rng_for(seed, f"tpl:{key}").choice(templates)
