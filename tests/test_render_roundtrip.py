"""Every gold value must be findable in the rendered PDF's text (phase 1).

If the renderer hides a field, the gold must say null; if the gold has a value, the PDF
must show it. This is what makes the synthetic labels exact.

Checked here: the fields the renderer copies verbatim. Dates, phone and city are
reformatted per style (cvx.generate.render.Fmt) and covered by the normaliser's tests.
Templates in render.INTERLEAVED put regions side by side, so extraction interleaves their
lines; there a value only has to appear word by word, in order. Every other template must
show it as one string.
"""

import pytest

from cvx.generate.extract_text import pdf_to_text
from cvx.generate.render import INTERLEAVED, available_templates, render_pdf, sample_style


def _verbatim(cv) -> list[str]:
    values = [cv.nome, cv.email, cv.linkedin, cv.resumo]
    for e in cv.experiencias:
        values += [e.cargo, e.empresa, e.cidade, e.descricao]
    for f in cv.formacao:
        values += [f.curso, f.instituicao]
    values += cv.habilidades + [i.idioma for i in cv.idiomas]
    return [v for v in values if v]


def _in_order(words: list[str], text_words: list[str]) -> bool:
    """Each word appears after the previous one; list punctuation ("Espanhol:") ignored."""
    strip = ":,.;•·"
    it = (t.strip(strip) for t in text_words)
    return all(any(w.strip(strip) == t for t in it) for w in words)


@pytest.mark.parametrize("template", available_templates())
@pytest.mark.parametrize("key", ["a", "b", "c"])
def test_gold_values_are_in_the_text(cv, template, key, tmp_path):
    long_cv = cv.model_copy(update={
        "email": "joana.ribeiro.de.souza.lopes@yahoo.com.br",
        "linkedin": "linkedin.com/in/joana-ribeiro-de-souza-lopes",
        "habilidades": cv.habilidades + ["Aplicação de normas de segurança NR-32",
                                         "Organização de eventos comunitários"],
    })
    style = sample_style(template, 3407, key)
    pdf = tmp_path / "cv.pdf"
    render_pdf(long_cv, style, pdf)
    text = pdf_to_text(pdf)
    flat = " ".join(text.split()).casefold()  # scoring casefolds too (cvx.normalize)
    for value in _verbatim(long_cv):
        if template in INTERLEAVED:
            assert _in_order(value.casefold().split(), flat.split()), value
        else:
            assert " ".join(value.casefold().split()) in flat, value


@pytest.mark.parametrize("template", available_templates())
def test_contact_items_never_break(cv, template, tmp_path):
    """An email or URL is one token in the text, whatever the layout."""
    long_cv = cv.model_copy(update={
        "email": "joana.ribeiro.de.souza.lopes@yahoo.com.br",
        "linkedin": "linkedin.com/in/joana-ribeiro-de-souza-lopes"})
    for key in "abc":
        pdf = tmp_path / f"{key}.pdf"
        render_pdf(long_cv, sample_style(template, 3407, key), pdf)
        words = pdf_to_text(pdf).split()
        assert long_cv.email in words and long_cv.linkedin in words
