"""Post-processing of teacher prose (cvx.generate.teacher). No server needed."""

from cvx.generate.teacher import _strip_title

CARGOS = ["Gerente de loja", "Gerente de restaurante"]


def test_leading_title_is_stripped():
    # Found in the smoke data: ~1 in 5 descriptions led with a title, sometimes another job's.
    text = "Gerente de restaurante: responsável por conduzir negociações comerciais."
    assert _strip_title(text, CARGOS) == "Responsável por conduzir negociações comerciais."


def test_other_colons_are_kept():
    text = "Responsável por rotinas: estoque, caixa e atendimento."
    assert _strip_title(text, CARGOS) == text
    assert _strip_title(None, CARGOS) is None
