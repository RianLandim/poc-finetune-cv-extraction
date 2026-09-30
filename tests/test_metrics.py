from cvx.metrics import score
from cvx.schema import Curriculo


def test_perfect_prediction(cv):
    s = score(cv, cv)
    assert all(s.scalars.values())
    assert s.experiencias.f1 == s.experiencia_datas.f1 == s.habilidades.f1 == 1.0


def test_formatting_differences_are_not_errors(cv):
    pred = cv.model_copy(update={"telefone": "48 998765432", "cidade": "florianopolis"})
    assert all(score(cv, pred).scalars.values())


def test_missing_job_does_not_also_count_as_wrong_dates(cv):
    pred = cv.model_copy(update={"experiencias": cv.experiencias[:1]})
    s = score(cv, pred)
    assert s.experiencias.recall == 0.5
    assert s.experiencia_datas.f1 == 1.0


def test_wrong_date_on_a_matched_job(cv):
    jobs = [cv.experiencias[0].model_copy(update={"inicio": "2020-03"}), cv.experiencias[1]]
    s = score(cv, cv.model_copy(update={"experiencias": jobs}))
    assert s.experiencias.f1 == 1.0
    assert s.experiencia_datas.tp == 3 and s.experiencia_datas.fn == 1


def test_hallucination_is_counted_against_source_text(cv):
    source = "Joana Ribeiro Lopes · joana.lopes@example.com · Florianópolis"
    pred = Curriculo(nome="Joana Ribeiro Lopes", habilidades=["Python"])
    s = score(cv, pred, source_text=source)
    assert s.extracted_values == 2 and s.hallucinated == 1
