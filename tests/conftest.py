import pytest

from cvx.schema import Curriculo


@pytest.fixture
def cv() -> Curriculo:
    return Curriculo.model_validate({
        "nome": "Joana Ribeiro Lopes",
        "email": "joana.lopes@example.com",
        "telefone": "(48) 99876-5432",
        "cidade": "Florianópolis",
        "uf": "SC",
        "linkedin": None,
        "resumo": "Técnica de enfermagem com seis anos de experiência hospitalar.",
        "experiencias": [
            {"empresa": "Hospital Vale Verde", "cargo": "Técnica de enfermagem",
             "inicio": "2021-03", "fim": None, "cidade": "Florianópolis",
             "descricao": "Atendimento em UTI adulto."},
            {"empresa": "Clínica Santa Luzia", "cargo": "Auxiliar de enfermagem",
             "inicio": "2018-02", "fim": "2021-01"},
        ],
        "formacao": [
            {"instituicao": "Escola Técnica Horizonte", "curso": "Técnico em Enfermagem",
             "nivel": "tecnico", "conclusao": "2017"},
        ],
        "habilidades": ["Administração de medicamentos", "Curativos", "Pacote Office"],
        "idiomas": [{"idioma": "Espanhol", "nivel": "intermediario"}],
    })
