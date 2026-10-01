"""Curated, deterministic vocabularies for resume generation (ADR 0003).

Nemotron-Personas-Brazil gives only 11 coarse occupation groups (ISCO major groups, in
Portuguese). Concrete job titles, courses and institutions come from here, never from an
LLM: they are scored fields. Institutions are generated from name parts, so they read as
plausible without naming real ones.
"""

from __future__ import annotations

UF = {
    "Acre": "AC", "Alagoas": "AL", "Amapá": "AP", "Amazonas": "AM", "Bahia": "BA",
    "Ceará": "CE", "Distrito Federal": "DF", "Espírito Santo": "ES", "Goiás": "GO",
    "Maranhão": "MA", "Mato Grosso": "MT", "Mato Grosso do Sul": "MS", "Minas Gerais": "MG",
    "Pará": "PA", "Paraíba": "PB", "Paraná": "PR", "Pernambuco": "PE", "Piauí": "PI",
    "Rio de Janeiro": "RJ", "Rio Grande do Norte": "RN", "Rio Grande do Sul": "RS",
    "Rondônia": "RO", "Roraima": "RR", "Santa Catarina": "SC", "São Paulo": "SP",
    "Sergipe": "SE", "Tocantins": "TO",
}

DDD = {
    "AC": [68], "AL": [82], "AP": [96], "AM": [92, 97], "BA": [71, 73, 74, 75, 77],
    "CE": [85, 88], "DF": [61], "ES": [27, 28], "GO": [62, 64], "MA": [98, 99],
    "MT": [65, 66], "MS": [67], "MG": [31, 32, 33, 34, 35, 37, 38], "PA": [91, 93, 94],
    "PB": [83], "PR": [41, 42, 43, 44, 45, 46], "PE": [81, 87], "PI": [86, 89],
    "RJ": [21, 22, 24], "RN": [84], "RS": [51, 53, 54, 55], "RO": [69], "RR": [95],
    "SC": [47, 48, 49], "SP": [11, 12, 13, 14, 15, 16, 17, 18, 19], "SE": [79], "TO": [63],
}

# education_level values in the dataset -> coarse tier used below.
EDUCATION_TIER = {
    "Sem instrução e fundamental incompleto": 0,
    "Fundamental completo e médio incompleto": 1,
    "Médio completo e superior incompleto": 2,
    "Superior completo": 3,
}

# occupation group -> {tier: [job titles]}. A tier falls back to the nearest lower one.
# Titles are gender-neutral forms where Portuguese allows; the renderer does not inflect.
JOB_TITLES: dict[str, dict[int, list[str]]] = {
    "Trabalhador dos serviços, vendedor do comércio ou mercado": {
        0: ["Auxiliar de limpeza", "Repositor de mercadorias", "Ajudante de cozinha",
            "Atendente de lanchonete", "Frentista"],
        1: ["Operador de caixa", "Atendente de loja", "Cozinheiro", "Garçom",
            "Cuidador de idosos", "Vigilante"],
        2: ["Vendedor", "Recepcionista", "Consultor de vendas", "Supervisor de loja",
            "Atendente de telemarketing", "Cabeleireiro"],
        3: ["Supervisor de vendas", "Gerente de loja", "Consultor comercial",
            "Coordenador de vendas", "Supervisor de atendimento"],
    },
    "Ocupação elementar": {
        0: ["Auxiliar de serviços gerais", "Servente de obras", "Carregador",
            "Ajudante geral", "Trabalhador rural", "Empregado doméstico"],
        1: ["Auxiliar de produção", "Auxiliar de carga e descarga", "Zelador",
            "Coletor de resíduos"],
        2: ["Auxiliar de logística", "Conferente", "Porteiro"],
    },
    "Profissional das ciências ou intelectual": {
        2: ["Estagiário de engenharia", "Auxiliar de pesquisa"],
        3: ["Engenheiro civil", "Analista de sistemas", "Enfermeiro", "Professor",
            "Advogado", "Contador", "Psicólogo", "Farmacêutico", "Engenheiro de software",
            "Nutricionista", "Arquiteto", "Economista", "Fisioterapeuta", "Pesquisador",
            "Biólogo", "Professor universitário", "Químico", "Jornalista"],
    },
    "Trabalhador qualificado, operário ou artesão da construção, das artes mecânicas ou de outro ofício": {
        0: ["Pedreiro", "Pintor de obras", "Ajudante de eletricista"],
        1: ["Eletricista", "Mecânico de automóveis", "Soldador", "Carpinteiro",
            "Encanador", "Marceneiro", "Costureiro"],
        2: ["Eletricista de manutenção", "Mecânico industrial", "Técnico em refrigeração",
            "Mestre de obras"],
    },
    "Trabalhador de apoio administrativo": {
        1: ["Auxiliar de escritório", "Office boy", "Arquivista"],
        2: ["Assistente administrativo", "Auxiliar administrativo", "Auxiliar financeiro",
            "Assistente de recursos humanos", "Faturista", "Almoxarife", "Secretário"],
        3: ["Analista administrativo", "Analista financeiro", "Analista de departamento pessoal"],
    },
    "Técnico ou profissional de nível médio": {
        1: ["Auxiliar técnico"],
        2: ["Técnico de enfermagem", "Técnico em segurança do trabalho", "Técnico em informática",
            "Técnico em eletrotécnica", "Técnico em contabilidade", "Corretor de imóveis",
            "Técnico em laboratório"],
        3: ["Técnico em edificações", "Desenvolvedor web", "Analista de suporte"],
    },
    "Operador de instalação ou máquina ou montador": {
        0: ["Ajudante de produção", "Operador de empilhadeira"],
        1: ["Operador de máquinas", "Montador", "Motorista de caminhão", "Motorista de ônibus",
            "Operador de produção"],
        2: ["Operador de máquinas CNC", "Operador de processos", "Motorista entregador"],
    },
    "Diretor ou gerente": {
        2: ["Gerente de loja", "Gerente de restaurante", "Coordenador administrativo"],
        3: ["Gerente de projetos", "Gerente comercial", "Coordenador de operações",
            "Diretor administrativo", "Gerente financeiro", "Gerente de recursos humanos"],
    },
    "Trabalhador qualificado da agropecuária, florestal, da caça ou da pesca": {
        0: ["Trabalhador rural", "Pescador", "Tratorista"],
        1: ["Operador de máquinas agrícolas", "Caseiro", "Criador de gado"],
        2: ["Técnico agrícola", "Viveirista"],
        3: ["Engenheiro agrônomo", "Zootecnista"],
    },
    "Ocupação mal definida": {
        0: ["Autônomo", "Ajudante geral"],
        1: ["Autônomo", "Auxiliar de serviços gerais"],
        2: ["Autônomo", "Assistente administrativo", "Vendedor"],
        3: ["Consultor autônomo", "Analista de projetos"],
    },
    "Membro das forças armadas, policial ou bombeiro militar": {
        1: ["Soldado", "Bombeiro civil"],
        2: ["Soldado", "Cabo", "Bombeiro militar", "Policial militar"],
        3: ["Sargento", "Tenente", "Policial militar"],
    },
}

# Undergraduate courses per occupation group (tier 3); a generic list otherwise.
GRADUACAO = {
    "Profissional das ciências ou intelectual": [
        "Engenharia Civil", "Sistemas de Informação", "Enfermagem", "Pedagogia", "Direito",
        "Ciências Contábeis", "Psicologia", "Farmácia", "Ciência da Computação", "Nutrição",
        "Arquitetura e Urbanismo", "Economia", "Fisioterapia", "Letras", "História"],
    "Diretor ou gerente": ["Administração", "Gestão Comercial", "Ciências Contábeis",
                           "Gestão de Recursos Humanos", "Engenharia de Produção"],
    "Trabalhador de apoio administrativo": ["Administração", "Ciências Contábeis",
                                            "Gestão Financeira", "Gestão de Recursos Humanos"],
    "Técnico ou profissional de nível médio": ["Análise e Desenvolvimento de Sistemas",
                                               "Engenharia Elétrica", "Enfermagem", "Logística"],
    "Trabalhador qualificado da agropecuária, florestal, da caça ou da pesca": [
        "Agronomia", "Zootecnia", "Medicina Veterinária"],
}
GRADUACAO_DEFAULT = ["Administração", "Logística", "Gestão Comercial", "Pedagogia",
                     "Marketing", "Recursos Humanos"]

TECNICO = ["Técnico em Enfermagem", "Técnico em Administração", "Técnico em Informática",
           "Técnico em Segurança do Trabalho", "Técnico em Eletrotécnica", "Técnico em Mecânica",
           "Técnico em Logística", "Técnico em Contabilidade", "Técnico em Agropecuária",
           "Técnico em Edificações"]

POS = ["MBA em Gestão de Projetos", "Especialização em Gestão Empresarial",
       "Especialização em Docência do Ensino Superior", "MBA em Finanças",
       "Especialização em Saúde Pública", "Especialização em Engenharia de Software"]

MESTRADO = ["Mestrado em Educação", "Mestrado em Engenharia Civil", "Mestrado em Administração",
            "Mestrado em Ciência da Computação", "Mestrado em Saúde Coletiva"]

DOUTORADO = ["Doutorado em Educação", "Doutorado em Ciências", "Doutorado em Administração"]

# Institution names are composed, not listed, so none names a real place of study.
INSTITUTION_PREFIX = {
    "fundamental": ["Escola Municipal", "Escola Estadual", "Colégio Municipal"],
    "medio": ["Escola Estadual", "Colégio Estadual", "Colégio", "Escola"],
    "tecnico": ["Escola Técnica", "Centro de Educação Profissional", "Instituto Técnico"],
    "graduacao": ["Faculdade", "Centro Universitário", "Universidade", "Faculdade de Tecnologia"],
    "pos": ["Faculdade", "Instituto de Pós-Graduação", "Centro Universitário"],
    "mestrado": ["Universidade", "Universidade Estadual"],
    "doutorado": ["Universidade", "Universidade Estadual"],
}
INSTITUTION_NAME = ["Horizonte", "Santa Clara", "Vale do Sol", "Monte Verde", "Nova Esperança",
                    "Boa Vista", "Dom Pedro", "Rio Claro", "Primavera", "São Lucas",
                    "Castelo Branco", "Alvorada", "Palmares", "Serra Azul", "Bela Vista",
                    "Tiradentes", "Anchieta", "Santo Antônio", "Ipiranga", "Aurora"]

LANGUAGES = ["Inglês", "Espanhol", "Francês", "Italiano", "Alemão", "Libras"]

# Employer names are composed: "<segment word> <name part> <suffix>", e.g. "Transportadora
# Moura Ltda.". The segment follows the occupation group so a nurse works at a clinic.
COMPANY_SEGMENTS: dict[str, list[str]] = {
    "Trabalhador dos serviços, vendedor do comércio ou mercado": [
        "Supermercado", "Magazine", "Lojas", "Restaurante", "Padaria", "Farmácia",
        "Hotel", "Comercial", "Atacadista"],
    "Ocupação elementar": ["Limpadora", "Serviços Gerais", "Construtora", "Agropecuária",
                           "Logística", "Supermercado"],
    "Profissional das ciências ou intelectual": [
        "Hospital", "Clínica", "Colégio", "Engenharia", "Tecnologia", "Consultoria",
        "Escritório de Advocacia", "Laboratório", "Instituto"],
    "Trabalhador qualificado, operário ou artesão da construção, das artes mecânicas ou de outro ofício": [
        "Construtora", "Auto Center", "Metalúrgica", "Serralheria", "Móveis", "Instalações Elétricas",
        "Confecções"],
    "Trabalhador de apoio administrativo": [
        "Contabilidade", "Administradora", "Distribuidora", "Consultoria", "Imobiliária",
        "Comercial"],
    "Técnico ou profissional de nível médio": [
        "Hospital", "Clínica", "Tecnologia", "Engenharia", "Laboratório", "Imobiliária",
        "Informática"],
    "Operador de instalação ou máquina ou montador": [
        "Indústria", "Metalúrgica", "Transportadora", "Plásticos", "Alimentos", "Têxtil",
        "Autopeças"],
    "Diretor ou gerente": ["Grupo", "Holding", "Comercial", "Indústria", "Consultoria",
                           "Distribuidora"],
    "Trabalhador qualificado da agropecuária, florestal, da caça ou da pesca": [
        "Fazenda", "Agropecuária", "Sítio", "Cooperativa Agrícola", "Granja", "Pescados"],
    "Ocupação mal definida": ["Comercial", "Serviços", "Distribuidora"],
    "Membro das forças armadas, policial ou bombeiro militar": [
        "Segurança Patrimonial", "Vigilância", "Brigada de Incêndio"],
}
COMPANY_SUFFIXES = ["Ltda.", "Ltda", "S.A.", "S/A", "ME", "EIRELI", "& Cia.", "", "", ""]
# Public-sector employers for the armed forces / police group: no suffix, fictitious unit.
PUBLIC_EMPLOYERS = ["{n}º Batalhão de Polícia Militar", "{n}º Grupamento de Bombeiros",
                    "{n}º Batalhão de Infantaria"]
