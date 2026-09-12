# -*- coding: utf-8 -*-
"""
config.py
---------
Configurações centrais do pipeline de predição de desempenho de equipes
em maratonas de programação (TCC - IFSULDEMINAS Campus Muzambinho).

Alterar os parâmetros aqui é a forma recomendada de ajustar o comportamento
do pipeline sem precisar mexer na lógica dos outros módulos.
"""

# Caminho do CSV bruto (relativo à raiz do projeto)
CAMINHO_CSV_BRUTO = "data/dados_maratonas.csv"

# -----------------------------------------------------------------------
# Classificação das olimpíadas quanto ao FORMATO (equipe x individual)
# -----------------------------------------------------------------------
# Algar e OBI são provas de resolução INDIVIDUAL (não há formação de trio).
# Elas não geram "eventos de equipe" para o modelo principal, mas a
# participação nelas conta como bagagem/experiência do aluno.
OLIMPIADAS_INDIVIDUAIS = {"Algar", "OBI"}

# Todas as demais são competições disputadas em TRIOS (equipes)
# (definidas dinamicamente em tempo de execução como: todas - individuais)

# -----------------------------------------------------------------------
# Ordem aproximada das competições dentro de um mesmo ano civil
# -----------------------------------------------------------------------
# Usada apenas para ordenar cronologicamente eventos que ocorrem no MESMO
# ano, evitando que uma prova de agosto "veja" o resultado de uma prova de
# novembro do mesmo ano ao calcular o histórico de um aluno.
# É uma aproximação (não temos o mês exato de cada evento na base) -
# documentar essa limitação no texto do TCC.
ORDEM_APROXIMADA_NO_ANO = {
    "OLIP": 1,
    "Mineira": 2,
    "Regional": 3,
    "Algar": 4,
    "Minas de Minas": 5,
    "InterIF": 6,
    "Escola de Inverno": 7,
    "Nacional": 8,
    "OBI": 9,
}
ORDEM_PADRAO = 5  # usada para olimpíadas não mapeadas acima

# -----------------------------------------------------------------------
# Regra de definição do rótulo "alto desempenho" (medalista) por evento
# -----------------------------------------------------------------------
# Resposta à pergunta da banca (2): o limiar é definido de forma RELATIVA
# a cada evento (Olimpíada + Ano + Edição), e não por um valor fixo de
# posição igual para toda a base. A lógica é:
#
#   - Se o evento tem N_MIN_EQUIPES_PARA_PODIO ou mais equipes:
#         alto_desempenho = 1 se Classificação <= TOP_N_PODIO (medalha:
#         ouro / prata / bronze), 0 caso contrário.
#   - Se o evento tem MENOS equipes que N_MIN_EQUIPES_PARA_PODIO:
#         o evento é marcado como "não confiável para treino"
#         (evento_valido = False) e é excluído do treinamento, pois com
#         poucas equipes qualquer corte é praticamente arbitrário e gera
#         rótulos pouco informativos (ex.: 2 equipes, top-3 = as 2 são
#         "alto desempenho", o que não discrimina nada).
#
# Esses parâmetros podem ser recalibrados por tipo de olimpíada dentro de
# LIMIAR_POR_OLIMPIADA (ex.: provas mais concorridas podem usar um corte
# percentual em vez de posição fixa).
TOP_N_PODIO_PADRAO = 3
N_MIN_EQUIPES_PARA_PODIO = 4

# -----------------------------------------------------------------------
# IMPORTANTE: descoberta feita durante a analise exploratoria dos dados.
# -----------------------------------------------------------------------
# A coluna "Classificacao" tem DUAS naturezas diferentes dependendo da
# olimpiada:
#
#   (a) Competicoes INTERNAS ao campus (a "Classificacao" e a posicao
#       relativa apenas entre as equipes do IFSULDEMINAS - Muzambinho
#       que disputaram aquele evento): OLIP, Escola de Inverno.
#       Nessas provas ha muitos EMPATES na classificacao (varias equipes
#       registradas com posicao "1"), entao um corte "top 3" captura
#       quase todo o campo (chegando a rotular 100% das equipes como
#       "alto desempenho" em alguns anos). Por isso usamos a posicao 1
#       (equipe vencedora / nivel ouro daquela edicao) como criterio.
#
#   (b) Competicoes EXTERNAS/multi-institucionais (a "Classificacao" e a
#       posicao da equipe do campus em um ranking GLOBAL, com dezenas de
#       equipes de outras instituicoes que nao estao nesta base de
#       dados): Regional, Mineira, Nacional, Minas de Minas, InterIF.
#       Nesses casos, "top 3" (ouro/prata/bronze) e o criterio de medalha
#       usual do ICPC/SBC e ja e diretamente comparavel entre eventos,
#       independentemente de quantas equipes do campus disputaram aquele
#       evento -- por isso NAO aplicamos o filtro de "minimo de equipes
#       do campus" nesses casos, apenas exigimos que a classificacao
#       exista (nao seja nula).
#
# rule = "posicao_interna" -> alto desempenho se classificacao <= n,
#                              exige >= min_equipes do CAMPUS no evento.
# rule = "posicao_global"  -> alto desempenho se classificacao <= n,
#                              valido sempre que a classificacao existir
#                              (nao depende de quantas equipes do campus
#                              disputaram aquele evento).
LIMIAR_POR_OLIMPIADA = {
    "OLIP":              {"rule": "posicao_interna", "n": 1, "min_equipes": 4},
    "Escola de Inverno": {"rule": "posicao_interna", "n": 1, "min_equipes": 4},
    "Regional":          {"rule": "posicao_global",  "n": 3},
    "Mineira":           {"rule": "posicao_global",  "n": 3},
    "Nacional":          {"rule": "posicao_global",  "n": 3},
    "Minas de Minas":    {"rule": "posicao_global",  "n": 3},
    "InterIF":           {"rule": "posicao_global",  "n": 3},
}

# -----------------------------------------------------------------------
# Divisão treino / teste temporal (resposta à pergunta da banca (1))
# -----------------------------------------------------------------------
# A OLIP 2026 ainda não está disponível na base (o evento é o próximo a
# acontecer no cronograma do projeto). Assim que os resultados da OLIP
# 2026 forem coletados e adicionados ao CSV, ANO_TESTE_HOLDOUT deve ser
# ajustado para 2026 -- o pipeline então treinará com tudo < 2026 e
# testará exatamente na OLIP 2026, simulando um cenário real de uso
# (prever o futuro com base apenas no passado).
#
# Enquanto a OLIP 2026 não ocorre, usamos o último ano completo disponível
# como holdout temporal "de ensaio", para já validar a metodologia.
ANO_TESTE_HOLDOUT = 2025

# -----------------------------------------------------------------------
# Reprodutibilidade
# -----------------------------------------------------------------------
RANDOM_STATE = 42
