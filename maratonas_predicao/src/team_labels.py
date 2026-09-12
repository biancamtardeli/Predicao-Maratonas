# -*- coding: utf-8 -*-
"""
team_labels.py
---------------
Etapa 2 do pipeline (Fase A): calcular, para cada EVENTO x EQUIPE, se a
equipe teve "alto desempenho" (medalista) ou não, segundo os critérios
definidos em config.LIMIAR_POR_OLIMPIADA.

Esta etapa é feita ANTES da engenharia de atributos individuais porque o
histórico de pódios de um aluno (feature "historico_podio") depende de
sabermos, para cada participação passada dele, se aquela equipe foi ou
não medalista.

Responde diretamente à pergunta (2) da banca: "Como será definido o
limiar de alto desempenho para uma equipe?"
"""

import pandas as pd

from . import config


def _classificar_evento(grupo, regra):
    """Aplica a regra de corte de um evento específico (já isolado por
    Olimpíada+Ano+Edição) e devolve uma Series booleana alto_desempenho,
    indexada da mesma forma que `grupo`, além de um flag de validade.

    Duas regras possíveis (ver comentário detalhado em config.py):
      - "posicao_interna": classificação é relativa só às equipes do
        campus presentes na base; exige um número mínimo de equipes do
        campus no evento para o corte fazer sentido.
      - "posicao_global": classificação já é um rank absoluto/estadual/
        nacional; o corte de medalha vale mesmo que só uma equipe do
        campus tenha disputado aquele evento -- só é necessário que a
        classificação exista.
    """
    corte_posicao = regra.get("n", config.TOP_N_PODIO_PADRAO)

    if regra["rule"] == "posicao_interna":
        n_equipes_campus = grupo.index.nunique()
        min_equipes = regra.get("min_equipes", config.N_MIN_EQUIPES_PARA_PODIO)
        if n_equipes_campus < min_equipes:
            # Poucas equipes do campus no evento: corte se torna
            # arbitrário/pouco discriminativo. Evento inválido p/ treino.
            return pd.Series(False, index=grupo.index), False
        alto = grupo["classificacao"] <= corte_posicao
        return alto, True

    elif regra["rule"] == "posicao_global":
        tem_classificacao = grupo["classificacao"].notna()
        if not tem_classificacao.any():
            return pd.Series(False, index=grupo.index), False
        alto = (grupo["classificacao"] <= corte_posicao) & tem_classificacao
        # Evento é válido linha a linha: equipes sem classificação
        # registrada (raro) simplesmente não entram no treino.
        return alto, tem_classificacao

    else:
        raise ValueError(f"Regra desconhecida: {regra['rule']}")


def construir_tabela_eventos_equipe(df_limpo):
    """Recebe o DataFrame limpo (uma linha por participação individual)
    e devolve uma tabela com uma linha por EQUIPE x EVENTO, já com o
    rótulo alto_desempenho e a lista de integrantes (nome_norm).

    Eventos de olimpíadas INDIVIDUAIS (Algar, OBI) são ignorados aqui --
    eles não formam equipe (ver discussão na pergunta 4 da banca,
    tratada em README.md).
    """
    df_equipes = df_limpo[
        (~df_limpo["eh_individual"]) & (df_limpo["equipe"].notna())
    ].copy()

    linhas = []
    membros_por_evento_equipe = {}

    # Processa evento a evento (Olimpíada+Ano+Edição), pois o corte de
    # "alto desempenho" é relativo ao tamanho daquele evento específico.
    for evento_id, grupo_evento in df_equipes.groupby("evento_id"):
        olimpiada = grupo_evento["olimpiada"].iloc[0]
        regra = config.LIMIAR_POR_OLIMPIADA.get(
            olimpiada,
            {"rule": "medal", "n": config.TOP_N_PODIO_PADRAO,
             "min_equipes": config.N_MIN_EQUIPES_PARA_PODIO},
        )

        # Uma linha por equipe (classificação já é igual para os membros)
        por_equipe = grupo_evento.groupby("equipe").agg(
            classificacao=("classificacao", "first"),
            ano=("ano", "first"),
            edicao=("edicao", "first"),
            evento_ordem=("evento_ordem", "first"),
            n_membros=("nome_norm", "nunique"),
        ).reset_index()
        por_equipe["olimpiada"] = olimpiada
        por_equipe["evento_id"] = evento_id
        por_equipe["n_equipes_evento"] = por_equipe["equipe"].nunique()

        alto, valido = _classificar_evento(por_equipe.set_index("equipe"), regra)
        alto = alto.reset_index(drop=True)
        por_equipe["alto_desempenho"] = alto.astype(int)
        if isinstance(valido, bool):
            por_equipe["evento_valido"] = valido
        else:
            por_equipe["evento_valido"] = valido.reset_index(drop=True)

        linhas.append(por_equipe)

        for equipe, membros in grupo_evento.groupby("equipe")["nome_norm"]:
            membros_por_evento_equipe[(evento_id, equipe)] = list(membros)

    tabela = pd.concat(linhas, ignore_index=True)
    tabela["membros"] = tabela.apply(
        lambda r: membros_por_evento_equipe[(r["evento_id"], r["equipe"])], axis=1
    )

    colunas = [
        "evento_id", "olimpiada", "ano", "edicao", "evento_ordem", "equipe",
        "membros", "n_membros", "classificacao", "n_equipes_evento",
        "alto_desempenho", "evento_valido",
    ]
    return tabela[colunas].sort_values(["evento_ordem", "evento_id", "equipe"]).reset_index(drop=True)


if __name__ == "__main__":
    from .data_processing import carregar_e_limpar

    dados = carregar_e_limpar()
    eventos_equipe = construir_tabela_eventos_equipe(dados)

    print(f"Total de equipes-evento: {len(eventos_equipe)}")
    print(f"Eventos válidos para treino: {eventos_equipe['evento_valido'].sum()}")
    print()
    print("Distribuição de alto_desempenho (apenas eventos válidos):")
    print(eventos_equipe[eventos_equipe["evento_valido"]]["alto_desempenho"].value_counts())
    print()
    print("Por olimpíada (eventos válidos):")
    validos = eventos_equipe[eventos_equipe["evento_valido"]]
    print(validos.groupby("olimpiada")["alto_desempenho"].agg(["count", "sum"]))
