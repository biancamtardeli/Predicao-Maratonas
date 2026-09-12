# -*- coding: utf-8 -*-
"""
team_features.py
------------------
Etapa 4 do pipeline (Fase C): agregar os atributos individuais (já
calculados de forma cronologicamente segura na Fase B) em uma linha por
EQUIPE x EVENTO, pronta para treinar os modelos de classificação.

Segue a seção "Tabulação e Agrupamento por Equipes" do projeto de
pesquisa: para cada trio, calculamos média/máximo de experiência, soma
de pódios anteriores e índices de diversidade/composição.
"""

import numpy as np
import pandas as pd


def construir_tabela_equipes(dados_participacoes, eventos_equipe):
    """Recebe:
        dados_participacoes -> saída de individual_features (uma linha
                                por aluno x evento, já com as 3 features
                                individuais calculadas)
        eventos_equipe       -> saída de team_labels (uma linha por
                                 equipe x evento, com o rótulo)

    Devolve uma linha por EQUIPE x EVENTO com os atributos agregados +
    o rótulo alvo (alto_desempenho) + a flag evento_valido.
    """
    part = dados_participacoes[
        (~dados_participacoes["eh_individual"]) & (dados_participacoes["equipe"].notna())
    ].copy()

    linhas = []
    for _, evento in eventos_equipe.iterrows():
        membros_norm = evento["membros"]
        sub = part[
            (part["evento_id"] == evento["evento_id"])
            & (part["nome_norm"].isin(membros_norm))
        ]
        if sub.empty:
            continue

        linha = {
            "evento_id": evento["evento_id"],
            "olimpiada": evento["olimpiada"],
            "ano": evento["ano"],
            "edicao": evento["edicao"],
            "evento_ordem": evento["evento_ordem"],
            "equipe": evento["equipe"],
            "integrantes": ", ".join(sorted(sub["nome"].unique())),
            "integrantes_norm": tuple(sorted(sub["nome_norm"].unique())),
            "tamanho_equipe": sub["nome_norm"].nunique(),

            "experiencia_media": sub["experiencia_acumulada"].mean(),
            "experiencia_maxima": sub["experiencia_acumulada"].max(),
            "experiencia_minima": sub["experiencia_acumulada"].min(),

            "soma_podios_anteriores": sub["historico_podio"].sum(),
            "tem_algum_medalista": int(sub["historico_podio"].max() > 0),

            "bagagem_nivel_superior_media": sub["bagagem_nivel_superior"].mean(),
            "bagagem_nivel_superior_soma": sub["bagagem_nivel_superior"].sum(),

            "proporcao_feminino": (sub["sexo"] == "F").mean(),
            "proporcao_ensino_superior": (sub["categoria"] == "Superior").mean(),

            # diversidade do período/ano de curso dos integrantes: quanto
            # maior o desvio padrão, mais heterogênea (veteranos + novatos)
            "diversidade_periodo": sub["ano_periodo"].std(ddof=0)
            if sub["ano_periodo"].notna().sum() > 1 else 0.0,
            "periodo_medio": sub["ano_periodo"].mean(),

            "n_equipes_evento": evento["n_equipes_evento"],
            "classificacao": evento["classificacao"],
            "alto_desempenho": evento["alto_desempenho"],
            "evento_valido": evento["evento_valido"],
        }
        linhas.append(linha)

    tabela = pd.DataFrame(linhas)

    # NaN em diversidade/experiência mínima quando não há dado suficiente
    tabela["diversidade_periodo"] = tabela["diversidade_periodo"].fillna(0.0)

    return tabela.sort_values(["evento_ordem", "evento_id", "equipe"]).reset_index(drop=True)


COLUNAS_FEATURES_NUMERICAS = [
    "experiencia_media", "experiencia_maxima", "experiencia_minima",
    "soma_podios_anteriores", "tem_algum_medalista",
    "bagagem_nivel_superior_media", "bagagem_nivel_superior_soma",
    "proporcao_feminino", "proporcao_ensino_superior",
    "diversidade_periodo", "periodo_medio", "tamanho_equipe",
]
COLUNA_CATEGORICA = "olimpiada"
COLUNA_ALVO = "alto_desempenho"


if __name__ == "__main__":
    from .data_processing import carregar_e_limpar
    from .name_reconciliation import reconciliar_nomes
    from .team_labels import construir_tabela_eventos_equipe
    from .individual_features import calcular_atributos_individuais

    dados = carregar_e_limpar()
    dados, _ = reconciliar_nomes(dados)
    eventos_equipe = construir_tabela_eventos_equipe(dados)
    dados_feat = calcular_atributos_individuais(dados, eventos_equipe)
    tabela_equipes = construir_tabela_equipes(dados_feat, eventos_equipe)

    print(f"Total de linhas (equipe x evento): {len(tabela_equipes)}")
    print(f"Linhas válidas para treino: {tabela_equipes['evento_valido'].sum()}")
    print()
    print(tabela_equipes[tabela_equipes["evento_valido"]][
        ["evento_id", "equipe", "experiencia_media", "soma_podios_anteriores",
         "proporcao_feminino", "alto_desempenho"]
    ].head(15))
