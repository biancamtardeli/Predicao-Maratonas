# -*- coding: utf-8 -*-
"""
individual_features.py
------------------------
Etapa 3 do pipeline (Fase B): para cada participação individual (aluno x
evento), calcular o histórico ACUMULADO do aluno ANTES daquele evento
(nunca usando informação do próprio evento ou de eventos futuros -- isso
evitaria vazamento de dados / data leakage).

Gera exatamente as três variáveis descritas no projeto de pesquisa
(seção "Procedimentos Técnicos e Engenharia de Atributos"):

    - experiencia_acumulada : nº de eventos anteriores em que o aluno
                               já havia participado (qualquer olimpíada,
                               inclusive individuais como Algar/OBI).
    - historico_podio       : 1 se, em algum evento de EQUIPE anterior,
                               o aluno já foi "alto desempenho" (medalha),
                               conforme os rótulos calculados na Fase A.
    - bagagem_nivel_superior: nº de participações anteriores em etapas
                               de abrangência regional/nacional (ou seja,
                               fora da competição interna OLIP).
"""

import pandas as pd

from . import config


def calcular_atributos_individuais(df_limpo, eventos_equipe):
    """Recebe:
        df_limpo      -> saída de data_processing.limpar_dados
        eventos_equipe-> saída de team_labels.construir_tabela_eventos_equipe

    Devolve df_limpo com três novas colunas:
        experiencia_acumulada, historico_podio, bagagem_nivel_superior
    """
    df = df_limpo.sort_values(["nome_norm", "evento_ordem", "evento_id"]).copy()

    # Mapa (nome_norm, evento_id) -> a equipe daquele aluno naquele evento
    # foi "alto_desempenho"? (só existe para eventos de equipe válidos)
    medalha_por_participacao = {}
    for _, linha in eventos_equipe.iterrows():
        for membro in linha["membros"]:
            medalha_por_participacao[(membro, linha["evento_id"])] = bool(
                linha["alto_desempenho"] and linha["evento_valido"]
            )

    df["_teve_medalha_neste_evento"] = df.apply(
        lambda r: medalha_por_participacao.get((r["nome_norm"], r["evento_id"]), False),
        axis=1,
    )
    df["_eh_nivel_superior_olimpiada"] = ~df["olimpiada"].isin(["OLIP"])

    experiencia, historico_podio, bagagem = [], [], []
    experiencia_apos, historico_podio_apos, bagagem_apos = [], [], []

    for nome_norm, grupo in df.groupby("nome_norm", sort=False):
        grupo = grupo.sort_values(["evento_ordem", "evento_id"])

        exp_acum = 0
        podio_acum = False
        bagagem_acum = 0

        for _, linha in grupo.iterrows():
            # Os valores "ANTES" deste evento são anexados primeiro...
            experiencia.append(exp_acum)
            historico_podio.append(int(podio_acum))
            bagagem.append(bagagem_acum)

            # ...e só DEPOIS atualizamos os acumuladores com este evento,
            # de forma que a própria participação atual nunca "vaze" para
            # as suas próprias features.
            exp_acum += 1
            if linha["_teve_medalha_neste_evento"]:
                podio_acum = True
            if linha["_eh_nivel_superior_olimpiada"]:
                bagagem_acum += 1

            # ...e também guardamos o estado JÁ ATUALIZADO (útil para
            # predict.py simular o "estado atual" de um aluno a partir
            # da sua última participação registrada).
            experiencia_apos.append(exp_acum)
            historico_podio_apos.append(int(podio_acum))
            bagagem_apos.append(bagagem_acum)

    df["experiencia_acumulada"] = experiencia
    df["historico_podio"] = historico_podio
    df["bagagem_nivel_superior"] = bagagem
    df["experiencia_acumulada_apos"] = experiencia_apos
    df["historico_podio_apos"] = historico_podio_apos
    df["bagagem_nivel_superior_apos"] = bagagem_apos

    df = df.drop(columns=["_teve_medalha_neste_evento", "_eh_nivel_superior_olimpiada"])
    return df.sort_values(["evento_ordem", "evento_id"]).reset_index(drop=True)


if __name__ == "__main__":
    from .data_processing import carregar_e_limpar
    from .team_labels import construir_tabela_eventos_equipe

    dados = carregar_e_limpar()
    eventos_equipe = construir_tabela_eventos_equipe(dados)
    dados_feat = calcular_atributos_individuais(dados, eventos_equipe)

    cols = ["nome", "evento_id", "olimpiada", "experiencia_acumulada",
            "historico_podio", "bagagem_nivel_superior"]
    print(dados_feat[dados_feat["nome_norm"] == "joao claudio dias araujo"][cols])
