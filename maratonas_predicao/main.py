# -*- coding: utf-8 -*-
"""
main.py
--------
Executa o pipeline completo, do CSV bruto até a avaliação dos modelos,
e salva os artefatos intermediários e finais em outputs/, para que a
autora possa inspecionar cada etapa (útil para a escrita do TCC e para
apresentar à banca).

Uso:
    python main.py
"""

import warnings
import pandas as pd

from src.data_processing import carregar_e_limpar
from src.name_reconciliation import reconciliar_nomes
from src.team_labels import construir_tabela_eventos_equipe
from src.individual_features import calcular_atributos_individuais
from src.team_features import construir_tabela_equipes, COLUNAS_FEATURES_NUMERICAS, COLUNA_CATEGORICA
from src.train import validacao_cruzada, avaliar_holdout_temporal, separar_treino_teste_temporal
from src import config

PASTA_SAIDA = "outputs"


def main():
    from src.train import TEM_XGBOOST
    if not TEM_XGBOOST:
        print("[aviso] pacote 'xgboost' não encontrado neste ambiente -- os modelos "
              "usarão apenas RandomForest e GradientBoosting. Para incluir o XGBoost "
              "(recomendado na metodologia do projeto), rode: pip install xgboost\n")

    print("1/6 - Carregando e limpando dados brutos...")
    dados = carregar_e_limpar()

    print("2/6 - Reconciliando nomes com pequenas variações de digitação...")
    dados, relatorio_nomes = reconciliar_nomes(dados)
    relatorio_nomes.to_csv(f"{PASTA_SAIDA}/relatorio_nomes_reconciliados.csv", index=False)
    print(f"      -> {len(relatorio_nomes)} pares fundidos "
          f"(revisar {PASTA_SAIDA}/relatorio_nomes_reconciliados.csv)")

    print("3/6 - Calculando rótulos de alto desempenho por equipe/evento...")
    eventos_equipe = construir_tabela_eventos_equipe(dados)
    eventos_equipe.drop(columns=["membros"]).to_csv(
        f"{PASTA_SAIDA}/eventos_equipe_rotulados.csv", index=False
    )

    print("4/6 - Engenharia de atributos individuais (histórico cronológico)...")
    dados_feat = calcular_atributos_individuais(dados, eventos_equipe)
    dados_feat.to_csv(f"{PASTA_SAIDA}/participacoes_com_atributos.csv", index=False)

    print("5/6 - Agregando atributos em nível de equipe...")
    tabela_equipes = construir_tabela_equipes(dados_feat, eventos_equipe)
    tabela_equipes.drop(columns=["integrantes_norm"]).to_csv(
        f"{PASTA_SAIDA}/equipes_features.csv", index=False
    )
    validas = tabela_equipes["evento_valido"].sum()
    print(f"      -> {len(tabela_equipes)} equipes-evento no total, "
          f"{validas} válidas para treino "
          f"({tabela_equipes[tabela_equipes.evento_valido]['alto_desempenho'].mean():.1%} "
          f"de taxa de alto desempenho)")

    print("6/6 - Treinando e avaliando os modelos...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        cv_resultados = validacao_cruzada(tabela_equipes)
        cv_resultados.to_csv(f"{PASTA_SAIDA}/metricas_validacao_cruzada.csv", index=False)

        holdout_resultados, modelos = avaliar_holdout_temporal(tabela_equipes)
        if holdout_resultados is not None:
            holdout_resultados.to_csv(f"{PASTA_SAIDA}/metricas_holdout_temporal.csv", index=False)

    print()
    print("=" * 72)
    print("VALIDAÇÃO CRUZADA ESTRATIFICADA (toda a base válida)")
    print("=" * 72)
    print(cv_resultados.to_string(index=False))

    if holdout_resultados is not None:
        print()
        print("=" * 72)
        print(f"HOLDOUT TEMPORAL -- treino < {config.ANO_TESTE_HOLDOUT}, "
              f"teste = {config.ANO_TESTE_HOLDOUT}")
        print("=" * 72)
        print(holdout_resultados.to_string(index=False))

    print()
    print("Importância dos atributos (RandomForest, treinado com toda a base válida):")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print(importancia_atributos(tabela_equipes))

    print()
    print(f"Arquivos salvos em {PASTA_SAIDA}/. Pipeline concluído.")


def importancia_atributos(tabela_equipes):
    """Treina um RandomForest com toda a base válida e extrai a
    importância de cada atributo -- ajuda a responder a pergunta de
    pesquisa sobre quais características mais se correlacionam com o
    desempenho das equipes (objetivo específico 3 do projeto)."""
    from src.train import _montar_modelos
    from src.team_features import COLUNA_ALVO

    validas = tabela_equipes[tabela_equipes["evento_valido"]]
    X = validas[COLUNAS_FEATURES_NUMERICAS + [COLUNA_CATEGORICA]]
    y = validas[COLUNA_ALVO].astype(int)

    pipe = _montar_modelos()["RandomForest"]
    pipe.fit(X, y)

    nomes_colunas = list(
        pipe.named_steps["prep"].named_transformers_["olimpiada"]
        .get_feature_names_out(["olimpiada"])
    ) + COLUNAS_FEATURES_NUMERICAS

    importancias = pd.Series(
        pipe.named_steps["clf"].feature_importances_, index=nomes_colunas
    ).sort_values(ascending=False)

    importancias.to_csv(f"{PASTA_SAIDA}/importancia_atributos.csv", header=["importancia"])
    return importancias.to_string()


if __name__ == "__main__":
    main()
