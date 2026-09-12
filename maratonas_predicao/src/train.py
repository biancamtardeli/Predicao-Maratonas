# -*- coding: utf-8 -*-
"""
train.py
---------
Etapa 5 do pipeline (Fase D): treinar e avaliar os modelos de
classificação supervisionada que estimam se uma equipe terá "alto
desempenho" (medalha) ou não.

Este módulo responde diretamente a duas perguntas da banca avaliadora:

  (1) Uso da OLIP 2026 como teste temporal "no mundo real": o parâmetro
      config.ANO_TESTE_HOLDOUT separa um HOLDOUT CRONOLÓGICO -- o modelo
      é treinado apenas com eventos ANTERIORES a esse ano e avaliado nos
      eventos DAQUELE ano (que o modelo nunca viu). Assim que a OLIP 2026
      acontecer e for adicionada ao CSV, basta trocar
      ANO_TESTE_HOLDOUT = 2026 para reproduzir exatamente a validação
      sugerida pela banca.

  (3) Classes desbalanceadas: usamos
        - StratifiedKFold na validação cruzada (garante as duas classes
          em todas as dobras de treino/validação);
        - class_weight="balanced" nos algoritmos baseados em árvore
          (penaliza mais os erros na classe minoritária, sem precisar
          duplicar/inventar dados);
        - SMOTE (oversampling sintético) como alternativa opcional,
          aplicado SOMENTE dentro dos folds de treino (nunca no teste),
          para evitar vazamento de informação.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score,
    recall_score, f1_score, accuracy_score,
)

from . import config
from .team_features import COLUNAS_FEATURES_NUMERICAS, COLUNA_CATEGORICA, COLUNA_ALVO

try:
    from xgboost import XGBClassifier
    TEM_XGBOOST = True
except ImportError:
    TEM_XGBOOST = False

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    TEM_IMBLEARN = True
except ImportError:
    TEM_IMBLEARN = False


def _preprocessador():
    """OneHotEncoder para a olimpíada (permite ao modelo ponderar de
    forma diferente cada tipo de competição -- resposta parcial à
    pergunta 4 da banca sobre ponderar competições de forma diferente).
    As variáveis numéricas passam por uma imputação simples pela mediana
    (algumas equipes antigas não têm "Ano / período" preenchido na
    planilha original, o que geraria NaN em periodo_medio).
    """
    return ColumnTransformer(
        transformers=[
            ("olimpiada", OneHotEncoder(handle_unknown="ignore"), [COLUNA_CATEGORICA]),
            ("numericas", SimpleImputer(strategy="median"), COLUNAS_FEATURES_NUMERICAS),
        ],
    )


def _montar_modelos(usar_smote=False):
    """Devolve um dicionário {nome_modelo: pipeline sklearn}."""
    modelos_base = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=config.RANDOM_STATE,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            random_state=config.RANDOM_STATE,
        ),
    }
    if TEM_XGBOOST:
        modelos_base["XGBoost"] = XGBClassifier(
            n_estimators=300, eval_metric="logloss",
            random_state=config.RANDOM_STATE,
        )
    else:
        warnings.warn(
            "Pacote 'xgboost' não encontrado -- pulando o modelo XGBoost. "
            "Instale com: pip install xgboost"
        )

    pipelines = {}
    for nome, modelo in modelos_base.items():
        pre = _preprocessador()
        if usar_smote and TEM_IMBLEARN:
            pipelines[nome] = ImbPipeline(steps=[
                ("prep", pre),
                ("smote", SMOTE(random_state=config.RANDOM_STATE, k_neighbors=3)),
                ("clf", modelo),
            ])
        else:
            if usar_smote and not TEM_IMBLEARN:
                warnings.warn(
                    "Pacote 'imbalanced-learn' não encontrado -- SMOTE "
                    "desativado, usando apenas class_weight='balanced'. "
                    "Instale com: pip install imbalanced-learn"
                )
            pipelines[nome] = Pipeline(steps=[("prep", pre), ("clf", modelo)])

    return pipelines


def separar_treino_teste_temporal(tabela_equipes, ano_holdout=None):
    """Separa a base em treino (anos < ano_holdout) e teste (ano ==
    ano_holdout), usando apenas linhas de eventos_valido=True.

    Esta é a validação "no mundo real" pedida pela banca (pergunta 1):
    o modelo nunca vê o futuro durante o treino.
    """
    ano_holdout = ano_holdout or config.ANO_TESTE_HOLDOUT
    validas = tabela_equipes[tabela_equipes["evento_valido"]].copy()

    treino = validas[validas["ano"] < ano_holdout]
    teste = validas[validas["ano"] == ano_holdout]

    return treino, teste


def validacao_cruzada(tabela_equipes, usar_smote=False, n_splits=5):
    """Roda validação cruzada estratificada em TODA a base válida
    (não usa o holdout temporal) e devolve um DataFrame comparando os
    modelos por Acurácia, Precisão, Recall e F1 (médias +/- desvio).

    Endereça a pergunta 3 da banca: StratifiedKFold garante que cada
    fold de treino/validação tenha as duas classes representadas.
    """
    validas = tabela_equipes[tabela_equipes["evento_valido"]].copy()
    X = validas[COLUNAS_FEATURES_NUMERICAS + [COLUNA_CATEGORICA]]
    y = validas[COLUNA_ALVO].astype(int)

    n_splits = min(n_splits, y.value_counts().min())
    if n_splits < 2:
        raise ValueError(
            "Classe minoritária tem menos de 2 exemplos -- não é possível "
            "fazer validação cruzada estratificada com os dados atuais."
        )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    scoring = {
        "acuracia": "accuracy",
        "precisao": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    resultados = []
    for nome, pipe in _montar_modelos(usar_smote=usar_smote).items():
        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
        resultados.append({
            "modelo": nome,
            "n_folds": n_splits,
            **{f"{m}_media": scores[f"test_{m}"].mean() for m in scoring},
            **{f"{m}_desvio": scores[f"test_{m}"].std() for m in scoring},
        })

    return pd.DataFrame(resultados).sort_values("f1_media", ascending=False)


def avaliar_holdout_temporal(tabela_equipes, ano_holdout=None, usar_smote=False):
    """Treina cada modelo só com o passado e avalia no ano-holdout
    (ex.: OLIP 2026, quando disponível). Devolve métricas + o modelo
    treinado, para poder ser usado depois em predict.py.
    """
    treino, teste = separar_treino_teste_temporal(tabela_equipes, ano_holdout)

    if len(teste) == 0:
        warnings.warn(
            f"Nenhum evento válido encontrado para o ano de holdout "
            f"({ano_holdout or config.ANO_TESTE_HOLDOUT}). Ajuste "
            f"config.ANO_TESTE_HOLDOUT ou insira dados desse ano."
        )
        return None, None

    X_treino = treino[COLUNAS_FEATURES_NUMERICAS + [COLUNA_CATEGORICA]]
    y_treino = treino[COLUNA_ALVO].astype(int)
    X_teste = teste[COLUNAS_FEATURES_NUMERICAS + [COLUNA_CATEGORICA]]
    y_teste = teste[COLUNA_ALVO].astype(int)

    resultados = []
    modelos_treinados = {}
    for nome, pipe in _montar_modelos(usar_smote=usar_smote).items():
        pipe.fit(X_treino, y_treino)
        y_pred = pipe.predict(X_teste)

        resultados.append({
            "modelo": nome,
            "n_treino": len(treino),
            "n_teste": len(teste),
            "acuracia": accuracy_score(y_teste, y_pred),
            "precisao": precision_score(y_teste, y_pred, zero_division=0),
            "recall": recall_score(y_teste, y_pred, zero_division=0),
            "f1": f1_score(y_teste, y_pred, zero_division=0),
        })
        modelos_treinados[nome] = pipe

    return pd.DataFrame(resultados).sort_values("f1", ascending=False), modelos_treinados


if __name__ == "__main__":
    from .data_processing import carregar_e_limpar
    from .name_reconciliation import reconciliar_nomes
    from .team_labels import construir_tabela_eventos_equipe
    from .individual_features import calcular_atributos_individuais
    from .team_features import construir_tabela_equipes

    dados = carregar_e_limpar()
    dados, _ = reconciliar_nomes(dados)
    eventos_equipe = construir_tabela_eventos_equipe(dados)
    dados_feat = calcular_atributos_individuais(dados, eventos_equipe)
    tabela_equipes = construir_tabela_equipes(dados_feat, eventos_equipe)

    print("=" * 70)
    print("VALIDAÇÃO CRUZADA ESTRATIFICADA (toda a base válida)")
    print("=" * 70)
    print(validacao_cruzada(tabela_equipes).to_string(index=False))

    print()
    print("=" * 70)
    print(f"HOLDOUT TEMPORAL (treino < {config.ANO_TESTE_HOLDOUT}, "
          f"teste = {config.ANO_TESTE_HOLDOUT})")
    print("=" * 70)
    metricas, _ = avaliar_holdout_temporal(tabela_equipes)
    if metricas is not None:
        print(metricas.to_string(index=False))
