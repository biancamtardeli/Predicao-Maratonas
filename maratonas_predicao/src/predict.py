# -*- coding: utf-8 -*-
"""
predict.py
-----------
Etapa 6 do pipeline (Fase E): dado um trio de alunos (pelos nomes) e a
olimpíada-alvo, estimar a probabilidade de a equipe ter "alto
desempenho" (medalha).

Este é o objetivo final do TCC: "testar combinações de perfis antes da
inscrição final", como descrito no projeto de pesquisa.

Responde à pergunta 4 da banca: o modelo prevê POR EQUIPE (a partir dos
nomes dos 3 integrantes) e pondera de forma diferente cada olimpíada
porque "olimpiada" é uma variável categórica de entrada do modelo -- ou
seja, a mesma equipe pode ter uma probabilidade diferente estimada para
o OLIP, para o Regional, etc.
"""

import numpy as np
import pandas as pd

from .data_processing import _normalizar_texto
from .team_features import COLUNAS_FEATURES_NUMERICAS, COLUNA_CATEGORICA


class PreditorDeEquipe:
    """Encapsula um modelo treinado + o histórico mais recente de cada
    aluno, permitindo simular equipes hipotéticas.
    """

    def __init__(self, modelo_treinado, dados_participacoes):
        """
        modelo_treinado      -> pipeline sklearn já treinado (ver train.py)
        dados_participacoes  -> saída de individual_features (histórico
                                 completo, usado para pegar o estado MAIS
                                 RECENTE de cada aluno)
        """
        self.modelo = modelo_treinado
        self._historico_mais_recente = (
            dados_participacoes
            .sort_values(["nome_norm", "evento_ordem"])
            .groupby("nome_norm")
            .tail(1)
            .set_index("nome_norm")
        )

    def _estado_atual_aluno(self, nome_aluno):
        """Devolve o histórico ACUMULADO mais atual do aluno (ou seja,
        já contando a última participação registrada), para simular
        "como ele está hoje". Alunos nunca vistos na base entram como
        estreantes (todas as features zeradas).
        """
        chave = _normalizar_texto(nome_aluno)
        if chave in self._historico_mais_recente.index:
            linha = self._historico_mais_recente.loc[chave]
            return {
                "experiencia_acumulada": linha["experiencia_acumulada_apos"],
                "historico_podio": linha["historico_podio_apos"],
                "bagagem_nivel_superior": linha["bagagem_nivel_superior_apos"],
                "sexo": linha["sexo"],
                "categoria": linha["categoria"],
                "ano_periodo": linha["ano_periodo"],
                "encontrado": True,
            }
        return {
            "experiencia_acumulada": 0, "historico_podio": 0,
            "bagagem_nivel_superior": 0, "sexo": None, "categoria": None,
            "ano_periodo": None, "encontrado": False,
        }

    def montar_features_equipe(self, nomes_integrantes, olimpiada):
        """Constrói a linha de features da equipe hipotética, no mesmo
        formato usado no treinamento (ver team_features.py)."""
        estados = [self._estado_atual_aluno(n) for n in nomes_integrantes]
        nao_encontrados = [n for n, e in zip(nomes_integrantes, estados) if not e["encontrado"]]

        exp = [e["experiencia_acumulada"] for e in estados]
        podio = [e["historico_podio"] for e in estados]
        bagagem = [e["bagagem_nivel_superior"] for e in estados]
        sexo = [e["sexo"] for e in estados if e["sexo"] is not None]
        categoria = [e["categoria"] for e in estados if e["categoria"] is not None]
        periodo = [e["ano_periodo"] for e in estados if e["ano_periodo"] is not None]

        linha = {
            "experiencia_media": np.mean(exp),
            "experiencia_maxima": np.max(exp),
            "experiencia_minima": np.min(exp),
            "soma_podios_anteriores": np.sum(podio),
            "tem_algum_medalista": int(np.max(podio) > 0) if podio else 0,
            "bagagem_nivel_superior_media": np.mean(bagagem),
            "bagagem_nivel_superior_soma": np.sum(bagagem),
            "proporcao_feminino": (
                np.mean([s == "F" for s in sexo]) if sexo else 0.0
            ),
            "proporcao_ensino_superior": (
                np.mean([c == "Superior" for c in categoria]) if categoria else 0.0
            ),
            "diversidade_periodo": np.std(periodo) if len(periodo) > 1 else 0.0,
            "periodo_medio": np.mean(periodo) if periodo else np.nan,
            "tamanho_equipe": len(nomes_integrantes),
            COLUNA_CATEGORICA: olimpiada,
        }
        return pd.DataFrame([linha]), nao_encontrados

    def prever(self, nomes_integrantes, olimpiada):
        """Devolve (probabilidade_alto_desempenho, alunos_nao_encontrados).

        `nomes_integrantes` deve ter os nomes dos 3 (ou 2) alunos, como
        escritos normalmente (não precisa ser exatamente igual à grafia
        da planilha -- o nome é normalizado internamente).
        """
        X, nao_encontrados = self.montar_features_equipe(nomes_integrantes, olimpiada)
        colunas_modelo = COLUNAS_FEATURES_NUMERICAS + [COLUNA_CATEGORICA]
        proba = self.modelo.predict_proba(X[colunas_modelo])[0, 1]
        return proba, nao_encontrados


def carregar_preditor(nome_modelo="RandomForest", ano_holdout=None):
    """Função de conveniência: roda o pipeline inteiro (limpeza até
    treino) e devolve um PreditorDeEquipe pronto para uso interativo.

    Uso típico (a partir da raiz do projeto):

        from src.predict import carregar_preditor
        preditor = carregar_preditor()
        prob, faltantes = preditor.prever(
            ["Fulano de Tal", "Ciclana Silva", "Beltrano Souza"],
            olimpiada="OLIP",
        )
        print(f"Probabilidade de alto desempenho: {prob:.1%}")
    """
    from .data_processing import carregar_e_limpar
    from .name_reconciliation import reconciliar_nomes
    from .team_labels import construir_tabela_eventos_equipe
    from .individual_features import calcular_atributos_individuais
    from .team_features import construir_tabela_equipes
    from .train import avaliar_holdout_temporal, separar_treino_teste_temporal, _montar_modelos
    from . import config

    dados = carregar_e_limpar()
    dados, _ = reconciliar_nomes(dados)
    eventos_equipe = construir_tabela_eventos_equipe(dados)
    dados_feat = calcular_atributos_individuais(dados, eventos_equipe)
    tabela_equipes = construir_tabela_equipes(dados_feat, eventos_equipe)

    # Para uso "em produção" (prever equipes futuras), treinamos com
    # TODOS os dados válidos disponíveis (não só com o holdout de
    # validação), pois queremos o modelo mais atualizado possível.
    validas = tabela_equipes[tabela_equipes["evento_valido"]]
    X = validas[COLUNAS_FEATURES_NUMERICAS + [COLUNA_CATEGORICA]]
    y = validas["alto_desempenho"].astype(int)

    modelos = _montar_modelos()
    if nome_modelo not in modelos:
        raise ValueError(
            f"Modelo '{nome_modelo}' indisponível. Opções: {list(modelos.keys())}"
        )
    pipe = modelos[nome_modelo]
    pipe.fit(X, y)

    return PreditorDeEquipe(pipe, dados_feat)


if __name__ == "__main__":
    preditor = carregar_preditor()

    # Exemplo ilustrativo com nomes fictícios / reais da base --
    # troque pelos nomes reais dos alunos que deseja simular.
    exemplo_nomes = ["João Cláudio Dias Araújo", "Lilian Cristina de Paula",
                      "Emerson Dias do Prado"]
    prob, faltantes = preditor.prever(exemplo_nomes, olimpiada="OLIP")

    print(f"Equipe: {', '.join(exemplo_nomes)}")
    print(f"Probabilidade estimada de alto desempenho na OLIP: {prob:.1%}")
    if faltantes:
        print(f"Atenção -- alunos não encontrados no histórico (tratados "
              f"como estreantes): {faltantes}")
