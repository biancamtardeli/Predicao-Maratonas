# -*- coding: utf-8 -*-
"""
data_processing.py
-------------------
Etapa 1 do pipeline: carregar o CSV bruto exportado da planilha e produzir
uma tabela limpa, com um registro por PARTICIPAÇÃO (aluno x evento),
pronta para as etapas seguintes de engenharia de atributos.

Nenhuma feature preditiva é calculada aqui -- este módulo só limpa e
padroniza os dados brutos.
"""

import unicodedata
import pandas as pd

from . import config


def _normalizar_texto(valor):
    """Remove espaços extras, acentos e padroniza para minúsculas.

    Usado para criar uma CHAVE de comparação de nomes (nome_norm),
    já que a planilha tem inconsistências como espaços a mais ou
    "Luis" vs "Luís". O nome original (com acentos) é mantido em
    outra coluna para exibição.
    """
    if pd.isna(valor):
        return None
    texto = str(valor).strip()
    texto = " ".join(texto.split())  # colapsa espaços internos duplicados
    sem_acento = "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )
    return sem_acento.lower()


def carregar_dados_brutos(caminho_csv=None):
    """Lê o CSV exportado da planilha e devolve o DataFrame cru."""
    caminho_csv = caminho_csv or config.CAMINHO_CSV_BRUTO
    df = pd.read_csv(caminho_csv)
    return df


def limpar_dados(df):
    """Remove linhas vazias (artefato da exportação da planilha) e
    padroniza tipos e nomes de colunas relevantes.

    Retorna um DataFrame com uma linha por participação de aluno em um
    evento, contendo:
        nome, nome_norm, categoria, sexo, ano_periodo, equipe,
        classificacao, olimpiada, edicao, ano, evento_id, evento_ordem
    """
    df = df.copy()

    # Mantém só as colunas que de fato existem na planilha (ignora as
    # colunas "Coluna 1", "Coluna 2", ... que vêm vazias da exportação)
    colunas_uteis = [
        "Nome", "Categoria", "Sexo", "Ano / período", "Equipe",
        "Classificação", "Olimpíada", "Edição da Maratona", "Ano",
    ]
    df = df[colunas_uteis]

    df = df.rename(columns={
        "Nome": "nome",
        "Categoria": "categoria",
        "Sexo": "sexo",
        "Ano / período": "ano_periodo",
        "Equipe": "equipe",
        "Classificação": "classificacao",
        "Olimpíada": "olimpiada",
        "Edição da Maratona": "edicao",
        "Ano": "ano",
    })

    # Linhas totalmente vazias (sobra da planilha do Google Sheets)
    df = df[df["nome"].notna()].copy()

    # Padronização de texto
    df["nome"] = df["nome"].astype(str).str.strip()
    df["nome_norm"] = df["nome"].apply(_normalizar_texto)
    df["equipe"] = df["equipe"].astype(str).str.strip()
    df.loc[df["equipe"].isin(["nan", "None", ""]), "equipe"] = pd.NA

    df["olimpiada"] = df["olimpiada"].astype(str).str.strip()
    df["categoria"] = df["categoria"].astype(str).str.strip()
    df.loc[~df["categoria"].isin(["Superior", "Técnico"]), "categoria"] = pd.NA

    df["sexo"] = df["sexo"].astype(str).str.strip().str.upper()
    df.loc[~df["sexo"].isin(["M", "F"]), "sexo"] = pd.NA

    # Tipos numéricos
    for col in ["ano", "edicao", "classificacao", "ano_periodo"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ano"] = df["ano"].astype("Int64")
    df["edicao"] = df["edicao"].astype("Int64")

    # Identificador único de evento (uma "prova" específica)
    df["evento_id"] = (
        df["olimpiada"] + "_" + df["ano"].astype(str) + "_ed" + df["edicao"].astype(str)
    )

    # Ordem cronológica aproximada (ver comentário em config.py)
    prioridade = df["olimpiada"].map(config.ORDEM_APROXIMADA_NO_ANO).fillna(
        config.ORDEM_PADRAO
    )
    df["evento_ordem"] = df["ano"].astype(int) * 100 + prioridade.astype(int)

    df["eh_individual"] = df["olimpiada"].isin(config.OLIMPIADAS_INDIVIDUAIS)

    df = df.sort_values(["evento_ordem", "evento_id"]).reset_index(drop=True)

    return df


def carregar_e_limpar(caminho_csv=None):
    """Atalho: carrega + limpa em uma só chamada."""
    bruto = carregar_dados_brutos(caminho_csv)
    return limpar_dados(bruto)


if __name__ == "__main__":
    dados = carregar_e_limpar()
    print(f"Registros de participação válidos: {len(dados)}")
    print(f"Alunos únicos (nome normalizado): {dados['nome_norm'].nunique()}")
    print(f"Eventos únicos: {dados['evento_id'].nunique()}")
    print(dados.head(10))
