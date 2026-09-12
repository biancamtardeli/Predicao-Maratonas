# -*- coding: utf-8 -*-
"""
name_reconciliation.py
------------------------
A planilha original tem pequenos erros de digitação em alguns nomes
(ex.: "Abner Sanuel P. Palmeira" em uma linha e "Abner Samuel P.
Palmeira" em outra). Como o histórico de cada aluno é calculado por
casamento exato do nome normalizado (nome_norm), um erro de digitação
faz o pipeline "perder" a experiência acumulada da pessoa, subestimando
suas features.

Este módulo agrupa nomes muito parecidos (usando similaridade de string)
em um único "nome_norm" canônico. É uma correspondência aproximada, então
o critério de similaridade é propositalmente conservador (limiar alto) e
o resultado do agrupamento é salvo em CSV para revisão manual da autora
do TCC -- é importante ela CONFERIR o relatório antes de confiar
cegamente na fusão, pois nomes parecidos podem, em casos raros,
pertencer a duas pessoas diferentes.
"""

import difflib
import pandas as pd

LIMIAR_SIMILARIDADE = 0.93


def reconciliar_nomes(df, coluna_nome_norm="nome_norm", limiar=LIMIAR_SIMILARIDADE):
    """Agrupa variações (erros de digitação) do mesmo nome normalizado.

    Retorna:
        df_atualizado -> cópia de df com nome_norm remapeado para o
                          nome canônico do cluster.
        relatorio      -> DataFrame listando cada nome original e para
                           qual canônico ele foi mapeado (para revisão
                           manual).
    """
    nomes_unicos = sorted(df[coluna_nome_norm].dropna().unique())

    pais = {n: n for n in nomes_unicos}

    def find(n):
        while pais[n] != n:
            pais[n] = pais[pais[n]]
            n = pais[n]
        return n

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            # canônico = string mais longa (heurística simples: nomes
            # completos tendem a ser a grafia mais "correta"/completa)
            canon = ra if len(ra) >= len(rb) else rb
            outro = rb if canon == ra else ra
            pais[outro] = canon

    for i, n1 in enumerate(nomes_unicos):
        for n2 in nomes_unicos[i + 1:]:
            # Otimização: só compara nomes com tamanho parecido
            if abs(len(n1) - len(n2)) > 4:
                continue
            ratio = difflib.SequenceMatcher(None, n1, n2).ratio()
            if ratio >= limiar:
                union(n1, n2)

    mapa = {n: find(n) for n in nomes_unicos}

    df_atualizado = df.copy()
    df_atualizado[coluna_nome_norm] = df_atualizado[coluna_nome_norm].map(mapa).fillna(
        df_atualizado[coluna_nome_norm]
    )

    relatorio = (
        pd.DataFrame({"nome_norm_original": list(mapa.keys()),
                       "nome_norm_canonico": list(mapa.values())})
        .query("nome_norm_original != nome_norm_canonico")
        .sort_values("nome_norm_canonico")
        .reset_index(drop=True)
    )

    return df_atualizado, relatorio


if __name__ == "__main__":
    from .data_processing import carregar_e_limpar

    dados = carregar_e_limpar()
    dados_rec, relatorio = reconciliar_nomes(dados)

    print(f"Nomes únicos antes: {dados['nome_norm'].nunique()}")
    print(f"Nomes únicos depois: {dados_rec['nome_norm'].nunique()}")
    print()
    print("Pares fundidos (REVISAR MANUALMENTE):")
    print(relatorio.to_string())
