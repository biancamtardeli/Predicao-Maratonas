import pandas as pd
import numpy as np
from regras import limite_medalha
from configuracoes import MIN_MEMBROS

def criar_dataset_equipes(df):
    
    # Descartar registros sem equipe (Algar já foi removido, mas pode haver NaN)
    df_eq = df.dropna(subset=['Equipe']).copy()
    df_eq = df_eq[df_eq['Equipe'].str.lower() != 'nan']
    
    def agregar_equipe(g):
        """Transforma N linhas de membros em 1 linha de equipe."""
        n = len(g)

        # ── Atributos de composição da equipe ─────────────────────────────────
        proporcao_feminino  = (g['Sexo'] == 'F').mean()
        proporcao_superior  = (g['Categoria'] == 'Superior').mean()
        periodo_medio       = g['Periodo'].mean()
        periodo_max         = g['Periodo'].max()

        # ── Atributos históricos agregados ────────────────────────────────────
        exp_media           = g['exp_total'].mean()
        exp_max             = g['exp_total'].max()       
        exp_min             = g['exp_total'].min()       
        
        # Novas agregações de pódio:
        podios_indiv_soma   = g['podios_indiv'].sum() 
        score_podios_soma   = g['score_podios'].sum() 

        exp_superior_media  = g['exp_superior'].mean()   
        membros_ja_venceram = g['ja_venceu'].sum()

        # ── Variável alvo ──────────────────────────────────────────────────────
        # As colunas Olimpíada, Ano e Categoria ainda existem no grupo
        # (include_groups=False não remove as chaves do DataFrame, só do cálculo)
        classificacao   = g['Classificação'].iloc[0]
        olimpiada       = g.name[1]   # índice 1 = Olimpíada na chave (Ano, Olimpíada, Equipe)
        categoria_grupo = g['Categoria'].iloc[0]
        ano             = g.name[0]   # índice 0 = Ano

        lim = limite_medalha(olimpiada, categoria_grupo, int(ano))
        if lim == 0 or pd.isna(classificacao):
            alto_desempenho = np.nan
        else:
            alto_desempenho = int(int(classificacao) <= lim)

        return pd.Series({
            'olimpiada':             olimpiada,
            'categoria':             categoria_grupo,
            'ano':                   ano,
            'n_membros':             n,
            'proporcao_feminino':    proporcao_feminino,
            'proporcao_superior':    proporcao_superior,
            'periodo_medio':         periodo_medio,
            'periodo_max':           periodo_max,
            'exp_media':             exp_media,
            'exp_max':               exp_max,
            'exp_min':               exp_min,
            'exp_superior_media':    exp_superior_media,
            'membros_ja_venceram':   membros_ja_venceram,
            'classificacao':         classificacao,
            'alto_desempenho':       alto_desempenho,
            'exp_min':               exp_min,
            'podios_indiv_soma':     podios_indiv_soma, # Retornando
            'score_podios_soma':     score_podios_soma, # Retornando
            'exp_superior_media':    exp_superior_media,
        })

    equipes_df = (df_eq
                .groupby(['Ano', 'Olimpíada', 'Equipe'], sort=False)
                .apply(agregar_equipe, include_groups=False)
                .reset_index())

    # Filtrar equipes com tamanho inválido e alvo indefinido
    equipes_df = equipes_df[equipes_df['n_membros'] >= MIN_MEMBROS]
    equipes_df = equipes_df.dropna(subset=['alto_desempenho'])
    
    return equipes_df