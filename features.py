import pandas as pd

def calcular_historico(df):
    df['evento_id'] = df.groupby(['Ano', 'Olimpíada']).ngroup()
    historico_rows = []

    for idx, row in df.iterrows():
        nome      = row['Nome']
        evento    = row['evento_id']

        passado = df[(df['Nome'] == nome) & (df['evento_id'] < evento)]

        # Separar o que é equipe do que é individual
        passado_indiv = passado[passado['Olimpíada'].isin({'Algar', 'OBI'})]
        passado_equipe = passado[~passado['Olimpíada'].isin({'Algar', 'OBI'})]

        exp_total    = len(passado)
        
        podios_equipe = int(passado_equipe['podio'].sum())
        podios_indiv  = int(passado_indiv['podio'].sum())
        
        # ── CÁLCULO DE PESO ──
        # Pódio individual vale 1.5 vezes o pódio em equipe
        score_podios = podios_equipe + (podios_indiv * 1.5) 

        exp_superior = int((passado['Olimpíada'].isin({'Regional', 'Mineira', 'Nacional'})).sum())
        exp_olip     = int((passado['Olimpíada'] == 'OLIP').sum())
        ja_venceu    = int((podios_equipe + podios_indiv) > 0)

        historico_rows.append({
            'idx': idx,
            'exp_total':    exp_total,
            'podios_indiv': podios_indiv,
            'score_podios': score_podios,
            'exp_superior': exp_superior,
            'exp_olip':     exp_olip,
            'ja_venceu':    ja_venceu,
        })
    
    hist_df = pd.DataFrame(historico_rows).set_index('idx')
    df = df.join(hist_df)
    
    return df