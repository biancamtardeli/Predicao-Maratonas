import pandas as pd
from dados import carregar_dados

df = carregar_dados()

def calcular_historico(df):
    # Criar chave de evento para ordenação (ano + ordem interna)
    df['evento_id'] = df.groupby(['Ano', 'Olimpíada']).ngroup()

    historico_rows = []

    for idx, row in df.iterrows():
        nome      = row['Nome']
        evento    = row['evento_id']
        categoria = row['Categoria']

    # Participações ANTERIORES do mesmo aluno
    passado = df[(df['Nome'] == nome) & (df['evento_id'] < evento)]

    # ── Atributos individuais ─────────────────────────────────────────────
    exp_total    = len(passado)                        # nº de maratonas anteriores
    podios_hist  = int(passado['podio'].sum())          # pódios conquistados antes
    exp_superior = int((                               # participações em fase superior
    passado['Olimpíada'].isin({'Regional', 'Mineira', 'Nacional'})).sum())
    exp_olip     = int((passado['Olimpíada'] == 'OLIP').sum())
    ja_venceu    = int(podios_hist > 0)                # variável binária: já ganhou?

    historico_rows.append({
    'idx': idx,
    'exp_total':    exp_total,
    'podios_hist':  podios_hist,
    'exp_superior': exp_superior,
    'exp_olip':     exp_olip,
    'ja_venceu':    ja_venceu,
    })
    
    hist_df = pd.DataFrame(historico_rows).set_index('idx')
    df = df.join(hist_df)
    
    return df