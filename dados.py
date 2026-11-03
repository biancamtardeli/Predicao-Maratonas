import pandas as pd
from configuracoes import ARQUIVO_CSV

def carregar_dados():
    try:
        df_raw = pd.read_csv(ARQUIVO_CSV, encoding='utf-8')
    except UnicodeDecodeError:
        df_raw = pd.read_csv(ARQUIVO_CSV, encoding='latin-1')

    # Manter apenas colunas relevantes e linhas com categoria válida
    COLUNAS = ['Nome', 'Categoria', 'Sexo', 'Ano / período',
           'Equipe', 'Classificação', 'Olimpíada', 'Ano']

    df = (df_raw[COLUNAS]
        .query("Categoria in ('Superior', 'Técnico')")
        .copy())

    # Normalização de strings
    for col in ['Nome', 'Categoria', 'Sexo', 'Equipe', 'Olimpíada']:
        df[col] = df[col].astype(str).str.strip()

    # Tipos numéricos
    df['Classificação'] = pd.to_numeric(df['Classificação'], errors='coerce')
    df['Ano']           = pd.to_numeric(df['Ano'],            errors='coerce')
    df['Periodo']       = pd.to_numeric(df['Ano / período'],  errors='coerce')

    # ── NOVA LÓGICA: INFERÊNCIA DE NÃO-MEDALHISTAS NA OLIP ───────────
    regional_df = df[df['Olimpíada'] == 'Regional'].copy()
    novas_linhas = []
    
    # Agrupar quem participou da Regional por Ano e Equipe
    grupos_regional = regional_df.groupby(['Ano', 'Equipe'])
    
    for (ano, equipe), dados_equipe in grupos_regional:
        if pd.isna(equipe) or equipe == 'nan' or equipe == 'None':
            continue
            
        # Verificar se essa equipe tem registro na OLIP nesse mesmo ano
        participou_olip = df[(df['Olimpíada'] == 'OLIP') & 
                             (df['Ano'] == ano) & 
                             (df['Equipe'] == equipe)].shape[0] > 0
        
        # Se não tem registro na OLIP, assumimos que participou mas não ganhou medalha
        if not participou_olip:
            equipe_olip = dados_equipe.copy()
            equipe_olip['Olimpíada'] = 'OLIP'
            # Uma classificação impossível de ganhar medalha para gerar alto_desempenho = 0
            equipe_olip['Classificação'] = 999 
            
            novas_linhas.append(equipe_olip)
            
    # Adicionar as equipes inferidas ao dataframe original
    if novas_linhas:
        df_inferidos = pd.concat(novas_linhas, ignore_index=True)
        df = pd.concat([df, df_inferidos], ignore_index=True)
    # ─────────────────────────────────────────────────────────────────

    # === TRECHO PARA IMPRESSÃO DE TESTE ===
    # Filtra apenas os registros inferidos (Classificação = 999)
    inferidos = df[df['Classificação'] == 999]
    
    print(f"\n[!] Encontrados {len(inferidos)} registros inferidos para a OLIP.")
    print("Exemplo das primeiras 5 linhas:\n")
    print(inferidos[['Ano', 'Equipe', 'Nome', 'Olimpíada', 'Classificação']].head())
    
    inferidos.to_csv('conferencia_olip.csv', index=False, encoding='utf-8')
    # =======================================

    # Ordenação cronológica — essencial para feature engineering
    df = df.sort_values(['Ano', 'Olimpíada', 'Nome']).reset_index(drop=True)
    df.to_csv('dados_processados.csv', index=False, encoding='utf-8')

    return df