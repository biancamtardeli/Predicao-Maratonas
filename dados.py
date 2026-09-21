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

    # Remover Algar/OBI: competições individuais (sem coluna Equipe preenchida)
    OLIMPIADAS_INDIVIDUAIS = {'Algar', 'OBI'}
    df = df[~df['Olimpíada'].isin(OLIMPIADAS_INDIVIDUAIS)].copy()

    # Ordenação cronológica — essencial para feature engineering
    df = df.sort_values(['Ano', 'Olimpíada', 'Nome']).reset_index(drop=True)
    return df