import pandas as pd

def limite_medalha(olimpiada: str, categoria: str, ano: int) -> int:
    """
    Retorna a posição máxima que conta como medalha/alto-desempenho.
    Retorna 0 quando a combinação não distribui medalhas.
    """
    if olimpiada == 'OLIP':
        return 3  # top-3 por categoria

    if olimpiada in ('Regional', 'Mineira', 'Nacional'):
        if categoria != 'Superior':
            return 0       # Técnico não compete nessas fases
        if ano <= 2017:
            return 9       # 3 ouro + 3 prata + 3 bronze
        elif ano <= 2025:
            return 12      # 4 de cada
        else:
            return 15      # 5 de cada (regra 2026+)

    # Olimpíadas menores (Minas de Minas, InterIF, Escola de Inverno…)
    return 3


def eh_podio(row) -> bool:
    if pd.isna(row['Classificação']) or pd.isna(row['Ano']):
        return False
    lim = limite_medalha(row['Olimpíada'], row['Categoria'], int(row['Ano']))
    return lim > 0 and int(row['Classificação']) <= lim