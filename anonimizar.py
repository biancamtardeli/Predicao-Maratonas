import pandas as pd

df = pd.read_csv("dados_das_maratonas.csv")
nomes_unicos = df["Nome"].dropna().unique()
mapa_anonimo = {nome: f"Estudante_{i+1:03d}" for i, nome in enumerate(nomes_unicos)}

df["Nome"] = df["Nome"].map(mapa_anonimo)
df.to_csv("dados_das_maratonas_anonimizado.csv", index=False, encoding="utf-8")