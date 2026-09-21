"""
MODELO PREDITIVO DE DESEMPENHO EM MARATONAS DE PROGRAMAÇÃO
IFSULDEMINAS - Campus Muzambinho

Metodologia conforme Projeto de Pesquisa (Bianca Martins Tardeli, 2026):
  1. Limpeza e filtragem dos dados
  2. Engenharia de Atributos Históricos (Feature Engineering)
  3. Agrupamento por equipes (linha = equipe única)
  4. Binarização da variável alvo por olimpíada/categoria/ano
  5. Treinamento: Random Forest, Gradient Boosting, XGBoost
  6. Avaliação: Acurácia, Precisão, Sensibilidade, F1-Score
  7. Importância dos atributos
  8. Simulador de equipes
"""

import warnings
from configuracoes import (
    PASTA_SAIDA,
    N_FOLDS
)
from equipes import criar_dataset_equipes
from features import calcular_historico
from regras import eh_podio
from dados import carregar_dados
from treinamento import preparar_dados, criar_modelos, treinar_modelos
from relatorios import (
    gerar_relatorio,
    gerar_graficos,
    salvar_resultados,
)
from simulador import simular_equipe

warnings.filterwarnings('ignore')

# ── Configurações ─────────────────────────────────────────────────────────────

print("=" * 80)
print("MODELO PREDITIVO — MARATONAS DE PROGRAMAÇÃO")
print("IFSULDEMINAS – Campus Muzambinho")
print("=" * 80)


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 1 — CARREGAMENTO E LIMPEZA
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/7] Carregando e limpando dados...")

df = carregar_dados()

print(f"  ✓ {len(df)} registros válidos | "
      f"{df['Nome'].nunique()} competidores únicos | "
      f"Olimpíadas: {df['Olimpíada'].unique().tolist()}")


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 2 — REGRAS DE MEDALHA
# ══════════════════════════════════════════════════════════════════════════════

df['podio'] = df.apply(eh_podio, axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 3 — ENGENHARIA DE ATRIBUTOS HISTÓRICOS (Feature Engineering)
#           Calculada "look-back"
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/7] Calculando atributos históricos por competidor...")

df = calcular_historico(df)

print(f"  ✓ Atributos históricos calculados para {len(df)} registros")


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 4 — AGRUPAMENTO POR EQUIPE (linha = 1 equipe)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/7] Agregando registros por equipe...")

equipes_df = criar_dataset_equipes(df)

print(f"  ✓ {len(equipes_df)} equipes únicas | "
      f"Alto desempenho: {int(equipes_df['alto_desempenho'].sum())} "
      f"({equipes_df['alto_desempenho'].mean()*100:.1f}%) | "
      f"Baixo desempenho: {int((equipes_df['alto_desempenho']==0).sum())}")


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 5 — PREPARAÇÃO PARA O MODELO
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/7] Preparando features e tratando desbalanceamento...")

X, y, X_bal, y_bal = preparar_dados(equipes_df)

print(f"  ✓ Features: {X.shape[1]} | "
      f"Amostras originais: {len(y)} | "
      f"Após balanceamento: {len(y_bal)}")

# Verificar distribuição de classes após balanceamento
n_pos = int(y_bal.sum())
n_neg = int((y_bal == 0).sum())
print(f"  ✓ Alto desempenho: {n_pos} | Baixo desempenho: {n_neg}")


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 6 — TREINAMENTO E AVALIAÇÃO (validação cruzada estratificada)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[5/7] Treinando modelos com validação cruzada ({N_FOLDS}-fold)...")

modelos = criar_modelos()

melhor_nome, resultados, melhor_modelo, modelos = treinar_modelos(
    X_bal,
    y_bal,
    modelos
)

print(f"\n  🏆 Melhor modelo: {melhor_nome} "
      f"(F1={resultados[melhor_nome]['F1-Score']:.4f})")

# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 7 — RELATÓRIO DETALHADO NO CONJUNTO ORIGINAL
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[6/7] Gerando relatório e gráficos...")

y_pred, y_prob = gerar_relatorio(
    melhor_modelo,
    melhor_nome,
    X,
    y,
)

equipes_pred = gerar_graficos(
    melhor_modelo,
    melhor_nome,
    modelos,
    resultados,
    X,
    y,
    X_bal,
    y_bal,
    equipes_df,
    y_pred,
    y_prob,
    PASTA_SAIDA,
)

salvar_resultados(
    resultados,
    equipes_pred,
    PASTA_SAIDA,
)

# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 8 — SIMULADOR INTERATIVO DE EQUIPES
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[7/7] Gerando simulador de equipes...")

# ── Exemplos de simulação ──────────────────────────────────────────────────
print("\n  ── Exemplos de uso do simulador ────────────────────────────")

# Exemplo 1: equipe experiente na OLIP Superior
simular_equipe(
    membros=[
        {'nome': 'Aluno A', 'sexo': 'M', 'periodo': 6, 'exp_total': 5, 'podios_hist': 2, 'exp_superior': 3},
        {'nome': 'Aluno B', 'sexo': 'M', 'periodo': 5, 'exp_total': 4, 'podios_hist': 1, 'exp_superior': 2},
        {'nome': 'Aluno C', 'sexo': 'F', 'periodo': 6, 'exp_total': 3, 'podios_hist': 1, 'exp_superior': 1},
    ],
    olimpiada='OLIP', 
    melhor_modelo=melhor_modelo, 
    X=X, 
    categoria='Superior', 
    ano=2026
)

# Exemplo 2: equipe novata na OLIP Superior
simular_equipe(
    membros=[
        {'nome': 'Calouro A', 'sexo': 'M', 'periodo': 1, 'exp_total': 0, 'podios_hist': 0, 'exp_superior': 0},
        {'nome': 'Calouro B', 'sexo': 'M', 'periodo': 1, 'exp_total': 0, 'podios_hist': 0, 'exp_superior': 0},
        {'nome': 'Calouro C', 'sexo': 'M', 'periodo': 2, 'exp_total': 1, 'podios_hist': 0, 'exp_superior': 0},
    ],
    olimpiada='Regional', 
    melhor_modelo=melhor_modelo, 
    X=X, 
    categoria='Superior', 
    ano=2026
)

# Exemplo 3: equipe veterana na Regional Superior
simular_equipe(
    membros=[
        {'nome': 'Veterano A', 'sexo': 'M', 'periodo': 8, 'exp_total': 7, 'podios_hist': 3, 'exp_superior': 4},
        {'nome': 'Veterano B', 'sexo': 'F', 'periodo': 7, 'exp_total': 6, 'podios_hist': 2, 'exp_superior': 3},
        {'nome': 'Intermediário', 'sexo': 'M', 'periodo': 5, 'exp_total': 3, 'podios_hist': 1, 'exp_superior': 1},
    ],
    olimpiada='Regional', 
    melhor_modelo=melhor_modelo, 
    X=X, 
    categoria='Superior', 
    ano=2026
)


# ══════════════════════════════════════════════════════════════════════════════
# FINALIZANDO
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("CONCLUÍDO!")
print(f"{'='*80}")
print(f"\nArquivos gerados em: {PASTA_SAIDA}")
print("  • modelo_preditivo_resultados.png  — gráficos completos")
print("  • comparacao_modelos.csv           — métricas dos 3 modelos")
print("  • equipes_com_predicao.csv         — todas as equipes com prob. prevista")
print(f"\nMelhor modelo: {melhor_nome}")
print(f"  F1-Score (CV): {resultados[melhor_nome]['F1-Score']:.4f}")
print(f"  Acurácia (CV): {resultados[melhor_nome]['Acurácia']:.4f}")
print(f"\nPara simular uma nova equipe, chame a função simular_equipe() no final do script.")
print(f"{'='*80}\n")
