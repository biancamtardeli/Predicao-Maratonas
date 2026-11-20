"""
MODELO PREDITIVO DE DESEMPENHO EM MARATONAS DE PROGRAMAÇÃO
IFSULDEMINAS - Campus Muzambinho

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
    gerar_graficos,
    salvar_resultados,
)
from simulador import simular_equipe
from sklearn.metrics import classification_report

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

X, y = preparar_dados(equipes_df)

print(f"  ✓ Features: {X.shape[1]} | Amostras originais: {len(y)}")
print(f"  ✓ Alto desempenho: {int(y.sum())} | Baixo desempenho: {int((y == 0).sum())}")

# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 6 — TREINAMENTO E AVALIAÇÃO (validação cruzada estratificada)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[5/7] Treinando modelos com validação cruzada ({N_FOLDS}-fold)...")

modelos = criar_modelos()

melhor_nome, resultados, melhor_modelo, modelos, y_pred_cv, y_prob_cv = treinar_modelos(
    X,
    y,
    modelos
)

print(f"\n  🏆 Melhor modelo: {melhor_nome} "
      f"(F1={resultados[melhor_nome]['F1-Score']:.4f})")

# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 7 — RELATÓRIO DETALHADO NO CONJUNTO ORIGINAL
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[6/7] Gerando relatório e gráficos...")
print(classification_report(y, y_pred_cv, target_names=['Baixo desempenho', 'Alto desempenho']))

equipes_pred = gerar_graficos(
    melhor_modelo,
    melhor_nome,
    modelos,
    resultados,
    X,
    y,
    equipes_df,
    y_pred_cv,
    y_prob_cv,
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

# ==============================================================================
# CENÁRIOS DE SIMULAÇÃO PARA A SEÇÃO 5.4 (OLIP E REGIONAL)
# ==============================================================================

cenarios_simulacao = [
    # --- CENÁRIOS OLIP ---
    {
        "nome_cenario": "OLIP 1 - Calouros Puros (Superior)",
        "olimpiada": "OLIP",
        "categoria": "Superior",
        "membros": [
            {'nome': 'Calouro A', 'sexo': 'M', 'periodo': 2, 'exp_total': 0, 'podios_hist': 0, 'exp_superior': 0},
            {'nome': 'Caloura B', 'sexo': 'F', 'periodo': 2, 'exp_total': 0, 'podios_hist': 0, 'exp_superior': 0},
            {'nome': 'Calouro C', 'sexo': 'M', 'periodo': 2, 'exp_total': 0, 'podios_hist': 0, 'exp_superior': 0},
        ]
    },
    {
        "nome_cenario": "OLIP 2 - Técnico Experiente",
        "olimpiada": "OLIP",
        "categoria": "Técnico",
        "membros": [
            {'nome': 'Técnico A', 'sexo': 'M', 'periodo': 3, 'exp_total': 2, 'podios_hist': 1, 'exp_superior': 0},
            {'nome': 'Técnica B', 'sexo': 'F', 'periodo': 3, 'exp_total': 2, 'podios_hist': 0, 'exp_superior': 0},
            {'nome': 'Técnico C', 'sexo': 'M', 'periodo': 3, 'exp_total': 2, 'podios_hist': 0, 'exp_superior': 0},
        ]
    },
    {
        "nome_cenario": "OLIP 3 - Mista (Âncora + Novatos)",
        "olimpiada": "OLIP",
        "categoria": "Superior",
        "membros": [
            {'nome': 'Veterana A', 'sexo': 'F', 'periodo': 6, 'exp_total': 4, 'podios_hist': 2, 'exp_superior': 2},
            {'nome': 'Calouro B',  'sexo': 'M', 'periodo': 2, 'exp_total': 0, 'podios_hist': 0, 'exp_superior': 0},
            {'nome': 'Calouro C',  'sexo': 'M', 'periodo': 2, 'exp_total': 0, 'podios_hist': 0, 'exp_superior': 0},
        ]
    },
    {
        "nome_cenario": "OLIP 4 - Trio Consolidado (Superior)",
        "olimpiada": "OLIP",
        "categoria": "Superior",
        "membros": [
            {'nome': 'Veterano A', 'sexo': 'M', 'periodo': 8, 'exp_total': 5, 'podios_hist': 2, 'exp_superior': 3},
            {'nome': 'Veterana B', 'sexo': 'F', 'periodo': 6, 'exp_total': 4, 'podios_hist': 1, 'exp_superior': 2},
            {'nome': 'Veterano C', 'sexo': 'M', 'periodo': 6, 'exp_total': 4, 'podios_hist': 1, 'exp_superior': 2},
        ]
    },
    {
        "nome_cenario": "OLIP 5 - Trio Consolidado + Pódios Individuais",
        "olimpiada": "OLIP",
        "categoria": "Superior",
        "membros": [
            {'nome': 'Veterano A', 'sexo': 'M', 'periodo': 8, 'exp_total': 5, 'podios_hist': 2, 'podios_indiv': 2, 'exp_superior': 3},
            {'nome': 'Veterana B', 'sexo': 'F', 'periodo': 6, 'exp_total': 4, 'podios_hist': 1, 'podios_indiv': 1, 'exp_superior': 2},
            {'nome': 'Veterano C', 'sexo': 'M', 'periodo': 6, 'exp_total': 4, 'podios_hist': 1, 'podios_indiv': 0, 'exp_superior': 2},
        ]
    },

    # --- CENÁRIOS REGIONAL (SBC) ---
    {
        "nome_cenario": "Regional 1 - Estreantes na Regional",
        "olimpiada": "Regional",
        "categoria": "Superior",
        "membros": [
            {'nome': 'Estreante A', 'sexo': 'M', 'periodo': 4, 'exp_total': 1, 'podios_hist': 0, 'exp_superior': 0},
            {'nome': 'Estreante B', 'sexo': 'M', 'periodo': 4, 'exp_total': 1, 'podios_hist': 0, 'exp_superior': 0},
            {'nome': 'Estreante C', 'sexo': 'F', 'periodo': 4, 'exp_total': 1, 'podios_hist': 0, 'exp_superior': 0},
        ]
    },
    {
        "nome_cenario": "Regional 2 - Mista (Desequilibrada)",
        "olimpiada": "Regional",
        "categoria": "Superior",
        "membros": [
            {'nome': 'Veterana A', 'sexo': 'F', 'periodo': 6, 'exp_total': 5, 'podios_hist': 3, 'exp_superior': 3},
            {'nome': 'Calouro B',  'sexo': 'M', 'periodo': 2, 'exp_total': 0, 'podios_hist': 0, 'exp_superior': 0},
            {'nome': 'Calouro C',  'sexo': 'M', 'periodo': 2, 'exp_total': 0, 'podios_hist': 0, 'exp_superior': 0},
        ]
    },
    {
        "nome_cenario": "Regional 3 - Nivelada Intermediária",
        "olimpiada": "Regional",
        "categoria": "Superior",
        "membros": [
            {'nome': 'Intermediário A', 'sexo': 'M', 'periodo': 6, 'exp_total': 3, 'podios_hist': 1, 'exp_superior': 1},
            {'nome': 'Intermediária B', 'sexo': 'F', 'periodo': 5, 'exp_total': 2, 'podios_hist': 1, 'exp_superior': 1},
            {'nome': 'Intermediário C', 'sexo': 'M', 'periodo': 5, 'exp_total': 2, 'podios_hist': 0, 'exp_superior': 1},
        ]
    },
    {
        "nome_cenario": "Regional 4 - Trio Veterano de Elite",
        "olimpiada": "Regional",
        "categoria": "Superior",
        "membros": [
            {'nome': 'Veterano A',    'sexo': 'M', 'periodo': 8, 'exp_total': 7, 'podios_hist': 3, 'exp_superior': 4},
            {'nome': 'Veterano B',    'sexo': 'F', 'periodo': 7, 'exp_total': 6, 'podios_hist': 2, 'exp_superior': 3},
            {'nome': 'Intermediário', 'sexo': 'M', 'periodo': 5, 'exp_total': 3, 'podios_hist': 1, 'exp_superior': 1},
        ]
    },
    {
        "nome_cenario": "Regional 5 - Estreantes na Regional + Pódios Individuais",
        "olimpiada": "Regional",
        "categoria": "Superior",
        "membros": [
            {'nome': 'Estreante A', 'sexo': 'M', 'periodo': 4, 'exp_total': 1, 'podios_hist': 0, 'podios_indiv': 2, 'exp_superior': 0},
            {'nome': 'Estreante B', 'sexo': 'M', 'periodo': 4, 'exp_total': 1, 'podios_hist': 0, 'podios_indiv': 1, 'exp_superior': 0},
            {'nome': 'Estreante C', 'sexo': 'F', 'periodo': 4, 'exp_total': 1, 'podios_hist': 0, 'podios_indiv': 0, 'exp_superior': 0},
        ]
    }
]

print("\n" + "="*65)
print("RESULTADOS DAS SIMULAÇÕES (OLIP E REGIONAL - 2027)")
print("="*65)

for cenario in cenarios_simulacao:
    print(f"\n>>> {cenario['nome_cenario']}")
    simular_equipe(
        membros=cenario["membros"],
        olimpiada=cenario["olimpiada"],
        melhor_modelo=melhor_modelo,
        X=X,
        categoria=cenario["categoria"],
        ano=2027
    )


# ══════════════════════════════════════════════════════════════════════════════
# FINALIZANDO
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("CONCLUÍDO!")
print(f"{'='*80}")
print(f"\nArquivos gerados em: {PASTA_SAIDA}")
print("  • grafico_1_metricas.png")
print("  • grafico_2_matriz_confusao.png")
print("  • grafico_3_features.png")
print("  • grafico_4_roc.png")
print("  • grafico_5_distribuicao.png")
print("  • grafico_6_acuracia_olimpiada.png")
print("  • equipes_com_predicao.csv")
print(f"\nMelhor modelo: {melhor_nome}")
print(f"  F1-Score (CV): {resultados[melhor_nome]['F1-Score']:.4f}")
print(f"  Acurácia (CV): {resultados[melhor_nome]['Acurácia']:.4f}")
print(f"{'='*80}\n")
