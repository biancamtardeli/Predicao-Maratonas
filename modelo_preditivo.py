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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, roc_auc_score, roc_curve)
from sklearn.utils import resample

try:
    from xgboost import XGBClassifier
    XGBOOST_DISPONIVEL = True
except ImportError:
    XGBOOST_DISPONIVEL = False
    print("⚠  XGBoost não instalado. Execute: pip install xgboost")

warnings.filterwarnings('ignore')

# ── Configurações ─────────────────────────────────────────────────────────────
PASTA_SAIDA = Path.cwd() / "saida_modelo"
PASTA_SAIDA.mkdir(exist_ok=True)

ARQUIVO_CSV = "dados_das_maratonas.csv"   # ← ajuste se necessário

SEED = 42
N_FOLDS = 5          # K-fold para validação cruzada
MIN_MEMBROS = 2      # equipes com menos membros são descartadas

print("=" * 80)
print("MODELO PREDITIVO — MARATONAS DE PROGRAMAÇÃO")
print("IFSULDEMINAS – Campus Muzambinho")
print("=" * 80)


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 1 — CARREGAMENTO E LIMPEZA
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/7] Carregando e limpando dados...")

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

print(f"  ✓ {len(df)} registros válidos | "
      f"{df['Nome'].nunique()} competidores únicos | "
      f"Olimpíadas: {df['Olimpíada'].unique().tolist()}")


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 2 — REGRAS DE MEDALHA (mesmas do analise_maratonas_local.py)
# ══════════════════════════════════════════════════════════════════════════════
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

df['podio'] = df.apply(eh_podio, axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 3 — ENGENHARIA DE ATRIBUTOS HISTÓRICOS (Feature Engineering)
#           Calculada "look-back": só usa dados ANTES do evento atual
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/7] Calculando atributos históricos por competidor...")

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
        passado['Olimpíada'].isin({'Regional', 'Mineira', 'Nacional'})
    ).sum())
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

print(f"  ✓ Atributos históricos calculados para {len(df)} registros")


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 4 — AGRUPAMENTO POR EQUIPE (linha = 1 equipe)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/7] Agregando registros por equipe...")

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
    exp_max             = g['exp_total'].max()       # membro mais veterano
    exp_min             = g['exp_total'].min()       # membro mais novato
    podios_soma         = g['podios_hist'].sum()     # pódios acumulados do trio
    exp_superior_media  = g['exp_superior'].mean()   # bagagem em fases avançadas
    membros_ja_venceram = g['ja_venceu'].sum()       # quantos já ganharam medalha

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
        'podios_soma':           podios_soma,
        'exp_superior_media':    exp_superior_media,
        'membros_ja_venceram':   membros_ja_venceram,
        'classificacao':         classificacao,
        'alto_desempenho':       alto_desempenho,
    })

equipes_df = (df_eq
              .groupby(['Ano', 'Olimpíada', 'Equipe'], sort=False)
              .apply(agregar_equipe, include_groups=False)
              .reset_index())

# Filtrar equipes com tamanho inválido e alvo indefinido
equipes_df = equipes_df[equipes_df['n_membros'] >= MIN_MEMBROS]
equipes_df = equipes_df.dropna(subset=['alto_desempenho'])

print(f"  ✓ {len(equipes_df)} equipes únicas | "
      f"Alto desempenho: {int(equipes_df['alto_desempenho'].sum())} "
      f"({equipes_df['alto_desempenho'].mean()*100:.1f}%) | "
      f"Baixo desempenho: {int((equipes_df['alto_desempenho']==0).sum())}")


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 5 — PREPARAÇÃO PARA O MODELO
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/7] Preparando features e tratando desbalanceamento...")

FEATURES = [
    'proporcao_feminino',
    'proporcao_superior',
    'periodo_medio',
    'periodo_max',
    'exp_media',
    'exp_max',
    'exp_min',
    'podios_soma',
    'exp_superior_media',
    'membros_ja_venceram',
    'n_membros',
]

# Codificar olimpíada como dummies (não ordinal)
olimp_dummies = pd.get_dummies(equipes_df['olimpiada'], prefix='olimp', drop_first=True)
X_base = equipes_df[FEATURES].fillna(0)
X = pd.concat([X_base, olimp_dummies], axis=1)
y = equipes_df['alto_desempenho'].astype(int)

# ── Balanceamento por oversampling da classe minoritária ──────────────────
X_y = pd.concat([X, y.rename('target')], axis=1)
classe_maior  = X_y[X_y['target'] == X_y['target'].value_counts().idxmax()]
classe_menor  = X_y[X_y['target'] == X_y['target'].value_counts().idxmin()]

classe_menor_up = resample(classe_menor,
                            replace=True,
                            n_samples=len(classe_maior),
                            random_state=SEED)
X_y_bal = pd.concat([classe_maior, classe_menor_up]).sample(frac=1, random_state=SEED)
X_bal   = X_y_bal.drop(columns='target')
y_bal   = X_y_bal['target']

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

modelos = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=2,
        class_weight='balanced', random_state=SEED),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        random_state=SEED),
}

if XGBOOST_DISPONIVEL:
    modelos['XGBoost'] = XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        use_label_encoder=False, eval_metric='logloss',
        random_state=SEED, verbosity=0)

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
METRICAS_CV = ['accuracy', 'precision', 'recall', 'f1']

resultados = {}
for nome, modelo in modelos.items():
    scores = cross_validate(modelo, X_bal, y_bal,
                            cv=cv, scoring=METRICAS_CV,
                            return_train_score=False)
    resultados[nome] = {
        'Acurácia':     scores['test_accuracy'].mean(),
        'Precisão':     scores['test_precision'].mean(),
        'Sensibilidade':scores['test_recall'].mean(),
        'F1-Score':     scores['test_f1'].mean(),
        'Acurácia std': scores['test_accuracy'].std(),
        'F1 std':       scores['test_f1'].std(),
    }

    print(f"\n  ── {nome} ──────────────────────────────")
    print(f"     Acurácia    : {resultados[nome]['Acurácia']:.4f} "
          f"(±{resultados[nome]['Acurácia std']:.4f})")
    print(f"     Precisão    : {resultados[nome]['Precisão']:.4f}")
    print(f"     Sensibilidade: {resultados[nome]['Sensibilidade']:.4f}")
    print(f"     F1-Score    : {resultados[nome]['F1-Score']:.4f} "
          f"(±{resultados[nome]['F1 std']:.4f})")

# Identificar o melhor modelo pelo F1
melhor_nome = max(resultados, key=lambda k: resultados[k]['F1-Score'])
melhor_modelo = modelos[melhor_nome]
melhor_modelo.fit(X_bal, y_bal)   # Treinar no dataset completo balanceado
print(f"\n  🏆 Melhor modelo: {melhor_nome} "
      f"(F1={resultados[melhor_nome]['F1-Score']:.4f})")


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 7 — RELATÓRIO DETALHADO NO CONJUNTO ORIGINAL (sem balanceamento)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[6/7] Gerando relatório e gráficos...")

y_pred = melhor_modelo.predict(X)
y_prob = melhor_modelo.predict_proba(X)[:, 1]

print(f"\n  ── Relatório de Classificação ({melhor_nome}) nos dados originais ──")
print(classification_report(y, y_pred,
                             target_names=['Baixo desempenho', 'Alto desempenho']))

# ── Gráficos ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(f'Modelo Preditivo — Maratonas IFSULDEMINAS\n'
             f'Melhor modelo: {melhor_nome}', fontsize=14, fontweight='bold')

# 1. Comparação de métricas entre modelos
ax = axes[0, 0]
metricas_plot = ['Acurácia', 'Precisão', 'Sensibilidade', 'F1-Score']
x_pos = np.arange(len(metricas_plot))
largura = 0.25
cores_modelos = ['#2E86AB', '#A23B72', '#06A77D']
for i, (nome, res) in enumerate(resultados.items()):
    vals = [res[m] for m in metricas_plot]
    ax.bar(x_pos + i * largura, vals, largura,
           label=nome, color=cores_modelos[i], alpha=0.85)
ax.set_title('Comparação de Métricas (Validação Cruzada)', fontweight='bold')
ax.set_xticks(x_pos + largura)
ax.set_xticklabels(metricas_plot)
ax.set_ylim([0, 1.05])
ax.set_ylabel('Score')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# 2. Matriz de confusão
ax = axes[0, 1]
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(cm,
    display_labels=['Baixo\ndesempenho', 'Alto\ndesempenho'])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f'Matriz de Confusão\n{melhor_nome}', fontweight='bold')

# 3. Importância das features (top 12)
ax = axes[0, 2]
if hasattr(melhor_modelo, 'feature_importances_'):
    imp = pd.Series(melhor_modelo.feature_importances_, index=X.columns)
    top12 = imp.nlargest(12).sort_values()
    top12.plot(kind='barh', ax=ax, color='#4ECDC4', alpha=0.8)
    ax.set_title('Top 12 Features Mais Importantes', fontweight='bold')
    ax.set_xlabel('Importância')
    ax.grid(True, alpha=0.3, axis='x')

# 4. Curva ROC
ax = axes[1, 0]
for nome, modelo in modelos.items():
    modelo.fit(X_bal, y_bal)
    prob = modelo.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, prob)
    auc = roc_auc_score(y, prob)
    ax.plot(fpr, tpr, label=f'{nome} (AUC={auc:.3f})', linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
ax.set_title('Curva ROC', fontweight='bold')
ax.set_xlabel('Taxa de Falso Positivo')
ax.set_ylabel('Taxa de Verdadeiro Positivo')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 5. Distribuição de probabilidade prevista
ax = axes[1, 1]
ax.hist(y_prob[y == 1], bins=15, alpha=0.7, color='#51CF66',
        label='Alto desempenho real', density=True)
ax.hist(y_prob[y == 0], bins=15, alpha=0.7, color='#FF6B6B',
        label='Baixo desempenho real', density=True)
ax.axvline(0.5, color='black', linestyle='--', linewidth=1.5, label='Limiar 0.5')
ax.set_title('Distribuição de Probabilidades Previstas', fontweight='bold')
ax.set_xlabel('Probabilidade de Alto Desempenho')
ax.set_ylabel('Densidade')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 6. F1-Score por olimpíada
ax = axes[1, 2]
equipes_pred = equipes_df.copy()
equipes_pred['pred'] = y_pred
equipes_pred['correto'] = (equipes_pred['alto_desempenho'].astype(int) ==
                            equipes_pred['pred'])
acertos_olimp = equipes_pred.groupby('olimpiada')['correto'].agg(['mean', 'count'])
acertos_olimp = acertos_olimp.sort_values('mean', ascending=True)
bars = ax.barh(acertos_olimp.index, acertos_olimp['mean'],
               color='#FFB703', alpha=0.8)
for bar, (_, row) in zip(bars, acertos_olimp.iterrows()):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f"{row['mean']:.0%} (n={int(row['count'])})",
            va='center', fontsize=9)
ax.set_title('Acurácia por Olimpíada', fontweight='bold')
ax.set_xlabel('Taxa de Acerto')
ax.set_xlim([0, 1.2])
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
fig.savefig(PASTA_SAIDA / 'modelo_preditivo_resultados.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Gráfico salvo: modelo_preditivo_resultados.png")

# ── Salvar tabela de resultados ────────────────────────────────────────────
resultado_df = pd.DataFrame(resultados).T.reset_index()
resultado_df.columns = ['Modelo', 'Acurácia', 'Precisão',
                         'Sensibilidade', 'F1-Score', 'Acurácia std', 'F1 std']
resultado_df.to_csv(PASTA_SAIDA / 'comparacao_modelos.csv', index=False)
print("  ✓ Comparação salva: comparacao_modelos.csv")

# ── Salvar equipes com predição ────────────────────────────────────────────
equipes_pred['probabilidade_alto'] = y_prob
equipes_pred.to_csv(PASTA_SAIDA / 'equipes_com_predicao.csv', index=False)
print("  ✓ Predições salvas: equipes_com_predicao.csv")


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 8 — SIMULADOR INTERATIVO DE EQUIPES
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[7/7] Gerando simulador de equipes...")


def simular_equipe(
    membros: list[dict],
    olimpiada: str,
    categoria: str = 'Superior',
    ano: int = 2026,
    verbose: bool = True
) -> dict:
    """
    Estima a probabilidade de alto desempenho de uma nova equipe.

    Parâmetros
    ----------
    membros : lista de dicionários, 1 por integrante. Chaves:
        - nome       : str  (usado apenas para display)
        - sexo       : 'M' ou 'F'
        - periodo    : int  (semestre atual, ex: 4)
        - exp_total  : int  (quantas maratonas já participou)
        - podios_hist: int  (quantos pódios já ganhou)
        - exp_superior: int (participações em Regional/Mineira/Nacional)
    olimpiada : 'OLIP' | 'Regional' | 'Mineira' | 'Nacional' | ...
    categoria : 'Superior' | 'Técnico'
    ano       : ano do evento (afeta o limite de medalhas)
    verbose   : imprime resultado formatado
    """
    n = len(membros)
    if n < 2:
        raise ValueError("Uma equipe precisa de pelo menos 2 membros.")

    # Agregar atributos dos membros
    feminino      = sum(1 for m in membros if m.get('sexo', 'M') == 'F') / n
    sup_prop      = 1.0 if categoria == 'Superior' else 0.0
    periodo_med   = np.mean([m.get('periodo', 3) for m in membros])
    periodo_max_v = max(m.get('periodo', 3) for m in membros)
    exp_med       = np.mean([m.get('exp_total', 0) for m in membros])
    exp_max_v     = max(m.get('exp_total', 0) for m in membros)
    exp_min_v     = min(m.get('exp_total', 0) for m in membros)
    podios        = sum(m.get('podios_hist', 0) for m in membros)
    exp_sup_med   = np.mean([m.get('exp_superior', 0) for m in membros])
    ja_venc       = sum(1 for m in membros if m.get('podios_hist', 0) > 0)

    # Montar vetor de features na mesma ordem do treino
    features_base = {
        'proporcao_feminino':  feminino,
        'proporcao_superior':  sup_prop,
        'periodo_medio':       periodo_med,
        'periodo_max':         periodo_max_v,
        'exp_media':           exp_med,
        'exp_max':             exp_max_v,
        'exp_min':             exp_min_v,
        'podios_soma':         podios,
        'exp_superior_media':  exp_sup_med,
        'membros_ja_venceram': ja_venc,
        'n_membros':           n,
    }

    # Dummies de olimpíada
    olimp_cols = {c: 0 for c in X.columns if c.startswith('olimp_')}
    chave = f'olimp_{olimpiada}'
    if chave in olimp_cols:
        olimp_cols[chave] = 1

    vetor = {**features_base, **olimp_cols}
    X_novo = pd.DataFrame([vetor])[X.columns]

    prob = melhor_modelo.predict_proba(X_novo)[0, 1]
    pred = int(prob >= 0.5)
    lim  = limite_medalha(olimpiada, categoria, ano)

    if verbose:
        print("\n" + "─" * 60)
        print("  SIMULAÇÃO DE EQUIPE")
        print("─" * 60)
        for m in membros:
            print(f"  • {m.get('nome','?')} | "
                  f"Sexo: {m.get('sexo','?')} | "
                  f"Período: {m.get('periodo','?')} | "
                  f"Maratonas: {m.get('exp_total',0)} | "
                  f"Pódios: {m.get('podios_hist',0)}")
        print(f"\n  Olimpíada : {olimpiada} ({categoria}, {ano})")
        print(f"  Corte     : top {lim} = alto desempenho" if lim > 0
              else f"  ⚠  {categoria} não distribui medalhas na {olimpiada}")
        print(f"\n  {'🟢 ALTO DESEMPENHO' if pred else '🔴 BAIXO DESEMPENHO'}")
        print(f"  Probabilidade de alto desempenho: {prob:.1%}")

        if prob >= 0.75:
            print("  Avaliação: ⭐⭐⭐ Equipe muito promissora")
        elif prob >= 0.5:
            print("  Avaliação: ⭐⭐  Equipe com boas chances")
        elif prob >= 0.25:
            print("  Avaliação: ⭐   Equipe com chances limitadas")
        else:
            print("  Avaliação: ❌   Perfil desfavorável para esta olimpíada")
        print("─" * 60)

    return {
        'probabilidade': round(float(prob), 4),
        'predicao': 'Alto desempenho' if pred else 'Baixo desempenho',
        'limite_medalha': lim,
    }


# ── Exemplos de simulação ──────────────────────────────────────────────────
print("\n  ── Exemplos de uso do simulador ────────────────────────────")

# Exemplo 1: equipe experiente na OLIP Superior
simular_equipe(
    membros=[
        {'nome': 'Aluno A', 'sexo': 'M', 'periodo': 6,
         'exp_total': 5, 'podios_hist': 2, 'exp_superior': 3},
        {'nome': 'Aluno B', 'sexo': 'M', 'periodo': 5,
         'exp_total': 4, 'podios_hist': 1, 'exp_superior': 2},
        {'nome': 'Aluno C', 'sexo': 'F', 'periodo': 6,
         'exp_total': 3, 'podios_hist': 1, 'exp_superior': 1},
    ],
    olimpiada='OLIP', categoria='Superior', ano=2026
)

# Exemplo 2: equipe novata na OLIP Superior
simular_equipe(
    membros=[
        {'nome': 'Calouro A', 'sexo': 'M', 'periodo': 1,
         'exp_total': 0, 'podios_hist': 0, 'exp_superior': 0},
        {'nome': 'Calouro B', 'sexo': 'M', 'periodo': 1,
         'exp_total': 0, 'podios_hist': 0, 'exp_superior': 0},
        {'nome': 'Calouro C', 'sexo': 'M', 'periodo': 2,
         'exp_total': 1, 'podios_hist': 0, 'exp_superior': 0},
    ],
    olimpiada='Regional', categoria='Superior', ano=2026
)

# Exemplo 3: equipe veterana na Regional Superior
simular_equipe(
    membros=[
        {'nome': 'Veterano A', 'sexo': 'M', 'periodo': 8,
         'exp_total': 7, 'podios_hist': 3, 'exp_superior': 4},
        {'nome': 'Veterano B', 'sexo': 'F', 'periodo': 7,
         'exp_total': 6, 'podios_hist': 2, 'exp_superior': 3},
        {'nome': 'Intermediário', 'sexo': 'M', 'periodo': 5,
         'exp_total': 3, 'podios_hist': 1, 'exp_superior': 1},
    ],
    olimpiada='Regional', categoria='Superior', ano=2026
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
