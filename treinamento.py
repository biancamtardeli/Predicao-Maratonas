import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.utils import resample
from xgboost import XGBClassifier
from configuracoes import N_FOLDS, SEED, XGBOOST_DISPONIVEL

def preparar_dados(equipes_df):
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
    
    return X, y, X_bal, y_bal


def criar_modelos():
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
        
    return modelos


def treinar_modelos(X_bal, y_bal, modelos):
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
    
    return melhor_nome, resultados, melhor_modelo, modelos