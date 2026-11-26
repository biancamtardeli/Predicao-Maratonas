import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from configuracoes import N_FOLDS, SEED, XGBOOST_DISPONIVEL
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler

def preparar_dados(equipes_df):
    FEATURES = [
        'proporcao_feminino',
        'proporcao_superior',
        'periodo_medio',
        'periodo_max',
        'exp_media',
        'exp_max',
        'exp_min',
        'podios_indiv_soma',  
        'score_podios_soma',    
        'exp_superior_media',
        'membros_ja_venceram',
        'n_membros',
    ]

    # Codificar olimpíada como dummies (não ordinal)
    olimp_dummies = pd.get_dummies(equipes_df['olimpiada'], prefix='olimp', drop_first=True)
    X_base = equipes_df[FEATURES].fillna(0)
    X = pd.concat([X_base, olimp_dummies], axis=1)
    y = equipes_df['alto_desempenho'].astype(int)
    
    return X, y


def criar_modelos():
    modelos = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=2,
        random_state=SEED),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.05, max_depth=3,
        min_samples_leaf=4, subsample=0.8,
        random_state=SEED),
    }

    if XGBOOST_DISPONIVEL:
        modelos['XGBoost'] = XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            eval_metric='logloss',
            random_state=SEED, verbosity=0)
        
    return modelos


def treinar_modelos(X, y, modelos):
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    METRICAS_CV = ['accuracy', 'precision', 'recall', 'f1']

    resultados = {}
    pipelines = {}

    for nome, modelo in modelos.items():
        # O oversampling só acontece no fold de TREINO de cada rodada do CV
        pipe = ImbPipeline([
            ('oversampler', RandomOverSampler(random_state=SEED)),
            ('clf', modelo)
        ])
        pipelines[nome] = pipe

        scores = cross_validate(pipe, X, y,
                                cv=cv, scoring=METRICAS_CV,
                                return_train_score=False)
        resultados[nome] = {
            'Acurácia':      scores['test_accuracy'].mean(),
            'Precisão':      scores['test_precision'].mean(),
            'Sensibilidade': scores['test_recall'].mean(),
            'F1-Score':      scores['test_f1'].mean(),
            'Acurácia std':  scores['test_accuracy'].std(),
            'F1 std':        scores['test_f1'].std(),
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
    melhor_pipe = pipelines[melhor_nome]

    # Predições e probabilidades fora da amostra (Validação Cruzada)
    y_pred_cv = cross_val_predict(melhor_pipe, X, y, cv=cv)
    y_prob_cv = cross_val_predict(melhor_pipe, X, y, cv=cv, method='predict_proba')[:, 1]

    # Treina o pipeline vencedor na base completa (para o simulador e feature_importances_)
    melhor_pipe.fit(X, y)
    melhor_modelo = melhor_pipe.named_steps['clf']

    return melhor_nome, resultados, melhor_modelo, modelos, y_pred_cv, y_prob_cv