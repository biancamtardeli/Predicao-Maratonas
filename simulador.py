import pandas as pd
import numpy as np

from regras import limite_medalha


def simular_equipe(
    membros: list[dict],
    olimpiada: str,
    melhor_modelo,
    X,
    categoria: str = 'Superior',
    ano: int = 2026,
    verbose: bool = True
) -> dict:
    """
    Estima a probabilidade de alto desempenho de uma nova equipe.

    Parâmetros
    ----------
    membros : lista de dicionários, 1 por integrante. Chaves:
        - nome          : str  (usado apenas para display)
        - sexo          : 'M' ou 'F'
        - periodo       : int  (semestre atual, ex: 4)
        - exp_total     : int  (quantas maratonas já participou)
        - podios_hist   : int  (quantos pódios já ganhou)
        - exp_superior  : int  (participações em Regional/Mineira/Nacional)

    olimpiada : 'OLIP' | 'Regional' | 'Mineira' | 'Nacional' | ...

    melhor_modelo : modelo de machine learning já treinado.

    X : DataFrame utilizado no treinamento, usado para manter
        a mesma ordem e as mesmas colunas das features.

    categoria : 'Superior' | 'Técnico'

    ano : ano do evento (afeta o limite de medalhas)

    verbose : imprime resultado formatado
    """

    n = len(membros)

    if n < 2:
        raise ValueError("Uma equipe precisa de pelo menos 2 membros.")

    # ── Agregar atributos dos membros ─────────────────────────────────────

    feminino = sum(
        1 for m in membros
        if m.get('sexo', 'M') == 'F'
    ) / n

    sup_prop = 1.0 if categoria == 'Superior' else 0.0

    periodo_med = np.mean([
        m.get('periodo', 3)
        for m in membros
    ])

    periodo_max_v = max(
        m.get('periodo', 3)
        for m in membros
    )

    exp_med = np.mean([
        m.get('exp_total', 0)
        for m in membros
    ])

    exp_max_v = max(
        m.get('exp_total', 0)
        for m in membros
    )

    exp_min_v = min(
        m.get('exp_total', 0)
        for m in membros
    )

    # Mantemos 'podios_hist' como legado para os pódios em equipe
    podios_equipe = sum(m.get('podios_hist', 0) for m in membros)
    
    # Nova chave para pódios individuais (se não for passada, assume 0)
    podios_indiv_soma = sum(m.get('podios_indiv', 0) for m in membros)
    
    # Nosso novo cálculo de peso (peso 2 para o individual)
    score_podios_soma = podios_equipe + (podios_indiv_soma * 2)

    exp_sup_med = np.mean([
        m.get('exp_superior', 0)
        for m in membros
    ])

    ja_venc = sum(
        1 for m in membros
        if m.get('podios_hist', 0) > 0 or m.get('podios_indiv', 0) > 0
    )

    # ── Montar vetor de features na mesma ordem do treino ────────────────

    features_base = {
        'proporcao_feminino': feminino,
        'proporcao_superior': sup_prop,
        'periodo_medio': periodo_med,
        'periodo_max': periodo_max_v,
        'exp_media': exp_med,
        'exp_max': exp_max_v,
        'exp_min': exp_min_v,
        'podios_indiv_soma': podios_indiv_soma,  # <-- CHAVE NOVA
        'score_podios_soma': score_podios_soma,  # <-- CHAVE NOVA
        'exp_superior_media': exp_sup_med,
        'membros_ja_venceram': ja_venc,
        'n_membros': n,
    }

    # ── Dummies de olimpíada ──────────────────────────────────────────────

    olimp_cols = {
        c: 0
        for c in X.columns
        if c.startswith('olimp_')
    }

    chave = f'olimp_{olimpiada}'

    if chave in olimp_cols:
        olimp_cols[chave] = 1

    vetor = {
        **features_base,
        **olimp_cols
    }

    X_novo = pd.DataFrame([vetor])[X.columns]

    # ── Fazer previsão ───────────────────────────────────────────────────

    prob = melhor_modelo.predict_proba(X_novo)[0, 1]

    pred = int(prob >= 0.5)

    # ── Limite de medalha ────────────────────────────────────────────────

    lim = limite_medalha(
        olimpiada,
        categoria,
        ano
    )

    # ── Exibir resultado ─────────────────────────────────────────────────

    if verbose:
        print("\n" + "─" * 60)
        print("  SIMULAÇÃO DE EQUIPE")
        print("─" * 60)

        for m in membros:
            print(
                f"  • {m.get('nome', '?')} | "
                f"Sexo: {m.get('sexo', '?')} | "
                f"Período: {m.get('periodo', '?')} | "
                f"Maratonas: {m.get('exp_total', 0)} | "
                f"Pódios: {m.get('podios_hist', 0)}"
            )

        print(
            f"\n  Olimpíada : {olimpiada} "
            f"({categoria}, {ano})"
        )

        if lim > 0:
            print(
                f"  Corte     : top {lim} = alto desempenho"
            )
        else:
            print(
                f"  ⚠  {categoria} não distribui medalhas "
                f"na {olimpiada}"
            )

        if pred:
            print("\n  🟢 ALTO DESEMPENHO")
        else:
            print("\n  🔴 BAIXO DESEMPENHO")

        print(
            f"  Probabilidade de alto desempenho: "
            f"{prob:.1%}"
        )

        if prob >= 0.75:
            print(
                "  Avaliação: ⭐⭐⭐ Equipe muito promissora"
            )
        elif prob >= 0.5:
            print(
                "  Avaliação: ⭐⭐  Equipe com boas chances"
            )
        elif prob >= 0.25:
            print(
                "  Avaliação: ⭐   Equipe com chances limitadas"
            )
        else:
            print(
                "  Avaliação: ❌   Perfil desfavorável "
                "para esta olimpíada"
            )

        print("─" * 60)

    return {
        'probabilidade': round(float(prob), 4),
        'predicao': (
            'Alto desempenho'
            if pred
            else 'Baixo desempenho'
        ),
        'limite_medalha': lim,
    }