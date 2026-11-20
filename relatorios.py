import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler
from configuracoes import N_FOLDS, SEED


def gerar_graficos(
    melhor_modelo,
    melhor_nome,
    modelos,
    resultados,
    X,
    y,
    equipes_df,
    y_pred,
    y_prob,
    pasta_saida,
):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
    
    # 1. Comparação de métricas entre modelos
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    metricas_plot = ["Acurácia", "Precisão", "Sensibilidade", "F1-Score"]
    x_pos = np.arange(len(metricas_plot))
    largura = 0.25
    cores_modelos = ["#2E86AB", "#A23B72", "#06A77D"]
    for i, (nome, res) in enumerate(resultados.items()):
        vals = [res[m] for m in metricas_plot]
        ax1.bar(x_pos + i * largura, vals, largura, label=nome, color=cores_modelos[i], alpha=0.85)
    ax1.set_title("Comparação de Métricas (Validação Cruzada)", fontweight="bold")
    ax1.set_xticks(x_pos + largura)
    ax1.set_xticklabels(metricas_plot)
    ax1.set_ylim([0, 1.05])
    ax1.set_ylabel("Score")
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig1.savefig(pasta_saida / "grafico_1_metricas.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # 2. Matriz de confusão
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Baixo\ndesempenho", "Alto\ndesempenho"])
    disp.plot(ax=ax2, colorbar=False, cmap="Blues")
    ax2.set_title(f"Matriz de Confusão ({melhor_nome})", fontweight="bold")
    plt.tight_layout()
    fig2.savefig(pasta_saida / "grafico_2_matriz_confusao.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    # 3. Importância das features
    if hasattr(melhor_modelo, "feature_importances_"):
        fig3, ax3 = plt.subplots(figsize=(7, 6))
        imp = pd.Series(melhor_modelo.feature_importances_, index=X.columns)
        top12 = imp.nlargest(12).sort_values()
        top12.plot(kind="barh", ax=ax3, color="#4ECDC4", alpha=0.8)
        ax3.set_title("Top 12 Atributos Mais Importantes", fontweight="bold")
        ax3.set_xlabel("Importância")
        ax3.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        fig3.savefig(pasta_saida / "grafico_3_features.png", dpi=300, bbox_inches="tight")
        plt.close(fig3)

    # 4. Curva ROC
    fig4, ax4 = plt.subplots(figsize=(7, 5))
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for nome, modelo in modelos.items():
        pipe = ImbPipeline([
            ('oversampler', RandomOverSampler(random_state=SEED)),
            ('clf', modelo)
        ])
        # Probabilidades fora da amostra (Validação Cruzada)
        y_prob_modelo = cross_val_predict(
            pipe, X, y, cv=cv, method='predict_proba'
        )[:, 1]

        fpr, tpr, _ = roc_curve(y, y_prob_modelo)
        auc_score = roc_auc_score(y, y_prob_modelo)
        ax4.plot(fpr, tpr, lw=2, label=f'{nome} (AUC={auc_score:.3f})')
    ax4.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax4.set_title("Curva ROC", fontweight="bold")
    ax4.set_xlabel("Taxa de Falso Positivo")
    ax4.set_ylabel("Taxa de Verdadeiro Positivo")
    ax4.legend(fontsize=9, loc="lower right")
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    fig4.savefig(pasta_saida / "grafico_4_roc.png", dpi=300, bbox_inches="tight")
    plt.close(fig4)

    # 5. Distribuição de probabilidade
    fig5, ax5 = plt.subplots(figsize=(7, 5))
    ax5.hist(y_prob[y == 1], bins=15, alpha=0.7, color="#51CF66", label="Alto desempenho real", density=True)
    ax5.hist(y_prob[y == 0], bins=15, alpha=0.7, color="#FF6B6B", label="Baixo desempenho real", density=True)
    ax5.axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="Limiar 0.5")
    ax5.set_title("Distribuição de Probabilidades Previstas", fontweight="bold")
    ax5.set_xlabel("Probabilidade de Alto Desempenho")
    ax5.set_ylabel("Densidade")
    ax5.legend(fontsize=9, loc="upper center")
    ax5.grid(True, alpha=0.3)
    plt.tight_layout()
    fig5.savefig(pasta_saida / "grafico_5_distribuicao.png", dpi=300, bbox_inches="tight")
    plt.close(fig5)

    # 6. Acurácia por olimpíada
    fig6, ax6 = plt.subplots(figsize=(7, 5))
    equipes_pred = equipes_df.copy()
    equipes_pred["pred"] = y_pred
    equipes_pred["correto"] = (equipes_pred["alto_desempenho"].astype(int) == equipes_pred["pred"])
    acertos_olimp = equipes_pred.groupby("olimpiada")["correto"].agg(["mean", "count"]).sort_values("mean", ascending=True)
    bars = ax6.barh(acertos_olimp.index, acertos_olimp["mean"], color="#FFB703", alpha=0.8)
    
    for bar, (_, row) in zip(bars, acertos_olimp.iterrows()):
        ax6.text(max(0, bar.get_width() - 0.15), bar.get_y() + bar.get_height() / 2, 
                 f"{row['mean']:.0%} (n={int(row['count'])})", va="center", color='black', fontsize=9, fontweight='bold' if row['mean']>0.5 else 'normal')
    
    ax6.set_title("Acurácia por Competição", fontweight="bold")
    ax6.set_xlabel("Taxa de Acerto")
    ax6.set_xlim([0, 1.05])
    ax6.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    fig6.savefig(pasta_saida / "grafico_6_acuracia_olimpiada.png", dpi=300, bbox_inches="tight")
    plt.close(fig6)

    print("  ✓ Gráficos separados salvos na pasta de saída.")
    return equipes_pred


def salvar_resultados(
    resultados,
    equipes_pred,
    pasta_saida,
):
    resultado_df = (
        pd.DataFrame(resultados)
        .T
        .reset_index()
    )

    resultado_df.columns = [
        "Modelo",
        "Acurácia",
        "Precisão",
        "Sensibilidade",
        "F1-Score",
        "Acurácia std",
        "F1 std",
    ]

    resultado_df.to_csv(
        pasta_saida / "comparacao_modelos.csv",
        index=False,
    )

    print(
        "  ✓ Comparação salva: "
        "comparacao_modelos.csv"
    )

    equipes_pred.to_csv(
        pasta_saida / "equipes_com_predicao.csv",
        index=False,
    )

    print(
        "  ✓ Predições salvas: "
        "equipes_com_predicao.csv"
    )