import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
)


def gerar_relatorio(melhor_modelo, melhor_nome, X, y):
    y_pred = melhor_modelo.predict(X)
    y_prob = melhor_modelo.predict_proba(X)[:, 1]

    print(
        f"\n  ── Relatório de Classificação ({melhor_nome}) "
        f"nos dados originais ──"
    )

    print(
        classification_report(
            y,
            y_pred,
            target_names=["Baixo desempenho", "Alto desempenho"],
        )
    )

    return y_pred, y_prob


def gerar_graficos(
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
    pasta_saida,
):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    fig.suptitle(
        f"Modelo Preditivo — Maratonas IFSULDEMINAS\n"
        f"Melhor modelo: {melhor_nome}",
        fontsize=14,
        fontweight="bold",
    )

    # 1. Comparação de métricas entre modelos
    ax = axes[0, 0]

    metricas_plot = [
        "Acurácia",
        "Precisão",
        "Sensibilidade",
        "F1-Score",
    ]

    x_pos = np.arange(len(metricas_plot))
    largura = 0.25
    cores_modelos = ["#2E86AB", "#A23B72", "#06A77D"]

    for i, (nome, res) in enumerate(resultados.items()):
        vals = [res[m] for m in metricas_plot]

        ax.bar(
            x_pos + i * largura,
            vals,
            largura,
            label=nome,
            color=cores_modelos[i],
            alpha=0.85,
        )

    ax.set_title(
        "Comparação de Métricas (Validação Cruzada)",
        fontweight="bold",
    )
    ax.set_xticks(x_pos + largura)
    ax.set_xticklabels(metricas_plot)
    ax.set_ylim([0, 1.05])
    ax.set_ylabel("Score")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # 2. Matriz de confusão
    ax = axes[0, 1]

    cm = confusion_matrix(y, y_pred)

    disp = ConfusionMatrixDisplay(
        cm,
        display_labels=[
            "Baixo\ndesempenho",
            "Alto\ndesempenho",
        ],
    )

    disp.plot(
        ax=ax,
        colorbar=False,
        cmap="Blues",
    )

    ax.set_title(
        f"Matriz de Confusão\n{melhor_nome}",
        fontweight="bold",
    )

    # 3. Importância das features
    ax = axes[0, 2]

    if hasattr(melhor_modelo, "feature_importances_"):
        imp = pd.Series(
            melhor_modelo.feature_importances_,
            index=X.columns,
        )

        top12 = imp.nlargest(12).sort_values()

        top12.plot(
            kind="barh",
            ax=ax,
            color="#4ECDC4",
            alpha=0.8,
        )

        ax.set_title(
            "Top 12 Features Mais Importantes",
            fontweight="bold",
        )

        ax.set_xlabel("Importância")
        ax.grid(True, alpha=0.3, axis="x")

    # 4. Curva ROC
    ax = axes[1, 0]

    for nome, modelo in modelos.items():
        modelo.fit(X_bal, y_bal)

        prob = modelo.predict_proba(X)[:, 1]

        fpr, tpr, _ = roc_curve(y, prob)
        auc = roc_auc_score(y, prob)

        ax.plot(
            fpr,
            tpr,
            label=f"{nome} (AUC={auc:.3f})",
            linewidth=2,
        )

    ax.plot(
        [0, 1],
        [0, 1],
        "k--",
        linewidth=1,
    )

    ax.set_title(
        "Curva ROC",
        fontweight="bold",
    )

    ax.set_xlabel("Taxa de Falso Positivo")
    ax.set_ylabel("Taxa de Verdadeiro Positivo")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5. Distribuição de probabilidade prevista
    ax = axes[1, 1]

    ax.hist(
        y_prob[y == 1],
        bins=15,
        alpha=0.7,
        color="#51CF66",
        label="Alto desempenho real",
        density=True,
    )

    ax.hist(
        y_prob[y == 0],
        bins=15,
        alpha=0.7,
        color="#FF6B6B",
        label="Baixo desempenho real",
        density=True,
    )

    ax.axvline(
        0.5,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Limiar 0.5",
    )

    ax.set_title(
        "Distribuição de Probabilidades Previstas",
        fontweight="bold",
    )

    ax.set_xlabel("Probabilidade de Alto Desempenho")
    ax.set_ylabel("Densidade")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. Acurácia por olimpíada
    ax = axes[1, 2]

    equipes_pred = equipes_df.copy()

    equipes_pred["pred"] = y_pred

    equipes_pred["correto"] = (
        equipes_pred["alto_desempenho"].astype(int)
        == equipes_pred["pred"]
    )

    acertos_olimp = (
        equipes_pred
        .groupby("olimpiada")["correto"]
        .agg(["mean", "count"])
    )

    acertos_olimp = acertos_olimp.sort_values(
        "mean",
        ascending=True,
    )

    bars = ax.barh(
        acertos_olimp.index,
        acertos_olimp["mean"],
        color="#FFB703",
        alpha=0.8,
    )

    for bar, (_, row) in zip(
        bars,
        acertos_olimp.iterrows(),
    ):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{row['mean']:.0%} (n={int(row['count'])})",
            va="center",
            fontsize=9,
        )

    ax.set_title(
        "Acurácia por Olimpíada",
        fontweight="bold",
    )

    ax.set_xlabel("Taxa de Acerto")
    ax.set_xlim([0, 1.2])
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    fig.savefig(
        pasta_saida / "modelo_preditivo_resultados.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "  ✓ Gráfico salvo: "
        "modelo_preditivo_resultados.png"
    )

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