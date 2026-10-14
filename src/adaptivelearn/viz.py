"""Matplotlib figures used by the report and presentation."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .config import FIG_DIR, LEVELS, TOPIC_COLS, TOPICS

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = sns.color_palette("deep")


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)


def distributions(df):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.flat, TOPIC_COLS + ["final_score"]):
        sns.histplot(df[col].dropna(), bins=25, ax=ax, color=PALETTE[0])
        ax.set_title(col.replace("score_", "").replace("_", " ").title()
                     if col != "final_score" else "Final score (target)")
        ax.set_xlabel("score")
    counts = df["level"].value_counts().reindex(LEVELS)
    axes.flat[7].bar(LEVELS, counts.values, color=PALETTE[1:4])
    axes.flat[7].set_title("Level (classification target)")
    _save(fig, "01_distributions.png")


def missing(df, cols):
    pct = df[cols].isna().mean().sort_values(ascending=False) * 100
    pct = pct[pct > 0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(x=pct.values, y=pct.index, ax=ax, color=PALETTE[2])
    ax.set_xlabel("% missing")
    ax.set_title("Missing values by feature (before imputation)")
    _save(fig, "02_missing_values.png")


def correlation(df, cols):
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="vlag", center=0, annot=False, ax=ax, square=True)
    ax.set_title("Feature correlation (Pearson)")
    _save(fig, "03_correlation.png")


def cluster_selection(sweep, best_k):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(sweep.k, sweep.inertia, "o-")
    axes[0].set_title("Elbow (inertia)")
    axes[1].plot(sweep.k, sweep.silhouette, "o-", color=PALETTE[1])
    axes[1].set_title("Silhouette (higher is better)")
    axes[2].plot(sweep.k, sweep.davies_bouldin, "o-", color=PALETTE[3])
    axes[2].set_title("Davies-Bouldin (lower is better)")
    for ax in axes:
        ax.axvline(best_k, ls="--", color="grey")
        ax.set_xlabel("k")
    _save(fig, "04_cluster_selection.png")


def clusters_pca(emb, labels, personas, explained):
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in sorted(set(labels)):
        m = labels == c
        ax.scatter(emb[m, 0], emb[m, 1], s=12, alpha=0.6, label=personas[c], color=PALETTE[c % 10])
    ax.set_xlabel(f"PC1 ({explained[0]:.0%} variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.0%} variance)")
    ax.set_title("Learner personas (K-Means, PCA projection)")
    ax.legend(markerscale=2)
    _save(fig, "05_cluster_pca.png")


def persona_heatmap(profile):
    cols = [c for c in profile.columns if c not in ("persona", "size")]
    z = (profile[cols] - profile[cols].mean()) / profile[cols].std(ddof=0).replace(0, 1)
    fig, ax = plt.subplots(figsize=(13, 0.9 * len(profile) + 2))
    sns.heatmap(z, annot=profile[cols].round(1), fmt="", cmap="vlag", center=0, ax=ax,
                yticklabels=[f"{p} (n={n})" for p, n in zip(profile.persona, profile["size"])])
    ax.set_title("Persona profiles (colour = z-score across clusters, text = cluster mean)")
    _save(fig, "06_persona_profiles.png")


def model_bars(table, metrics, title, fname, labels=None):
    t = table.set_index("model")[metrics]
    fig, ax = plt.subplots(figsize=(10, 5))
    t.plot.barh(ax=ax, width=0.8)
    ax.set_title(title)
    ax.invert_yaxis()
    ax.legend(labels or metrics, loc="lower right")
    _save(fig, fname)


def confusion(cm, title):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LEVELS, yticklabels=LEVELS, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    _save(fig, "08_confusion_matrix.png")


def pred_vs_actual(y, yhat, title):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(y, yhat, s=10, alpha=0.5)
    lim = [min(y.min(), yhat.min()), max(y.max(), yhat.max())]
    axes[0].plot(lim, lim, "k--", lw=1)
    axes[0].set_xlabel("Actual final score")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title(title)
    sns.histplot(yhat - y, bins=30, kde=True, ax=axes[1], color=PALETTE[2])
    axes[1].set_title("Residuals (predicted - actual)")
    _save(fig, "10_regression_fit.png")


def importance(imp, top=15):
    t = imp.head(top)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(t.feature[::-1], t.importance[::-1], xerr=t["std"][::-1], color=PALETTE[0])
    ax.set_xlabel("Permutation importance (drop in macro-F1)")
    ax.set_title("What drives the level prediction?")
    _save(fig, "09_feature_importance.png")


def recommender_bars(table):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    t = table.set_index("model")
    t[["precision@5", "recall@5", "ndcg@5", "hit_rate@5"]].plot.barh(ax=axes[0], width=0.8)
    axes[0].set_title("Ranking quality @5 (higher is better)")
    axes[0].invert_yaxis()
    t[["rmse", "mae"]].plot.barh(ax=axes[1], width=0.7, color=[PALETTE[3], PALETTE[4]])
    axes[1].set_title("Rating error (lower is better)")
    axes[1].invert_yaxis()
    axes[1].set_yticklabels([])
    axes[1].set_ylabel("")
    _save(fig, "11_recommender_comparison.png")


def fairness(fair):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(len(fair))
    ax.bar(x - 0.2, fair.accuracy, 0.4, label="accuracy")
    ax.bar(x + 0.2, fair.f1_macro, 0.4, label="macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\n(n={n})" for a, n in zip(fair.internet_access, fair.n)])
    ax.set_ylim(0, 1)
    ax.set_title("Level prediction by internet-access group (SDG 10 check)")
    ax.legend()
    _save(fig, "12_fairness.png")
