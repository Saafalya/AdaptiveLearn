"""Matplotlib figures used by the report and presentation."""
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .config import FIG_DIR, RESULT_ORDER

sns.set_theme(style="whitegrid", context="notebook")
PAL = sns.color_palette("deep")
OUTCOME_COLORS = {"Distinction": PAL[2], "Pass": PAL[0], "Fail": PAL[1], "Withdrawn": PAL[3]}


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)


def outcomes(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    counts = df["final_result"].value_counts().reindex(RESULT_ORDER)
    axes[0].bar(counts.index, counts.values, color=[OUTCOME_COLORS[c] for c in counts.index])
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, f"{v / counts.sum():.0%}", ha="center", va="bottom")
    axes[0].set_title("Final result (classification target)")
    tmp = df.assign(log_clicks=np.log1p(df["total_clicks"]))
    sns.boxplot(data=tmp, x="final_result", y="log_clicks", order=RESULT_ORDER,
                hue="final_result", palette=OUTCOME_COLORS, legend=False, ax=axes[1])
    axes[1].set_title("Clicks by day 90 (log scale) vs outcome")
    axes[1].set_ylabel("log(1 + clicks)")
    sns.boxplot(data=df, x="final_result", y="mean_score_early", order=RESULT_ORDER,
                hue="final_result", palette=OUTCOME_COLORS, legend=False, ax=axes[2])
    axes[2].set_title("Mean early assessment score vs outcome")
    _save(fig, "01_outcomes_vs_engagement.png")


def missing(df, cols):
    pct = df[cols].isna().mean().sort_values(ascending=False) * 100
    pct = pct[pct > 0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(x=pct.values, y=pct.index, ax=ax, color=PAL[2])
    ax.set_xlabel("% missing (day-90 table, before imputation)")
    ax.set_title("Missing values by feature")
    _save(fig, "02_missing_values.png")


def correlation(df, cols):
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(df[cols].corr(), cmap="vlag", center=0, ax=ax, square=True)
    ax.set_title("Feature correlation (Pearson)")
    _save(fig, "03_correlation.png")


def cutoff_sweep(sw):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(sw.cutoff_day, sw.f1_macro, "o-", label="macro-F1 (4 classes)")
    ax.plot(sw.cutoff_day, sw.at_risk_auc, "s-", label="at-risk ROC-AUC")
    ax.plot(sw.cutoff_day, sw.accuracy, "^-", label="accuracy")
    ax.set_xlabel("Cut-off day (data available up to this day)")
    ax.set_ylim(0, 1)
    ax.set_title("How early can risk be detected?")
    ax.legend()
    _save(fig, "04_early_prediction.png")


def cluster_selection(sweep, best_k):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(sweep.k, sweep.inertia, "o-")
    axes[0].set_title("Elbow (inertia)")
    axes[1].plot(sweep.k, sweep.silhouette, "o-", color=PAL[1])
    axes[1].set_title("Silhouette (higher is better)")
    axes[2].plot(sweep.k, sweep.davies_bouldin, "o-", color=PAL[3])
    axes[2].set_title("Davies-Bouldin (lower is better)")
    for ax in axes:
        ax.axvline(best_k, ls="--", color="grey")
        ax.set_xlabel("k")
    _save(fig, "05_cluster_selection.png")


def clusters_pca(emb, labels, personas, explained):
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in sorted(set(labels)):
        m = labels == c
        ax.scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.4, label=f"{personas[c]} (n={m.sum()})",
                   color=PAL[c % 10])
    ax.set_xlabel(f"PC1 ({explained[0]:.0%} variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.0%} variance)")
    ax.set_title("Learner personas (K-Means, PCA projection)")
    ax.legend(markerscale=3)
    _save(fig, "06_cluster_pca.png")


def persona_outcomes(outcome_mix, profile):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = [f"{p}\n(n={n})" for p, n in zip(profile.persona, profile["size"])]
    bottom = np.zeros(len(outcome_mix))
    for r in RESULT_ORDER:
        ax.bar(labels, outcome_mix[r].values, bottom=bottom, label=r, color=OUTCOME_COLORS[r])
        bottom += outcome_mix[r].values
    ax.set_ylabel("share of students")
    ax.set_title("Real outcomes by persona (personas never saw the outcome)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    _save(fig, "07_persona_outcomes.png")


def model_bars(table, title, fname, metrics, labels):
    t = table.set_index("model")[metrics]
    fig, ax = plt.subplots(figsize=(10, 5))
    t.plot.barh(ax=ax, width=0.8)
    ax.set_title(title)
    ax.invert_yaxis()
    ax.legend(labels, loc="lower right")
    _save(fig, fname)


def confusion(cm, title):
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.2, 5))
    sns.heatmap(cmn, annot=np.array([[f"{cmn[i, j]:.0%}\n({cm[i, j]})" for j in range(4)]
                                     for i in range(4)]),
                fmt="", cmap="Blues", xticklabels=RESULT_ORDER, yticklabels=RESULT_ORDER, ax=ax,
                vmin=0, vmax=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    _save(fig, "09_confusion_matrix.png")


def importance(imp, top=15):
    t = imp.head(top)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(t.feature[::-1], t.importance[::-1], xerr=t["std"][::-1], color=PAL[0])
    ax.set_xlabel("Permutation importance (drop in macro-F1)")
    ax.set_title("What drives the outcome prediction?")
    _save(fig, "10_feature_importance.png")


def ablation(abl):
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    t = abl.set_index("feature_group")[["f1_macro", "at_risk_auc"]]
    t.plot.barh(ax=ax, width=0.7)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_title("Which data matters? (Hist. Gradient Boosting, test set)")
    ax.legend(["macro-F1", "at-risk ROC-AUC"], loc="lower right")
    _save(fig, "11_ablation.png")


def regression_fit(y, yhat, title):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(y, yhat, s=5, alpha=0.3)
    axes[0].plot([0, 100], [0, 100], "k--", lw=1)
    axes[0].set_xlabel("Actual mean score (future assessments)")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title(title)
    sns.histplot(yhat - y, bins=40, kde=True, ax=axes[1], color=PAL[2])
    axes[1].set_title("Residuals (predicted - actual)")
    _save(fig, "12_regression_fit.png")


def recommender_bars(table):
    t = table.set_index("model")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    t[["precision@10", "recall@10", "ndcg@10", "hit_rate@10"]].plot.barh(ax=axes[0], width=0.8)
    axes[0].invert_yaxis()
    axes[0].set_title("Next-resource ranking quality @10 (temporal hold-out)")
    t[["coverage"]].plot.barh(ax=axes[1], width=0.6, color=PAL[4], legend=False)
    axes[1].invert_yaxis()
    axes[1].set_yticklabels([])
    axes[1].set_ylabel("")
    axes[1].set_title("Catalogue coverage of top-10 lists")
    _save(fig, "13_recommender_comparison.png")


def fairness(fair):
    sub = fair[fair.attribute.isin(["gender", "disability", "imd_band", "age_band"])].copy()
    sub["label"] = sub.attribute + ": " + sub.group.astype(str)
    sub = sub.sort_values(["attribute", "group"])
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(sub) + 1.5))
    y = np.arange(len(sub))
    ax.barh(y - 0.2, sub["recall_at_risk (TPR)"], 0.4, label="at-risk recall (TPR)")
    ax.barh(y + 0.2, sub["false_alarm_rate (FPR)"], 0.4, label="false-alarm rate (FPR)")
    ax.set_yticks(y)
    ax.set_yticklabels(sub["label"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_title("Early-warning error rates by group (SDG 10 audit)")
    ax.legend(loc="lower right")
    _save(fig, "14_fairness.png")
