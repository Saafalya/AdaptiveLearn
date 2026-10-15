"""Unsupervised learner profiling (personas) on early engagement + assessment behaviour."""
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import SEED, TARGET_CLASS

COUNT_COLS = ["total_clicks", "active_days", "n_sites", "clicks_per_active_day",
              "clicks_last_14d", "n_submitted", "n_late"]
OTHER_COLS = ["days_since_last_active", "share_homepage", "share_content", "share_forum",
              "share_quiz", "submission_rate", "mean_score_early", "click_trend"]
CLUSTER_COLS = COUNT_COLS + OTHER_COLS
K_RANGE = range(2, 9)
K_ALLOWED = (3, 6)      # >= 3 personas are actionable; > 6 is unmanageable for instructors
SIL_SAMPLE = 6000       # silhouette / Ward are O(n^2): evaluate on a fixed random sample


def cluster_frame(df):
    """Heavy-tailed counts -> log1p; clip the click-trend ratio. Row-wise only."""
    x = df[CLUSTER_COLS].copy()
    for c in COUNT_COLS:
        x[c] = np.log1p(x[c].clip(lower=0))
    x["click_trend"] = x["click_trend"].clip(-1, 5)
    return x


def _prep():
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])


def _scores(X, labels, rng_seed=SEED):
    return {"silhouette": silhouette_score(X, labels, sample_size=min(SIL_SAMPLE, len(X)),
                                           random_state=rng_seed),
            "davies_bouldin": davies_bouldin_score(X, labels),
            "calinski_harabasz": calinski_harabasz_score(X, labels)}


def name_personas(df, labels):
    """Rule-based names from relative engagement and relative early score."""
    e = np.log1p(df["total_clicks"])
    s = df["mean_score_early"].fillna(df["mean_score_early"].median())
    f = df["share_forum"].fillna(0)
    ze, zs, zf = (e - e.mean()) / e.std(), (s - s.mean()) / s.std(), (f - f.mean()) / f.std()
    g = pd.DataFrame({"ze": ze, "zs": zs, "zf": zf, "c": labels}).groupby("c").mean()
    names = {}
    for c, r in g.iterrows():
        if r.zf > 0.8 and r.ze > -0.4:
            n = "Discussion-driven learners"
        elif r.ze > 0.4:
            n = "Engaged high performers" if r.zs > 0 else "Engaged but struggling"
        elif r.ze < -0.4:
            n = "Quiet high scorers" if r.zs > 0.3 else "Disengaged / at-risk"
        else:
            n = "Steady learners" if r.zs >= -0.2 else "Low-score, moderate effort"
        names[int(c)] = n
    counts = {}
    for c in sorted(names):
        counts[names[c]] = counts.get(names[c], 0) + 1
    seen = {}
    for c in sorted(names):
        if counts[names[c]] > 1:
            seen[names[c]] = seen.get(names[c], 0) + 1
            names[c] = f"{names[c]} ({seen[names[c]]})"
    return names


def run_clustering(df):
    prep = _prep().fit(cluster_frame(df))
    X = prep.transform(cluster_frame(df))
    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(X), size=min(SIL_SAMPLE, len(X)), replace=False)

    sweep = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=5, random_state=SEED).fit(X)
        sweep.append({"k": k, "inertia": km.inertia_, **_scores(X, km.labels_)})
    sweep = pd.DataFrame(sweep)
    allowed = sweep[(sweep.k >= K_ALLOWED[0]) & (sweep.k <= K_ALLOWED[1])]
    best_k = int(allowed.loc[allowed.silhouette.idxmax(), "k"])

    # algorithm comparison on a common random sample (Ward is O(n^2) memory)
    Xs = X[sample]
    comp = []
    for name, algo in {
        "K-Means": KMeans(n_clusters=best_k, n_init=10, random_state=SEED),
        "Gaussian Mixture": GaussianMixture(n_components=best_k, random_state=SEED, n_init=2),
        "Agglomerative (Ward)": AgglomerativeClustering(n_clusters=best_k),
    }.items():
        comp.append({"algorithm": name, **_scores(Xs, algo.fit_predict(Xs))})
    comparison = pd.DataFrame(comp)

    runs = [KMeans(n_clusters=best_k, n_init=1, random_state=s).fit_predict(X) for s in range(8)]
    ari = [adjusted_rand_score(runs[i], runs[j]) for i in range(8) for j in range(i + 1, 8)]

    km = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(X)
    labels = km.labels_
    personas = name_personas(df, labels)

    prof = df[["total_clicks", "active_days", "n_sites", "n_submitted", "submission_rate",
               "mean_score_early", "days_since_last_active", "share_forum", "future_mean_score"]].copy()
    prof["cluster"] = labels
    profile = prof.groupby("cluster").median().round(2)
    profile.insert(0, "size", prof.groupby("cluster").size())
    profile.insert(0, "persona", [personas[c] for c in profile.index])
    outcome_mix = (pd.crosstab(labels, df[TARGET_CLASS], normalize="index").round(3)
                   .reindex(columns=["Distinction", "Pass", "Fail", "Withdrawn"]))

    pca = PCA(n_components=2, random_state=SEED).fit(X)
    pipeline = {"prep": prep, "kmeans": km}
    return {"best_k": best_k, "sweep": sweep, "comparison": comparison, "profile": profile,
            "outcome_mix": outcome_mix, "stability_ari": float(np.mean(ari)),
            "embedding": pca.transform(X), "explained": pca.explained_variance_ratio_,
            "labels": labels, "personas": personas, "pipeline": pipeline}


def predict_cluster(artifact, df):
    p = artifact["pipeline"]
    return p["kmeans"].predict(p["prep"].transform(cluster_frame(df)))
