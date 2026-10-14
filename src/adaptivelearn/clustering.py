"""Unsupervised learner profiling (personas) with K-Means, GMM and Agglomerative."""
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

from .config import BEHAVIOUR_COLS, SEED, TARGET_LEVEL, TARGET_SCORE, TOPIC_COLS

CLUSTER_COLS = TOPIC_COLS + BEHAVIOUR_COLS + ["engagement_index"]
K_RANGE = range(2, 9)
K_ALLOWED = (3, 6)  # at least three personas are pedagogically useful; more than six is unmanageable


def _prep():
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])


def _scores(X, labels):
    return {"silhouette": silhouette_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
            "calinski_harabasz": calinski_harabasz_score(X, labels)}


def name_personas(frame, labels):
    """Rule-based persona names from relative achievement and engagement."""
    s = frame["mean_topic_score"]
    e = frame["engagement_index"]
    zs = (s - s.mean()) / s.std()
    ze = (e - e.mean()) / e.std()
    g = pd.DataFrame({"zs": zs, "ze": ze, "c": labels}).groupby("c").mean()
    names = {}
    for c, r in g.iterrows():
        if r.zs > 0.4:
            n = "High achievers" if r.ze > 0 else "Capable but disengaged"
        elif r.zs < -0.4:
            n = "Hard-working strugglers" if r.ze > 0 else "At-risk learners"
        else:
            n = "Steady learners" if r.ze >= -0.1 else "Drifting learners"
        names[int(c)] = n
    seen = {}
    for c in sorted(names):  # disambiguate duplicates
        seen[names[c]] = seen.get(names[c], 0) + 1
        if list(names.values()).count(names[c]) > 1:
            names[c] = f"{names[c]} ({seen[names[c]]})"
    return names


def run_clustering(df):
    prep = _prep()
    X = prep.fit_transform(df[CLUSTER_COLS])

    sweep = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        sweep.append({"k": k, "inertia": km.inertia_, **_scores(X, km.labels_)})
    sweep = pd.DataFrame(sweep)
    allowed = sweep[(sweep.k >= K_ALLOWED[0]) & (sweep.k <= K_ALLOWED[1])]
    best_k = int(allowed.loc[allowed.silhouette.idxmax(), "k"])

    algos = {
        "K-Means": KMeans(n_clusters=best_k, n_init=10, random_state=SEED),
        "Gaussian Mixture": GaussianMixture(n_components=best_k, random_state=SEED, n_init=3),
        "Agglomerative (Ward)": AgglomerativeClustering(n_clusters=best_k),
    }
    comp, labels_by = [], {}
    for name, algo in algos.items():
        lab = algo.fit_predict(X)
        labels_by[name] = lab
        comp.append({"algorithm": name, **_scores(X, lab)})
    comparison = pd.DataFrame(comp)

    # Stability: agreement of K-Means across random initialisations.
    runs = [KMeans(n_clusters=best_k, n_init=1, random_state=s).fit_predict(X) for s in range(10)]
    ari = [adjusted_rand_score(runs[i], runs[j]) for i in range(10) for j in range(i + 1, 10)]

    km = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(X)
    labels = km.labels_
    personas = name_personas(df, labels)

    prof = df[CLUSTER_COLS + [TARGET_SCORE, "mean_topic_score"]].copy()
    prof["cluster"] = labels
    profile = prof.groupby("cluster").mean().round(2)
    profile.insert(0, "size", prof.groupby("cluster").size())
    profile.insert(0, "persona", [personas[c] for c in profile.index])
    level_mix = (pd.crosstab(labels, df[TARGET_LEVEL], normalize="index").round(3)
                 .reindex(columns=["beginner", "intermediate", "advanced"]))

    pca = PCA(n_components=2, random_state=SEED).fit(X)
    pipeline = Pipeline([("prep", prep), ("kmeans", km)])
    return {"best_k": best_k, "sweep": sweep, "comparison": comparison, "profile": profile,
            "level_mix": level_mix, "stability_ari": float(np.mean(ari)),
            "embedding": pca.transform(X), "explained": pca.explained_variance_ratio_,
            "labels": labels, "personas": personas, "pipeline": pipeline}
