"""Next-resource recommender from implicit feedback (VLE site visits).

For every module presentation: learners who visited a set of sites in the first
`cutoff` days are given a ranked list of *new* sites; we check which sites they
actually first visited in the following `horizon` days (a temporal hold-out,
which is how the system would be used in production).
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

from .config import CUTOFF_DAY, SEED

HORIZON_DAYS = 60
TOP_K = 10
MIN_ITEM_USERS = 10      # drop sites visited by fewer than this many students
MIN_TRAIN_ITEMS = 3


class PresentationData:
    """Binary user x item matrices for one module presentation."""

    def __init__(self, keys, items, X_train, X_test):
        self.keys, self.items = np.asarray(keys), np.asarray(items)
        self.X_train, self.X_test = X_train.tocsr(), X_test.tocsr()


def build_presentation_data(visits, enrol, cutoff=CUTOFF_DAY, horizon=HORIZON_DAYS):
    """visits: (key, id_site, first_date); enrol: DataFrame with key, code_module, code_presentation."""
    out = {}
    v_all = visits.merge(enrol[["key", "code_module", "code_presentation"]], on="key", how="inner")
    for (mod, pres), grp in v_all.groupby(["code_module", "code_presentation"]):
        keys = grp["key"].unique()
        item_users = grp.groupby("id_site")["key"].nunique()
        items = item_users[item_users >= MIN_ITEM_USERS].index.to_numpy()
        if len(items) < 30 or len(keys) < 100:
            continue
        kpos = pd.Series(np.arange(len(keys)), index=keys)
        ipos = pd.Series(np.arange(len(items)), index=items)
        g = grp[grp["id_site"].isin(items)]
        u, i = kpos[g["key"]].to_numpy(), ipos[g["id_site"]].to_numpy()
        train = g["first_date"].to_numpy() <= cutoff
        test = (g["first_date"].to_numpy() > cutoff) & (g["first_date"].to_numpy() <= cutoff + horizon)
        shape = (len(keys), len(items))
        out[(mod, pres)] = PresentationData(
            keys, items,
            sp.csr_matrix((np.ones(train.sum()), (u[train], i[train])), shape=shape),
            sp.csr_matrix((np.ones(test.sum()), (u[test], i[test])), shape=shape))
    return out


# ----------------------------------------------------------------- scorers
def _popularity(X):
    return np.asarray(X.sum(axis=0)).ravel()


def _item_knn_similarity(X):
    X = X.astype(float)
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=0)).ravel()) + 1e-9
    S = (X.T @ X).toarray() / np.outer(norms, norms)
    np.fill_diagonal(S, 0.0)
    return S


def score_item_knn(X, S):
    return np.asarray(X @ S)


def score_svd(X, k=20):
    k = min(k, X.shape[1] - 1, X.shape[0] - 1)
    svd = TruncatedSVD(n_components=k, random_state=SEED)
    U = svd.fit_transform(X)
    return U @ svd.components_


def score_persona_popularity(X, persona):
    """Popularity inside each persona (share of persona members who visited the site)."""
    scores = np.zeros(X.shape, dtype=float)
    for p in np.unique(persona):
        m = persona == p
        scores[m] = np.asarray(X[m].mean(axis=0)).ravel()
    return scores


def _rownorm(S):
    lo, hi = S.min(axis=1, keepdims=True), S.max(axis=1, keepdims=True)
    return (S - lo) / np.maximum(hi - lo, 1e-9)


def all_scores(X, persona, rng):
    S = _item_knn_similarity(X)
    knn = score_item_knn(X, S)
    pers = score_persona_popularity(X, persona)
    return {
        "Random": rng.random(X.shape),
        "Popularity": np.tile(_popularity(X), (X.shape[0], 1)).astype(float),
        "Persona popularity": pers,
        "Item-kNN (cosine)": knn,
        "SVD (k=20)": score_svd(X),
        "Hybrid (kNN + persona)": 0.5 * _rownorm(knn) + 0.5 * _rownorm(pers),
    }, S


# ---------------------------------------------------------------- metrics
def ranking_metrics(scores, X_train, X_test, k=TOP_K):
    s = np.where(X_train.toarray() > 0, -np.inf, scores)
    rel = X_test.toarray() > 0
    ok = rel.sum(axis=1) > 0
    top = np.argsort(-s, axis=1, kind="stable")[:, :k]
    hits = np.take_along_axis(rel, top, axis=1)
    n_rel = rel.sum(axis=1)
    disc = 1 / np.log2(np.arange(2, k + 2))
    dcg = (hits * disc).sum(axis=1)
    idcg = np.array([disc[:min(n, k)].sum() for n in n_rel])
    return {
        f"precision@{k}": float((hits.sum(axis=1) / k)[ok].mean()),
        f"recall@{k}": float((hits.sum(axis=1) / np.maximum(n_rel, 1))[ok].mean()),
        f"ndcg@{k}": float((dcg / np.maximum(idcg, 1e-9))[ok].mean()),
        f"hit_rate@{k}": float(hits.any(axis=1)[ok].mean()),
        "coverage": float(len(np.unique(top)) / scores.shape[1]),
        "n_users": int(ok.sum()),
    }


def run_recommender(visits, enrol, personas_by_key, seed=SEED):
    """Evaluate every model on every presentation; returns macro-averaged table."""
    rng = np.random.default_rng(seed)
    data = build_presentation_data(visits, enrol)
    per_pres, artifacts = [], {}
    for (mod, pres), d in data.items():
        has_train = np.asarray(d.X_train.sum(axis=1)).ravel() >= MIN_TRAIN_ITEMS
        has_rel = np.asarray(d.X_test.sum(axis=1)).ravel() > 0
        persona = pd.Series(d.keys).map(personas_by_key).fillna(-1).to_numpy()
        scores, S = all_scores(d.X_train, persona, rng)
        for name, sc in scores.items():
            rows = np.where(has_train)[0]
            m = ranking_metrics(sc[rows], d.X_train[rows], d.X_test[rows])
            per_pres.append({"module": mod, "presentation": pres, "model": name, **m})
        artifacts[f"{mod}_{pres}"] = {"keys": d.keys, "items": d.items, "X_train": d.X_train,
                                      "S": S.astype(np.float32), "persona": persona}
    per = pd.DataFrame(per_pres)
    metric_cols = [c for c in per.columns if c not in ("module", "presentation", "model", "n_users")]
    table = (per.groupby("model", sort=False)[metric_cols].mean()
             .assign(n_presentations=len(data)).reset_index())
    return {"table": table, "per_presentation": per, "artifacts": artifacts,
            "n_presentations": len(data)}


# ----------------------------------------------------------- serving
def recommend_for(artifact, key, site_catalogue, top_k=5, persona=None):
    """Top-k unseen sites for an enrolment key using the hybrid scorer."""
    keys = artifact["keys"]
    idx = np.where(keys == key)[0]
    if len(idx) == 0:
        return pd.DataFrame(columns=["id_site", "activity_type", "week_from", "score"])
    u = idx[0]
    x = artifact["X_train"][u]
    seen = x.toarray().ravel() > 0
    knn = np.asarray(x @ artifact["S"]).ravel()
    p = persona if persona is not None else artifact["persona"][u]
    same = artifact["persona"] == p
    pers = np.asarray(artifact["X_train"][same].mean(axis=0)).ravel()
    def norm(v):
        return (v - v.min()) / max(v.max() - v.min(), 1e-9)
    score = 0.5 * norm(knn) + 0.5 * norm(pers)
    score[seen] = -np.inf
    top = np.argsort(-score)[:top_k]
    cat = site_catalogue.set_index("id_site")
    out = pd.DataFrame({"id_site": artifact["items"][top], "score": score[top]})
    out["activity_type"] = out["id_site"].map(cat["activity_type"])
    out["week_from"] = out["id_site"].map(cat["week_from"])
    return out
