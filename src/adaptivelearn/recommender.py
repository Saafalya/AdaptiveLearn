"""Resource recommender: baselines, collaborative filtering, content-based and hybrid.

Ratings (1-5) are predicted for every (student, resource) pair; the top-k
unseen resources are recommended. Evaluation is a per-student hold-out.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from .config import ADVANCED_MIN, BEGINNER_MAX, SEED, TOPICS, TOPIC_COLS

PAIR_FEATURES = [
    "topic_score", "level", "appropriate_level", "level_gap", "abs_level_gap",
    "format_match", "is_video", "is_reading", "is_practice", "duration_min",
    "mean_topic_score", "engagement_index",
]
RELEVANT_RATING = 4
TOP_K = 5


# ----------------------------------------------------------------- features
def pair_features(students, resources, s_idx, r_idx):
    """Feature frame for the (students[s_idx], resources[r_idx]) pairs.

    `students` must already be passed through preprocessing.prepare().
    """
    s_idx, r_idx = np.asarray(s_idx), np.asarray(r_idx)
    topic_pos = resources["topic"].map({t: i for i, t in enumerate(TOPICS)}).to_numpy()
    scores = students[TOPIC_COLS].to_numpy(dtype=float)
    mean_score = students["mean_topic_score"].fillna(60.0).to_numpy()
    topic_score = scores[s_idx, topic_pos[r_idx]]
    topic_score = np.where(np.isnan(topic_score), mean_score[s_idx], topic_score)

    level = resources["level"].to_numpy()[r_idx]
    appropriate = np.where(topic_score < BEGINNER_MAX, 1, np.where(topic_score < ADVANCED_MIN, 2, 3))
    rtype = resources["rtype"].to_numpy()[r_idx]
    preferred = students["preferred_format"].to_numpy()[s_idx]
    return pd.DataFrame({
        "topic_score": topic_score,
        "level": level,
        "appropriate_level": appropriate,
        "level_gap": level - appropriate,
        "abs_level_gap": np.abs(level - appropriate),
        "format_match": (rtype == preferred).astype(int),
        "is_video": (rtype == "video").astype(int),
        "is_reading": (rtype == "reading").astype(int),
        "is_practice": (rtype == "practice").astype(int),
        "duration_min": resources["duration_min"].to_numpy()[r_idx],
        "mean_topic_score": mean_score[s_idx],
        "engagement_index": students["engagement_index"].fillna(0.6).to_numpy()[s_idx],
    })[PAIR_FEATURES]


# ---------------------------------------------------------------- split/mats
def split_ratings(ratings, test_frac=0.2, seed=SEED):
    test = ratings.groupby("student_id", group_keys=False).sample(frac=test_frac, random_state=seed)
    return ratings.drop(test.index), test


def to_matrix(ratings, students, resources):
    s_pos = {s: i for i, s in enumerate(students["student_id"])}
    r_pos = {r: i for i, r in enumerate(resources["resource_id"])}
    si = ratings["student_id"].map(s_pos).to_numpy()
    ri = ratings["resource_id"].map(r_pos).to_numpy()
    mat = np.full((len(s_pos), len(r_pos)), np.nan)
    mat[si, ri] = ratings["rating"].to_numpy()
    return mat, si, ri


# --------------------------------------------------------------- the models
def fit_bias_model(train_mat, shrink=10.0):
    mask = ~np.isnan(train_mat)
    mu = np.nanmean(train_mat)
    resid = np.where(mask, train_mat - mu, 0.0)
    b_i = resid.sum(0) / (mask.sum(0) + shrink)
    resid_u = np.where(mask, train_mat - mu - b_i[None, :], 0.0)
    b_u = resid_u.sum(1) / (mask.sum(1) + shrink)
    return mu, b_i, b_u


def svd_complete(residual, mask, k=6, iters=30):
    """Soft-impute style low-rank completion of the residual matrix."""
    X = np.where(mask, residual, 0.0)
    low = np.zeros_like(X)
    for _ in range(iters):
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        low = (U[:, :k] * s[:k]) @ Vt[:k]
        X = np.where(mask, residual, low)
    return low


def fit_predict_cf(train_mat, k=6):
    mu, b_i, b_u = fit_bias_model(train_mat)
    base = mu + b_u[:, None] + b_i[None, :]
    mask = ~np.isnan(train_mat)
    low = svd_complete(np.where(mask, train_mat - base, 0.0), mask, k=k)
    return base, base + low


def fit_content_model(students, resources, train_df, students_ids):
    s_pos = {s: i for i, s in enumerate(students_ids)}
    r_pos = {r: i for i, r in enumerate(resources["resource_id"])}
    si = train_df["student_id"].map(s_pos).to_numpy()
    ri = train_df["resource_id"].map(r_pos).to_numpy()
    X = pair_features(students, resources, si, ri)
    model = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.08,
                                      subsample=0.8, random_state=SEED)
    return model.fit(X, train_df["rating"].to_numpy())


def predict_content_matrix(model, students, resources):
    n_s, n_r = len(students), len(resources)
    s_idx = np.repeat(np.arange(n_s), n_r)
    r_idx = np.tile(np.arange(n_r), n_s)
    pred = model.predict(pair_features(students, resources, s_idx, r_idx))
    return np.clip(pred, 1, 5).reshape(n_s, n_r)


# --------------------------------------------------------------- evaluation
def rating_errors(pred, test_mat):
    m = ~np.isnan(test_mat)
    d = np.clip(pred, 1, 5)[m] - test_mat[m]
    return float(np.sqrt(np.mean(d ** 2))), float(np.mean(np.abs(d)))


def ranking_metrics(score, train_mask, test_mat, k=TOP_K):
    rel = (~np.isnan(test_mat)) & (np.nan_to_num(test_mat) >= RELEVANT_RATING)
    s = np.where(train_mask, -np.inf, score)          # never recommend already-seen items
    top = np.argsort(-s, axis=1, kind="stable")[:, :k]
    hits = np.take_along_axis(rel, top, axis=1)
    n_rel = rel.sum(1)
    ok = n_rel > 0
    discount = 1 / np.log2(np.arange(2, k + 2))
    dcg = (hits * discount).sum(1)
    idcg = np.array([discount[:min(n, k)].sum() for n in n_rel])
    return {
        f"precision@{k}": float((hits.sum(1) / k)[ok].mean()),
        f"recall@{k}": float((hits.sum(1) / np.maximum(n_rel, 1))[ok].mean()),
        f"ndcg@{k}": float((dcg / np.maximum(idcg, 1e-9))[ok].mean()),
        f"hit_rate@{k}": float(hits.any(1)[ok].mean()),
    }


def run_recommender(students, resources, ratings, seed=SEED):
    """`students` must be prepared (preprocessing.prepare)."""
    train_df, test_df = split_ratings(ratings, seed=seed)
    train_mat, _, _ = to_matrix(train_df, students, resources)
    test_mat, _, _ = to_matrix(test_df, students, resources)
    train_mask = ~np.isnan(train_mat)

    rng = np.random.default_rng(seed)
    mu, b_i, _ = fit_bias_model(train_mat)
    scores = {
        "Random": rng.random(train_mat.shape) * 4 + 1,
        "Popularity (item mean)": np.tile(mu + b_i, (len(students), 1)),
    }
    bias_pred, cf_pred = fit_predict_cf(train_mat)
    scores["User+item bias"] = bias_pred
    scores["Collaborative (SVD)"] = cf_pred

    content = fit_content_model(students, resources, train_df, students["student_id"])
    content_pred = predict_content_matrix(content, students, resources)
    scores["Content-based (GBM)"] = content_pred
    scores["Hybrid (0.5 SVD + 0.5 content)"] = 0.5 * cf_pred + 0.5 * content_pred

    rows = []
    for name, mat in scores.items():
        rmse, mae = rating_errors(mat, test_mat)
        rows.append({"model": name, "rmse": rmse, "mae": mae,
                     **ranking_metrics(mat, train_mask, test_mat)})
    table = pd.DataFrame(rows)

    sweep = []
    for a in np.linspace(0, 1, 5):  # sensitivity analysis only; alpha is not tuned on test
        mat = a * cf_pred + (1 - a) * content_pred
        rmse, _ = rating_errors(mat, test_mat)
        sweep.append({"alpha_cf": a, "rmse": rmse, **ranking_metrics(mat, train_mask, test_mat)})

    # Final production model: content model refit on all ratings (it needs no user id,
    # so it also serves brand-new students -> solves the cold-start problem).
    content_full = fit_content_model(students, resources, ratings, students["student_id"])
    return {"table": table, "alpha_sweep": pd.DataFrame(sweep), "content_model": content_full,
            "n_train": len(train_df), "n_test": len(test_df),
            "density": float(train_mask.mean())}
