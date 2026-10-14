# Presentation outline (≈ 8–10 min + demo)

| # | Slide | Key message | Visual (from `results/figures`) | Time |
|---|---|---|---|---|
| 1 | Title & team | Project, SDG 4 + 10 | – | 0:30 |
| 2 | The problem | One-size-fits-all learning; who is affected | simple before/after sketch | 1:00 |
| 3 | Objectives & scope | 4 ML tasks → 1 learning path | pipeline diagram (report §3) | 1:00 |
| 4 | Data | 1,500 students × 22 features, 54 resources, 22.5k ratings; why synthetic; injected issues | `01_distributions`, `02_missing_values` | 1:15 |
| 5 | Preprocessing & features | Imputation + missing indicators, leakage-safe pipelines, engineered features | `03_correlation` | 1:00 |
| 6 | Learner personas | 3 personas, silhouette 0.19, ARI 0.97 | `05_cluster_pca`, `06_persona_profiles` | 1:00 |
| 7 | Predicting level & score | RF 0.81 macro-F1; Lasso RMSE 4.4; model comparison | `07_classification_comparison`, `08_confusion_matrix` | 1:30 |
| 8 | Recommending resources | Content-based beats popularity by 29 % RMSE; SVD does not – and why | `11_recommender_comparison` | 1:15 |
| 9 | Live demo | Move sliders → level, persona, path | Streamlit app | 1:30 |
| 10 | Limitations, fairness, future work | Synthetic data caveat; fairness by access group; real LMS data + knowledge tracing | `12_fairness` | 1:00 |

**Delivery tips**
- Lead with the learner's story (the SQL 42 % / DSA 35 % example), not the algorithm names.
- State limitations yourself before the examiner does — the rubric rewards "limitations and future scope".
- Split speaking roles by pipeline stage so every member shows individual understanding.
