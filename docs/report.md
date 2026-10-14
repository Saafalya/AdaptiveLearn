# AI-Based Personalized Learning Recommendation System Using Machine Learning

**AUCML Lab – ESE Case Study Project (TE7483)** · Department of AIML · Batch 2024-28
**Team:** _<Member 1 – PRN>, <Member 2 – PRN>, <Member 3 – PRN>, <Member 4 – PRN>_ · **Division:** _<div>_
**Application domain:** Education · **SDGs:** SDG 4 (Quality Education), SDG 10 (Reduced Inequalities)

> All numbers below are produced by `python run_all.py` (seed 42) and are stored in `results/metrics.json` and `results/tables/`.

---

## 1. Problem refinement and use-case relevance

**Problem.** Most learning platforms serve the same content in the same order to every learner, even though learners differ in prior knowledge, pace, engagement and weak areas. Weak students are lost; strong students are bored; and learners with limited connectivity are served the same heavy video content as everyone else.

**Objective.** Build an ML pipeline that, from a learner's diagnostic scores and behavioural logs,

1. **predicts the level** (beginner / intermediate / advanced) they will reach — *classification*;
2. **predicts the expected final-assessment score** — *regression*;
3. **groups learners into personas** to guide instructors — *clustering*;
4. **recommends resources** (topic, difficulty, format) — *recommendation*;
5. **assembles a prerequisite-aware learning path** with checkpoint/advance rules.

**Stakeholders.** Students (personalised path), instructors (cohort personas and early-warning of at-risk learners), institutions/EdTech platforms (scalable personalisation).

**Scope and boundaries.** Six technical topics (Python, SQL, Statistics, Linear Algebra, DSA, Machine Learning); 54 learning resources (6 topics × 3 difficulty tiers × 3 formats). The system *recommends*, it does not grade or replace instructors.

**SDG alignment.** *SDG 4* – targets personalised, effective learning outcomes. *SDG 10* – the data contains an `internet_access` attribute and the evaluation includes a **per-group fairness check**, so the system can be audited for under-serving low-connectivity learners (Section 5.5).

## 2. Dataset understanding, preprocessing and feature handling

### 2.1 Source and why it is synthetic
No public dataset combines (a) per-topic diagnostic scores, (b) behavioural logs, and (c) learner-resource feedback. The project therefore uses a **documented, seeded simulator** (`src/adaptivelearn/data_generation.py`):

```
aptitude, motivation, internet access  ->  topic mastery (with prerequisite effects: DSA<-Python; ML<-Python,Stats,LinAlg)
mastery + assignments + study hours    ->  hidden final score  ->  level
mastery vs. resource tier/format       ->  resource ratings (1-5)
```
Realistic data-quality problems are injected *after* the clean values are used to generate ratings, so the targets are not corrupted but the features are.

> **Limitation (stated openly):** results measure how well the ML pipeline recovers a simulated learning process; they are not evidence of real-world learning gains. The code is written so that a real dataset (e.g. OULAD or UCI *Student Performance*) can replace `data/students.csv` with the same column names.

### 2.2 Dataset summary
| Item | Value |
|---|---|
| Students | 1,500 (19 raw columns) |
| Resources | 54 (`resources.csv`) |
| Ratings | 22,500 (15 per student, matrix density 27.8 %) |
| Train / test split | 1,200 / 300, stratified on level |
| Target 1 (classification) | `level`: intermediate 53.9 %, beginner 26.3 %, advanced 19.7 % |
| Target 2 (regression) | `final_score` (0–100) |
| Model input features | 22 (6 topic scores, 8 behaviour, 2 context categorical, 5 engineered numeric, 1 engineered categorical) |

Level bands: `< 55` beginner, `55–74.9` intermediate, `≥ 75` advanced. Figures: `01_distributions.png`, `03_correlation.png`.

### 2.3 Data quality issues and handling
| Issue | Size | Treatment |
|---|---|---|
| Topic not attempted (`score_*` missing) | 5.3–7.0 % per topic | Median imputation **plus a missing-indicator column** (non-attempt is informative) |
| Missing `prior_cgpa`, `avg_session_min`, `logins_per_week`, `quiz_attempts` | 2–5 % | Median imputation inside the CV pipeline |
| Impossible `weekly_study_hours` (> 60, entry errors) | 15 rows | Domain rule → set to NaN → imputed |
| Bounded variables | – | Scores clipped to [0,100], CGPA to [0,10] |
| Scale differences | – | `StandardScaler` (for LR, KNN, SVM, clustering) |
| Categoricals | `internet_access`, `preferred_format`, `weakest_topic` | Mode imputation + one-hot |

See `02_missing_values.png`.

### 2.4 Feature engineering
`mean_topic_score`, `min_topic_score`, `score_spread`, `n_topics_attempted`, `weakest_topic`, `engagement_index = mean(logins/21, session/120, assignments/100)`. All are **row-wise** functions of a single student, so no dataset statistics leak across the split. Feature importance (permutation, test set): `mean_topic_score` (0.038) and `assignments_completed_pct` (0.036) lead, followed by `min_topic_score` and `weekly_study_hours` (`09_feature_importance.png`).

### 2.5 Validation strategy
Stratified 80/20 hold-out; **5-fold stratified CV on the training set** for model selection and grid search; the test set is touched once per model for reporting and is **never used to pick the model**. All imputers/scalers/encoders sit inside sklearn `Pipeline`s, so they are fitted on training folds only.

## 3. Methodology and ML pipeline design

```
raw logs ──► clean ──► feature engineering ──► ColumnTransformer (impute/scale/encode)
                                                   │
        ┌──────────────────┬───────────────────────┼──────────────────────┐
        ▼                  ▼                       ▼                      ▼
  Clustering          Classification          Regression           Recommender
  (personas)          (level)                 (final score)        (resource rating)
        └──────────────────┴────────────┬──────────┴──────────────────────┘
                                         ▼
                          Learning-path generator (prerequisites + checkpoints)
```

| Task | Candidate algorithms | Why |
|---|---|---|
| Classification | Baseline, Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting, SVM-RBF (+ grid-searched RF/GB) | Spans linear, instance-based, single tree, bagging, boosting and kernel methods; baseline sets the floor |
| Regression | Baseline, Linear, Ridge, Lasso, KNN, Random Forest, Gradient Boosting, SVR | Same reasoning; regularised linear models test whether complexity is needed |
| Clustering | K-Means, Gaussian Mixture, Agglomerative (Ward) | Centroid, probabilistic and hierarchical views; K-Means gives deployable `predict` |
| Recommendation | Random, Popularity, User+item bias, Collaborative (low-rank SVD completion), Content-based (Gradient Boosting on student–resource pair features), Hybrid | Standard ladder from naïve to personalised; content features handle **cold-start** students |

**Evaluation plan.** Classification: accuracy, macro precision/recall/F1, macro one-vs-rest ROC-AUC, confusion matrix (macro-F1 because classes are imbalanced). Regression: RMSE, MAE, R². Clustering: silhouette, Davies–Bouldin, Calinski–Harabasz, run-to-run stability (ARI). Recommendation: RMSE/MAE on held-out ratings and Precision/Recall/NDCG/Hit-rate @5 with a per-student 80/20 hold-out (relevant = rating ≥ 4).

**Learning-path rules.** Topics scoring < 60 (or unassessed) are remediated, weakest first; shaky prerequisites (< 65) are pulled in and ordered topologically; tier = Fundamentals (< 55) / Core (55–74) / Advanced (≥ 75); each topic ends in a checkpoint — *score ≥ 75 % → advance a tier, otherwise repeat practice*; mastered topics get stretch resources.

## 4. Implementation

| Component | File |
|---|---|
| Config, topic taxonomy, thresholds | `src/adaptivelearn/config.py` |
| Simulator | `src/adaptivelearn/data_generation.py` |
| Cleaning / features / transformer | `src/adaptivelearn/preprocessing.py` |
| Classification & regression | `src/adaptivelearn/supervised.py` |
| Personas | `src/adaptivelearn/clustering.py` |
| Recommender | `src/adaptivelearn/recommender.py` |
| Learning path | `src/adaptivelearn/learning_path.py` |
| Inference for one student | `src/adaptivelearn/inference.py` |
| Figures | `src/adaptivelearn/viz.py` |
| End-to-end runner | `run_all.py` |
| Demo web app | `app/streamlit_app.py` |
| Tests | `tests/test_pipeline.py` |

Reproducibility: fixed seed, `requirements.txt`, one command (`python run_all.py`, ≈ 4 min on a laptop CPU). The Streamlit app loads the saved models and, for any student profile, shows predicted level and score, persona, top-5 resources and the full learning path.

## 5. Results, evaluation and comparative analysis

### 5.1 Level classification (test set, n = 300)
| Model | CV macro-F1 (±sd) | Test acc. | Test macro-F1 | ROC-AUC |
|---|---|---|---|---|
| **Random Forest (tuned)** | **0.814 ± 0.023** | 0.820 | 0.809 | 0.944 |
| Random Forest | 0.812 ± 0.025 | 0.813 | 0.802 | 0.945 |
| Gradient Boosting (tuned) | 0.810 ± 0.028 | 0.813 | 0.805 | 0.940 |
| Logistic Regression | 0.800 ± 0.041 | 0.807 | 0.797 | 0.941 |
| Gradient Boosting | 0.799 ± 0.032 | 0.817 | 0.808 | 0.935 |
| SVM (RBF) | 0.783 ± 0.029 | 0.810 | 0.797 | 0.938 |
| Decision Tree | 0.768 ± 0.027 | 0.810 | 0.803 | 0.892 |
| KNN | 0.760 ± 0.016 | 0.763 | 0.740 | 0.918 |
| Baseline (majority) | 0.234 ± 0.001 | 0.540 | 0.234 | 0.500 |

*Reading.* Every real model beats the majority baseline by ~0.57 macro-F1. The top five models are within one CV standard deviation of each other, so **no single model is significantly better**; Random Forest (tuned) was selected on CV score, not test score. KNN is clearly worst (distance metrics suffer with 22+ mixed features). See `07_classification_comparison.png`, `08_confusion_matrix.png`.

### 5.2 Final-score regression (test set)
| Model | CV RMSE | Test RMSE | Test MAE | Test R² |
|---|---|---|---|---|
| **Lasso** | **4.63** | **4.37** | **3.48** | **0.876** |
| Ridge | 4.68 | 4.48 | 3.56 | 0.870 |
| Linear Regression | 4.69 | 4.47 | 3.56 | 0.870 |
| Gradient Boosting | 4.80 | 4.63 | 3.61 | 0.861 |
| Random Forest | 4.82 | 4.49 | 3.52 | 0.870 |
| KNN | 5.70 | 5.37 | 4.22 | 0.813 |
| SVR (RBF) | 5.75 | 5.16 | 4.05 | 0.827 |
| Baseline (mean) | 13.15 | 12.43 | 10.30 | 0.000 |

*Reading.* Simple regularised linear models win. This is expected here: the simulator builds the final score as a (noisy) linear combination of the inputs, so extra model capacity buys nothing. On real data we would expect ensembles to be more competitive. The average error of ≈ 3.5 points is smaller than the width of a level band (20 points).

**Feature-group ablation (Lasso):** topic scores only → RMSE 4.94; behaviour & context only → 6.23; all features → **4.37**. Diagnostic scores carry most of the signal, but behaviour adds a measurable improvement, which justifies collecting engagement logs.

### 5.3 Learner personas (clustering)
Selected **k = 3** (K-Means, silhouette 0.188, Davies–Bouldin 1.58, Calinski–Harabasz 606; restricted to k ∈ [3,6] for pedagogical usefulness). Stability across 10 random starts: **ARI = 0.97**.

| Persona | n | Mean topic score | Engagement idx | Final score | Level mix (B / I / A) |
|---|---|---|---|---|---|
| High achievers | 386 | 80.6 | 0.53 | 78.3 | 0 % / 34 % / 66 % |
| Steady learners | 675 | 66.2 | 0.42 | 64.0 | 8 % / 86 % / 6 % |
| At-risk learners | 439 | 50.5 | 0.33 | 49.1 | 78 % / 22 % / 0 % |

| Algorithm | Silhouette | Davies–Bouldin | Calinski–Harabasz |
|---|---|---|---|
| **K-Means** | 0.188 | 1.58 | 606 |
| Agglomerative (Ward) | 0.181 | 1.45 | 504 |
| Gaussian Mixture | 0.046 | 4.74 | 132 |

*Reading.* Silhouette values are low (< 0.25): the learner population is a **continuum, not well-separated blobs**, so personas are useful *segments*, not natural species. K-Means and Ward agree; GMM struggles with 15 correlated dimensions. The personas correlate strongly with the supervised level label although the label was never used in clustering. `ML` is the weakest topic for at-risk learners (41.6), consistent with its three prerequisites. See `04_cluster_selection.png`, `05_cluster_pca.png`, `06_persona_profiles.png`.

### 5.4 Recommender (per-student 80/20 hold-out; 18,000 train / 4,500 test ratings)
| Model | RMSE | MAE | P@5 | R@5 | NDCG@5 | Hit@5 |
|---|---|---|---|---|---|---|
| Random | 1.609 | 1.318 | 0.036 | 0.109 | 0.072 | 0.173 |
| Popularity (item mean) | 0.998 | 0.820 | 0.061 | 0.185 | 0.126 | 0.279 |
| User+item bias | 1.013 | 0.828 | 0.061 | 0.185 | 0.126 | 0.279 |
| Collaborative (SVD) | 1.022 | 0.818 | 0.058 | 0.176 | 0.120 | 0.265 |
| **Content-based (GBM)** | **0.705** | **0.551** | **0.085** | **0.260** | **0.174** | **0.389** |
| Hybrid (0.5/0.5) | 0.779 | 0.630 | 0.075 | 0.231 | 0.154 | 0.345 |

*Reading.* The content-based model reduces rating RMSE by 29 % over popularity and more than doubles Hit@5 relative to random (0.389 vs 0.173). **Collaborative SVD does not beat the popularity baseline** — with only 15 ratings per student and ratings driven mainly by *fit between mastery and resource level*, there is little latent collaborative structure to exploit; this is a legitimate finding, not a bug. Blending in SVD monotonically degrades results (α sweep, `recommender_alpha_sweep.csv`), so the deployed recommender is the content-based model. Because it takes student *features* rather than a user id, it also handles **cold-start** students. Caveat: ratings were simulated from the same pair features the content model sees, which favours it by construction.

### 5.5 Fairness check (SDG 10)
| Internet access | n | Accuracy | Macro-F1 |
|---|---|---|---|
| Limited | 67 | 0.836 | 0.848 |
| Moderate | 123 | 0.805 | 0.791 |
| Good | 110 | 0.827 | 0.807 |

No group is under-served by the level classifier (differences are within sampling noise at these group sizes; n = 67 for "limited").

## 6. Conclusion, limitations and future scope

**Conclusion.** A pipeline of four complementary ML tasks gives a working personalisation engine: it predicts learner level with ≈ 0.81 macro-F1, final score within ≈ 3.5 points, segments learners into three interpretable personas, ranks resources with 29 % lower error than a popularity baseline, and turns these outputs into a prerequisite-aware path.

**Limitations.** (1) Synthetic data — conclusions about *absolute* accuracy do not transfer to real learners; (2) linear generative process favours linear regressors; (3) simulated ratings favour the content model; (4) no longitudinal data, so *learning gain* from following the path is not measured; (5) small subgroup sizes limit fairness conclusions.

**Future scope.** Validate on real LMS data (OULAD, institutional logs); sequence models / knowledge tracing (BKT, DKT) for time-aware mastery; contextual bandits to learn from learner responses online; explainability (SHAP) in the instructor dashboard; A/B test of path-following vs. control for learning gain.

## References
1. Pedregosa et al., "Scikit-learn: Machine Learning in Python", *JMLR* 12, 2011.
2. Breiman, L., "Random Forests", *Machine Learning* 45(1), 2001.
3. Friedman, J., "Greedy Function Approximation: A Gradient Boosting Machine", *Annals of Statistics* 29(5), 2001.
4. Koren, Bell, Volinsky, "Matrix Factorization Techniques for Recommender Systems", *IEEE Computer* 42(8), 2009.
5. Rousseeuw, P., "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis", *J. Comput. Appl. Math.* 20, 1987.
6. Corbett & Anderson, "Knowledge tracing: Modeling the acquisition of procedural knowledge", *UMUAI* 4, 1994.
7. United Nations, *Sustainable Development Goals 4 and 10*, https://sdgs.un.org/goals.
