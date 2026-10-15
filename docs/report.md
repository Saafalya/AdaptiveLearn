# AI-Based Personalized Learning Recommendation System Using Machine Learning

**AUCML Lab – ESE Case Study Project (TE7483)** · Department of AIML · Batch 2024-28
**Team:** Swastik Gavhane (PRN 24070126224), Aryan Gawande (PRN 24070126223), Saafalya Sengupta (PRN 24070126158), _<Member 4 – name, PRN>_ · **Division:** C
**Application domain:** Education · **SDGs:** SDG 4 (Quality Education), SDG 10 (Reduced Inequalities)

> Every number below is produced by `python run_all.py` (seed 42) and stored in `results/metrics.json` and `results/tables/`. The data are **real**: the Open University Learning Analytics Dataset (OULAD).

---

## 1. Problem refinement and use-case relevance

**Problem.** Distance-learning courses lose a large share of students: in the dataset used here, 40 % of the students still enrolled at day 90 end up failing or withdrawing. Tutors cannot read thousands of activity logs, and every student receives the same materials in the same order. By the time a failed assignment reveals a problem, it is often too late to help.

**Objective.** Use only the data available **by day 90 of a course** (about one-third of the way through) to

1. **predict the final result** (Distinction / Pass / Fail / Withdrawn) — *classification*;
2. **predict performance on the remaining assessments** — *regression*;
3. **group learners into behavioural personas** for instructors — *clustering*;
4. **recommend the next learning resources** — *recommendation*;
5. **turn these outputs into a personalised, assessment-aligned study plan** with an early-warning flag.

**Stakeholders.** Students (what to study next), tutors/instructors (who to contact this week), institutions (retention, equitable support).

**Scope.** 7 modules, 22 presentations (2013–2014) from one distance-learning university. The system supports tutors and students; it does not make grading or admission decisions.

**SDG alignment.** *SDG 4* — earlier, personalised support for completing quality education. *SDG 10* — the data contain real socio-economic and demographic attributes (deprivation band, disability, gender, age), so the system is **audited for unequal error rates** across groups (Section 5.6).

## 2. Dataset understanding, preprocessing and feature handling

### 2.1 Source
**Open University Learning Analytics Dataset (OULAD)** — Kuzilek, Hlosta, Zdrahal, *Scientific Data* 4:170171 (2017), doi:10.1038/sdata.2017.171. Anonymised, **CC BY 4.0**. Obtained from the authors' Figshare record (the university's own download link was returning 404 at the time of download); file `anonymisedData.zip`, 46.7 MB, MD5 `7412686f…354308` verified.

| Table | Rows | Content |
|---|---|---|
| `studentInfo` | 32,593 | demographics, prior attempts, credits, **final_result** |
| `studentRegistration` | 32,593 | registration / unregistration day |
| `assessments` | 206 | type (TMA / CMA / Exam), due day, weight |
| `studentAssessment` | 173,912 | submission day and score |
| `vle` | 6,364 | learning-site metadata: activity type, week |
| `studentVle` | 10,655,280 | daily click counts per student per site |
| `courses` | 22 | module-presentation length |

32,593 enrolments by 28,785 students; **3,538 students enrolled more than once**.

### 2.2 Unit of analysis and "no peeking" rule
One row = one student's enrolment in one presentation, described **only by data up to day 90** (`CUTOFF_DAY`). Targets lie in the future:

| Target | Definition |
|---|---|
| `final_result` | Distinction, Pass, Fail, Withdrawn |
| `future_mean_score` | mean score of non-exam assessments *due after day 90* (available for 80.2 % of rows) |

Students who had **already unregistered by day 90** are removed (their outcome is known). This leaves **25,562 enrolments (23,398 students)**: Pass 48.4 %, Fail 27.6 %, Withdrawn 12.3 %, Distinction 11.8 %.

### 2.3 Features (34)
| Group | Examples |
|---|---|
| Demographic / context (12) | gender, region, highest education, IMD deprivation band, age band, disability, prior attempts, credits, module, season, registration day |
| Engagement from 10.6M clicks (14) | total clicks, active days, distinct sites, clicks before course start, last-14-day clicks and trend, days since last active, share of clicks on home / content / forum / quiz / other |
| Assessments up to day 90 (8) | number due, number submitted, submission rate, mean/min score, late submissions, mean days early, failed count |

### 2.4 Data quality and handling
| Issue | Size | Treatment |
|---|---|---|
| No early assessment submitted | 6.9 % of rows | median imputation **+ missing-indicator column** (non-submission is informative) |
| No VLE activity → click shares undefined | 2.0 % | median imputation |
| `imd_band` missing | 1,111 enrolments in raw data | own category "Unknown" |
| Skewed counts (clicks, sites) | – | `log1p` for clustering; scaling for linear/KNN models |
| Class imbalance (Withdrawn 12 %) | – | class weights, **macro-F1** as selection metric |
| Categorical variables | 8 | most-frequent imputation, one-hot (rare levels pooled) |
| Same student in several presentations | 3,538 students | **grouped splits** — no student is ever in both train and test (checked: 0 overlap) |

Figures: `01_outcomes_vs_engagement.png`, `02_missing_values.png`, `03_correlation.png`.

### 2.5 Validation strategy
Stratified, student-grouped 80/20 hold-out (20,449 train / 5,113 test). Model selection and grid search use **5-fold stratified group CV on the training set only**; the test set is used once per model for reporting. All imputers/scalers/encoders sit inside `Pipeline`s fitted per fold.

## 3. Methodology and ML pipeline design

```
OULAD CSVs ─► per-enrolment features @ day 90 ─► ColumnTransformer (impute / scale / one-hot)
                                         │
       ┌─────────────────┬───────────────┼────────────────┬──────────────────┐
       ▼                 ▼               ▼                ▼                  ▼
  Clustering       Classification   Regression      Recommender       Fairness audit
  (5 personas)     (4 outcomes)     (future score)  (next sites)      (SDG 10)
       └────────────┬────┴───────────────┴────────────────┘
                    ▼
        Study-plan generator (risk tier + peers + upcoming real assessments)
```

| Task | Algorithms | Rationale |
|---|---|---|
| Classification | Baseline, Logistic Regression, KNN, Decision Tree, Random Forest, Hist. Gradient Boosting (+ grid-searched RF and HGB) | linear → instance → tree → bagging → boosting; baseline sets the floor |
| Regression | Baseline, Ridge, Lasso, KNN, Random Forest, Hist. Gradient Boosting | same ladder |
| Clustering | K-Means, Gaussian Mixture, Agglomerative (Ward) | centroid / probabilistic / hierarchical views; K-Means gives deployable `predict` |
| Recommendation | Random, Popularity, Persona popularity, Item-kNN (cosine), SVD, Hybrid (kNN + persona) | standard ladder for implicit feedback; persona links to clustering |

**Evaluation plan.** Classification: accuracy, macro P/R/F1, macro OvR ROC-AUC, per-class F1, confusion matrix, plus the practical *at-risk* view (Fail ∪ Withdrawn) with ROC-AUC and average precision. Regression: RMSE, MAE, R². Clustering: silhouette, Davies–Bouldin, Calinski–Harabasz, stability (ARI) and *external validation against real outcomes*. Recommendation: Precision/Recall/NDCG/Hit-rate @10 and catalogue coverage under a **temporal hold-out** (train on sites first visited by day 90, test on sites first visited on days 91–150, averaged over 22 presentations). Additional analyses: feature-group ablation, **early-prediction sweep** (cut-offs 0–180), fairness audit.

**Study-plan rules.** From the predicted outcome probabilities: risk = P(Fail) + P(Withdrawn) → Low (< 35 %), Medium, High (≥ 60 %); tier from predicted future score: Foundations (< 55), Core, Stretch (≥ 75); an engagement-gap step if last-14-day clicks are below 80 % of successful peers in the same module; recommended resources from the recommender; and one "prepare for …" step with a checkpoint rule (≥ 75 % → move on, else repeat practice) for each of the next three *real* assessments of that module.

## 4. Implementation

| Component | File |
|---|---|
| Config, thresholds, feature groups | `src/adaptivelearn/config.py` |
| OULAD loading, 10.6M-row aggregation, features/targets | `src/adaptivelearn/oulad.py` |
| Preprocessing, grouped splits | `src/adaptivelearn/preprocessing.py` |
| Classification, regression, ablation, early sweep, fairness | `src/adaptivelearn/supervised.py` |
| Personas | `src/adaptivelearn/clustering.py` |
| Recommender | `src/adaptivelearn/recommender.py` |
| Study plan | `src/adaptivelearn/study_plan.py` |
| Single-student inference | `src/adaptivelearn/inference.py` |
| Runner · demo app · tests | `run_all.py` · `app/streamlit_app.py` · `tests/test_pipeline.py` |

Reproducible: fixed seed, `requirements.txt`, one command (≈ 6 min on a laptop). The Streamlit app lets a user pick a held-out student, see predicted outcome, risk, persona, recommended resources and study plan, and change behaviour sliders ("what if they were active more?").

## 5. Results, evaluation and comparative analysis

### 5.1 Outcome classification (4 classes; test n = 5,113)
| Model | CV macro-F1 (±sd) | Test acc. | Test macro-F1 | OvR AUC | At-risk AUC |
|---|---|---|---|---|---|
| **Random Forest (tuned)** | **0.507 ± 0.004** | 0.573 | 0.504 | 0.801 | 0.858 |
| Hist. Gradient Boosting (tuned) | 0.498 ± 0.001 | 0.531 | 0.492 | 0.800 | 0.859 |
| Random Forest | 0.495 ± 0.006 | 0.606 | 0.483 | 0.801 | 0.859 |
| Hist. Gradient Boosting | 0.495 ± 0.004 | 0.529 | 0.489 | 0.799 | 0.859 |
| Logistic Regression | 0.481 ± 0.004 | 0.505 | 0.473 | 0.788 | 0.851 |
| Decision Tree | 0.445 ± 0.010 | 0.480 | 0.447 | 0.766 | 0.830 |
| KNN | 0.393 ± 0.005 | 0.598 | 0.377 | 0.767 | 0.845 |
| Baseline (majority) | 0.163 | 0.483 | 0.163 | 0.500 | 0.500 |

Per class (selected model): Pass F1 0.66, Fail 0.58, Distinction 0.49, **Withdrawn 0.28**.

*Reading.* This is a genuinely hard task; honest numbers are modest. The selected model triples the baseline's macro-F1 (0.50 vs 0.16). Ensembles beat linear and distance-based models, but the top four are within ≈ 0.01 macro-F1, so none is decisively better. The weak class is *Withdrawn* (F1 0.28): leaving after day 90 is largely driven by life events that clicks cannot reveal. For **early warning the binary at-risk view is what matters: ROC-AUC 0.86, average precision 0.83** (baseline 0.40). Tuning RF improved CV macro-F1 by only 0.012 — the ceiling is the information in the data, not the algorithm. See `08_classification_comparison.png`, `09_confusion_matrix.png`.

### 5.2 Which data matters? (ablation, Hist. Gradient Boosting, test)
| Feature set | macro-F1 | at-risk AUC |
|---|---|---|
| Demographics only | 0.330 | 0.651 |
| Engagement (VLE clicks) only | 0.384 | 0.773 |
| Assessments only | 0.462 | 0.837 |
| Engagement + assessments | 0.475 | 0.852 |
| All features | 0.489 | 0.859 |

Assessment history is the strongest single source, engagement adds a clear increment, and demographics add little (and are the ethically sensitive ones). Permutation importance agrees: `mean_score_early` (0.044) ≫ `submission_rate` (0.022) > `min_score_early` (0.014) > `code_module` (0.010) (`10_feature_importance.png`, `11_ablation.png`).

### 5.3 How early can risk be detected? (cut-off sweep)
| Cut-off day | Students | Accuracy | Macro-F1 | At-risk AUC |
|---|---|---|---|---|
| 0 (course start) | 29,496 | 0.385 | 0.377 | 0.715 |
| 30 | 27,466 | 0.449 | 0.434 | 0.796 |
| 60 | 26,361 | 0.500 | 0.467 | 0.828 |
| **90** | 25,562 | 0.529 | 0.489 | 0.859 |
| 120 | 24,777 | 0.578 | 0.516 | 0.896 |
| 150 | 24,055 | 0.612 | 0.533 | 0.914 |
| 180 | 23,275 | 0.676 | 0.567 | 0.939 |

Even **before the course starts** (registration data + pre-start clicks) at-risk AUC is 0.72; by day 30 it is 0.80. Day 90 balances earliness against accuracy and still leaves ~170 days to intervene (`04_early_prediction.png`).

### 5.4 Future-score regression (test n = 4,080)
| Model | CV RMSE | Test RMSE | Test MAE | Test R² |
|---|---|---|---|---|
| **Hist. Gradient Boosting** | **11.39 ± 0.13** | **11.03** | **8.27** | **0.548** |
| Random Forest | 11.43 | 11.25 | 8.43 | 0.530 |
| Ridge | 12.29 | 12.03 | 9.16 | 0.462 |
| Lasso | 12.33 | 12.09 | 9.19 | 0.457 |
| KNN | 12.48 | 12.18 | 9.17 | 0.449 |
| Baseline (mean) | 16.41 | 16.41 | 12.81 | 0.000 |

Boosting explains ≈ 55 % of the variance in future assessment scores and cuts RMSE by 33 % versus predicting the mean. The ≈ 8-point MAE is enough to place a student in a Foundations / Core / Stretch tier, but not to predict an exact mark (`12_regression_fit.png`).

### 5.5 Learner personas (K-Means, k = 5)
k was restricted to 3–6 (a pedagogically usable range); within it silhouette peaks at k = 5 (0.186). Silhouette for k = 2 is higher (0.348) but only separates "active" from "inactive". Stability across 8 random starts: **ARI = 0.75** (moderate). Algorithm comparison on a 6,000-student sample: K-Means silhouette 0.184 / Davies–Bouldin 1.60 / Calinski–Harabasz 1,309; Agglomerative (Ward) 0.167 / 1.67 / 1,043; Gaussian Mixture 0.078 / 2.61 / 680.

| Persona | n | Median clicks | Active days | Median early score | Days idle | Real outcome (Distinction / Pass / Fail / Withdrawn) |
|---|---|---|---|---|---|---|
| Engaged high performers | 8,649 | 997 | 47 | 82 | 1 | 16 % / 62 % / 14 % / 8 % |
| Discussion-driven learners | 5,213 | 480 | 35 | 80 | 1 | 14 % / 56 % / 20 % / 10 % |
| Steady learners | 3,060 | 398 | 23 | 78 | 7 | 13 % / 43 % / 27 % / 17 % |
| Low-score, moderate effort | 6,701 | 209 | 16 | 69 | 15 | 7 % / 40 % / 38 % / 16 % |
| **Disengaged / at-risk** | 1,939 | **13** | **2** | – | **78** | 1 % / 7 % / **75 %** / 17 % |

*Reading.* Silhouette is low (< 0.2): learner behaviour is a continuum, so personas are *segments*, not natural species. But they are stable, interpretable, and — importantly — **validated against outcomes the clustering never saw**: the "Disengaged" persona fails or withdraws 92 % of the time versus 22 % for "Engaged high performers" (`06_cluster_pca.png`, `07_persona_outcomes.png`).

### 5.6 Fairness audit (SDG 10; "flag as at-risk" if predicted Fail/Withdrawn)
| Group | n | Actual at-risk | Flagged | Recall (TPR) | False-alarm (FPR) |
|---|---|---|---|---|---|
| Female | 2,371 | 38.8 % | 34.5 % | 0.66 | 0.15 |
| Male | 2,742 | 40.7 % | 35.3 % | 0.68 | 0.13 |
| No disability | 4,620 | 39.0 % | 34.2 % | 0.67 | 0.13 |
| Disability | 493 | 47.9 % | 41.4 % | 0.67 | 0.18 |
| IMD 0–10 % (most deprived) | 498 | 51.4 % | 48.2 % | 0.73 | 0.22 |
| IMD 90–100 % (least deprived) | 384 | 32.6 % | 27.1 % | 0.65 | 0.09 |
| Education below A-level | 1,956 | 49.6 % | 43.4 % | 0.70 | 0.17 |
| HE qualification | 703 | 32.9 % | 27.2 % | 0.64 | 0.09 |

*Reading.* Detection rates (recall) are similar across gender and disability and *higher* for the most deprived group, so the system does not miss disadvantaged students. The cost is **more false alarms** for the most deprived (22 % vs 9 %) and for students with lower prior education: the model flags in proportion to real differences in at-risk rates, which means some students would receive unnecessary outreach. Mitigation: use flags only to *offer* support, never to restrict opportunity; monitor these rates per cohort. The full table is in `results/tables/fairness_audit.csv` (`14_fairness.png`).

### 5.7 Next-resource recommender (22 presentations, temporal hold-out, top-10)
| Model | Precision@10 | Recall@10 | NDCG@10 | Hit-rate@10 | Coverage |
|---|---|---|---|---|---|
| Random | 0.096 | 0.077 | 0.103 | 0.545 | 0.99 |
| Popularity | 0.146 | 0.136 | 0.173 | 0.674 | 0.45 |
| Persona popularity | 0.148 | 0.135 | 0.177 | 0.675 | 0.48 |
| Item-kNN (cosine) | 0.150 | 0.138 | 0.176 | 0.676 | 0.47 |
| SVD (k = 20) | 0.150 | 0.141 | 0.174 | **0.721** | 0.60 |
| Hybrid (kNN + persona) | 0.150 | 0.138 | 0.177 | 0.677 | 0.47 |

*Reading.* Every learned model clearly beats random (precision +55 %), but **personalisation adds little over plain popularity** (precision 0.150 vs 0.146). The reason is structural: sites unlock on a teaching schedule, so what students open next is dominated by *when*, not *who*. SVD gives the best hit-rate (0.72 vs 0.67) and the widest coverage, but only the kNN-persona hybrid is deployed in the app because it is explainable ("popular among learners like you") and cheap to refresh. Improving this part would need schedule-aware features (`week_from`) and learning-gain feedback (`13_recommender_comparison.png`).

## 6. Conclusion, limitations and future scope

**Conclusion.** Using only the first 90 days of activity, the system identifies students who will fail or withdraw with ROC-AUC 0.86 (average precision 0.83), predicts future assessment scores with R² 0.55, segments learners into five outcome-validated personas, recommends next resources better than chance, and converts all this into a study plan tied to each module's real assessment calendar — with a fairness audit across deprivation, disability, gender and age.

**Limitations.** (1) One distance-learning university, 2013–2014: results may not transfer to other institutions or modalities. (2) Four-class accuracy is modest (macro-F1 0.50); *Withdrawn* is poorly predicted. (3) Predictive, not causal: the model shows who is at risk, not whether the recommended plan would help — **no learning-gain evaluation was possible offline**. (4) Recommender gains over popularity are small. (5) Demographic features can encode structural inequality; the audit found higher false-alarm rates for the most deprived. (6) Students who unregistered before day 90 are excluded by design.

**Future scope.** Pilot with tutors and A/B-test outreach (the only way to measure real benefit); survival models for time-to-withdrawal; sequence models of weekly activity; schedule-aware and content-aware recommenders; per-group threshold calibration and explanations (SHAP) for tutors; use of free-text forum content.

## Ethics and data protection
OULAD is publicly released, anonymised data under CC BY 4.0. No attempt is made to re-identify students. Predictions are risk estimates to trigger *offers of support*; they must not be used for grading, selection or sanction.

## References
1. Kuzilek, J., Hlosta, M., Zdrahal, Z., "Open University Learning Analytics dataset", *Scientific Data* 4:170171, 2017. doi:10.1038/sdata.2017.171
2. Pedregosa et al., "Scikit-learn: Machine Learning in Python", *JMLR* 12, 2011.
3. Breiman, L., "Random Forests", *Machine Learning* 45(1), 2001.
4. Friedman, J., "Greedy Function Approximation: A Gradient Boosting Machine", *Annals of Statistics* 29(5), 2001.
5. Sarwar, Karypis, Konstan, Riedl, "Item-based collaborative filtering recommendation algorithms", *WWW*, 2001.
6. Koren, Bell, Volinsky, "Matrix Factorization Techniques for Recommender Systems", *IEEE Computer* 42(8), 2009.
7. Rousseeuw, P., "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis", *J. Comput. Appl. Math.* 20, 1987.
8. United Nations, *Sustainable Development Goals 4 and 10*, https://sdgs.un.org/goals.
