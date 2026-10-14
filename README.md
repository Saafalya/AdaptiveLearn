# AdaptiveLearn

**AI-Based Personalized Learning Recommendation System Using Machine Learning**
AUCML Lab ESE project · Department of AIML · SDG 4 (Quality Education) & SDG 10 (Reduced Inequalities)

From a student's topic scores and behavioural logs, AdaptiveLearn

| Task | ML technique | Output |
|---|---|---|
| Learner level | Classification (RF, GB, LR, SVM, KNN, DT) | beginner / intermediate / advanced |
| Expected score | Regression (Lasso, Ridge, RF, GB, SVR, KNN) | predicted final score |
| Learner personas | Clustering (K-Means, GMM, Agglomerative) | High achievers / Steady / At-risk |
| Resources | Recommender (popularity, SVD, content-based GBM, hybrid) | top-k resources |
| Learning path | Rules on top of ML outputs | ordered, prerequisite-aware plan with checkpoints |

## Quick start

```bash
pip install -r requirements.txt
python run_all.py                      # generates data, trains, evaluates (~4 min) -> results/
python -m unittest discover -s tests   # fast sanity tests
streamlit run app/streamlit_app.py     # interactive demo
```

## Results at a glance (seed 42)

| Task | Best model | Score |
|---|---|---|
| Level classification | Random Forest (tuned) | macro-F1 0.809, accuracy 0.820, ROC-AUC 0.944 (baseline F1 0.234) |
| Score regression | Lasso | RMSE 4.37, MAE 3.48, R² 0.876 (baseline RMSE 12.43) |
| Personas | K-Means, k = 3 | silhouette 0.19, stability ARI 0.97 |
| Recommendation | Content-based GBM | RMSE 0.705 vs 0.998 popularity; Hit@5 0.389 vs 0.173 random |

Full discussion, tables and caveats: [`docs/report.md`](docs/report.md).

## Project layout

```
src/adaptivelearn/   config · data_generation · preprocessing · supervised · clustering
                     recommender · learning_path · inference · viz
run_all.py           end-to-end pipeline
app/                 Streamlit demo
data/                students.csv · resources.csv · ratings.csv  (generated, seed 42)
results/             metrics.json · tables/*.csv · figures/*.png · models/ (regenerated)
docs/                report · rubric_mapping · presentation_outline · viva_prep
tests/               unit tests
```

## Honest note on data
The dataset is **synthetic** (documented simulator in `data_generation.py`) because no public dataset combines per-topic scores, behaviour logs and resource feedback. Metrics therefore show that the pipeline recovers a known process, not real-world learning gains. To use real data, supply `data/students.csv`, `resources.csv` and `ratings.csv` with the same columns.

## Team
_<names / PRNs / division>_
