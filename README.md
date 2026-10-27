# AdaptiveLearn

**AI-Based Personalized Learning Recommendation System Using Machine Learning**
AUCML Lab ESE project · Department of AIML · SDG 4 (Quality Education) & SDG 10 (Reduced Inequalities)

Using only the **first 90 days** of a course, AdaptiveLearn works on the real **Open University Learning Analytics Dataset (OULAD)** — 32,593 enrolments, 10.6 million click records — to

| Task | ML technique | Output |
|---|---|---|
| Outcome prediction | Classification (RF, HGB, LR, KNN, DT) | Distinction / Pass / Fail / Withdrawn + at-risk probability |
| Score prediction | Regression (HGB, RF, Ridge, Lasso, KNN) | expected score on future assessments |
| Learner personas | Clustering (K-Means, GMM, Ward) | 5 outcome-validated personas |
| Next resources | Implicit-feedback recommender (popularity, item-kNN, SVD, hybrid) | top-k learning sites to open next |
| Study plan | Rules on top of ML outputs | steps tied to the module's real upcoming assessments |

plus a feature-group ablation, an early-prediction sweep (day 0–180) and a **fairness audit** across deprivation, disability, gender and age.

## Quick start

**Windows, easiest:** double-click `start.bat`. It sets up a private Python environment, installs the packages, trains the models on first run and opens the website. On any system, `python run_website.py` does the same (`--quick` for faster first-time training).

```bash
pip install -r requirements.txt
# 1. get the data (CC BY 4.0, 46.7 MB) - see "Data" below - and unzip the 7 CSVs into data/oulad_raw/
python run_all.py                      # ~6 min -> results/ and data/processed/
python -m unittest discover -s tests   # fast sanity tests (no data needed)
streamlit run app/streamlit_app.py     # interactive demo
```

No raw data? `python train_from_processed.py` trains the classifier, regressor and personas from the committed `data/processed/` table (about 20 min on 2 cores, or `--quick`), then `streamlit run app/streamlit_app.py` works. The resource recommender still needs the raw click logs and `run_all.py`.

## Results at a glance (seed 42, held-out students)

| Task | Best model | Score | Baseline |
|---|---|---|---|
| 4-class outcome | Random Forest (tuned) | macro-F1 0.50, accuracy 0.57 | macro-F1 0.16 |
| **At-risk early warning** (Fail ∪ Withdrawn) | same model | **ROC-AUC 0.86**, avg. precision 0.83 | 0.50 / 0.40 |
| Future score | Hist. Gradient Boosting | RMSE 11.0, MAE 8.3, R² 0.55 | RMSE 16.4 |
| Personas | K-Means, k = 5 | "Disengaged" persona fails/withdraws 92 % vs 22 % for "Engaged high performers" | – |
| Next resource | SVD / hybrid | Hit-rate@10 0.72 / 0.68 vs 0.67 popularity, 0.55 random | – |

Honest summary: early warning works well; exact 4-class prediction and personalised resource ranking are modest. Full tables, discussion and limitations: [`docs/report.md`](docs/report.md).

## Data

Kuzilek, Hlosta, Zdrahal, *Open University Learning Analytics dataset*, Scientific Data 4:170171 (2017), doi:10.1038/sdata.2017.171 — CC BY 4.0.

- Official page: https://analyse.kmi.open.ac.uk/open_dataset (its download button was returning 404 when this project was built).
- Authors' Figshare record: https://figshare.com/articles/OULAD_Open_University_Learning_Analytics_Dataset/5081998 → `anonymisedData.zip` (MD5 `7412686fd77cf0e0ee1e8c3e9b354308`).

Unzip into `data/oulad_raw/` (files: `studentInfo.csv`, `studentRegistration.csv`, `assessments.csv`, `studentAssessment.csv`, `vle.csv`, `studentVle.csv`, `courses.csv`). The raw files are not committed (450 MB); small derived tables are in `data/processed/`.

## Project layout

```
src/adaptivelearn/   config · oulad · preprocessing · supervised · clustering
                     recommender · study_plan · inference · viz
run_all.py           end-to-end pipeline
app/                 Streamlit demo (pick a held-out student, what-if sliders)
data/processed/      enrolments_d90.csv.gz · assessments.csv · sites.csv
results/             metrics.json · tables/*.csv · figures/*.png · models/ (regenerated)
docs/                report · rubric_mapping · presentation_outline · viva_prep
tests/               unit tests
```

## Design choices worth knowing

- **No leakage:** features use only data up to day 90; targets lie in the future; students who had already withdrawn by day 90 are excluded; splits are grouped by student (3,538 students enrolled more than once).
- **Selection on CV, not test:** models are chosen by 5-fold grouped CV on the training set; the test set is used for reporting only.
- **Fairness is measured, not assumed** — see `results/tables/fairness_audit.csv`.

## Team
- Swastik Gavhane – 24070126224
- Aryan Gawande – 24070126223
- Saafalya Sengupta – 24070126158
- _<Member 4 – name, PRN>_

Division: C
