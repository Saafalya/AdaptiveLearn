# Presentation outline (≈ 8–10 min + demo)

| # | Slide | Key message | Visual (`results/figures`) | Time |
|---|---|---|---|---|
| 1 | Title & team | Project, SDG 4 + 10, Division C | – | 0:30 |
| 2 | The problem | 40 % of students at day 90 end up failing or withdrawing; tutors can't read the logs | `01_outcomes_vs_engagement` | 1:00 |
| 3 | Idea & objectives | Day-90 data → risk, score, persona, resources, study plan | pipeline diagram (report §3) | 1:00 |
| 4 | Data | Real OULAD: 32.6k enrolments, 10.6M clicks, 7 tables, CC BY 4.0; no-peeking rule | table (report §2.1) | 1:15 |
| 5 | Preprocessing & features | 34 features; missing-indicators; grouped splits (students repeat) | `02_missing_values`, `03_correlation` | 1:00 |
| 6 | Predicting outcomes | RF macro-F1 0.50 (baseline 0.16); **at-risk AUC 0.86** | `08_classification_comparison`, `09_confusion_matrix` | 1:30 |
| 7 | How early / what matters | AUC 0.80 at day 30; assessments > engagement > demographics | `04_early_prediction`, `11_ablation` | 1:00 |
| 8 | Personas & recommender | 5 personas validated on real outcomes; recommender modest | `07_persona_outcomes`, `13_recommender_comparison` | 1:15 |
| 9 | Live demo | Pick a held-out student → risk, persona, plan; what-if sliders | Streamlit app | 1:30 |
| 10 | Fairness, limitations, future work | False alarms higher for most deprived; offline only; needs pilot | `14_fairness` | 1:00 |

**Delivery tips**
- Open with one student story from the demo, not algorithm names.
- Say the limitations first (modest 4-class accuracy, no learning-gain evidence) — the rubric rewards limitations and future scope.
- Split speaking by pipeline stage so every member shows individual understanding.
