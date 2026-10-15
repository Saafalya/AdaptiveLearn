# Viva preparation – likely questions

**Where does the data come from? Is it real?**
Yes — OULAD, the Open University Learning Analytics Dataset (Kuzilek et al., *Scientific Data*, 2017), anonymised, CC BY 4.0: 32,593 course enrolments by 28,785 students and 10.6 million click records. We downloaded it from the authors' Figshare record and verified the MD5.

**What exactly are you predicting, and from what?**
The final result (Distinction/Pass/Fail/Withdrawn) and the mean score on assessments due *after* day 90, using only activity and assessment data up to day 90.

**How did you avoid leakage?**
(1) Features use only data up to the cut-off day; targets are in the future. (2) Students who had already unregistered by day 90 are excluded. (3) Imputers/scalers/encoders are inside pipelines fitted per CV fold. (4) Splits are *grouped by student* because 3,538 students enrolled in more than one presentation. (5) The model is chosen by CV, not by the test set.

**Why is accuracy only 57 %? Isn't that bad?**
Four classes, one of them (Withdrawn) driven by life events that clicks can't show. Baseline macro-F1 is 0.16, we get 0.50. For the practical question — "is this student at risk?" — ROC-AUC is 0.86 and average precision 0.83 (baseline 0.40).

**Why macro-F1?**
Classes are imbalanced (Withdrawn 12 %); accuracy rewards ignoring minority classes. KNN even has 0.60 accuracy but only 0.38 macro-F1.

**Which features matter most?**
Early assessment scores and submission rate; then module and engagement. Demographics alone give at-risk AUC 0.65, so they add little — good, because they are the ethically sensitive ones.

**How early can you warn?**
At course start AUC is 0.72; at day 30 it is 0.80; at day 90 it is 0.86. We use day 90 as a trade-off with ~170 days left to act.

**Silhouette is 0.19 — is clustering meaningless?**
It means behaviour is a continuum. But the personas are stable (ARI 0.75) and *validated on outcomes the clustering never saw*: the disengaged persona fails/withdraws 92 % of the time. k was restricted to 3–6 because k = 2 (silhouette 0.35) only splits active from inactive.

**The recommender barely beats popularity — why?**
Course sites unlock on a schedule, so what a student opens next is mostly determined by time, not taste. All models beat random (precision 0.150 vs 0.096). SVD has the best hit-rate (0.72 vs 0.67). We deploy the kNN + persona hybrid because it is explainable. A schedule-aware model is the obvious next step.

**How do you know the study plan helps students?**
We don't — that needs a live pilot/A-B test; offline data can't show learning gain. The plan is a transparent rule layer over ML outputs and the module's real assessment calendar.

**What does the fairness audit show?**
Recall is similar across gender and disability and higher for the most deprived students (0.73), so they are not missed. But false alarms are higher for the most deprived (22 % vs 9 %) and lower-education students. So flags must only *offer* support, never restrict.

**Ethical concerns?**
Labelling students as "at-risk" can stigmatise them; demographic variables can encode inequality; data are anonymised and must not be re-identified. Predictions go to tutors to offer help, not to grade.

**Why Random Forest / why not deep learning?**
Tabular data of ~20k rows: tree ensembles are typically as good as or better than deep nets, train in minutes and are explainable. Top four models differ by ≈ 0.01 macro-F1 — the limit is the information in the data.

**What would you do next?**
Pilot with tutors, survival models for time-to-withdrawal, weekly-sequence models, schedule-aware recommenders, SHAP explanations, per-group threshold calibration.

**Know your numbers:** 25,562 enrolments at day 90 · 34 features · RF tuned macro-F1 0.504 / at-risk AUC 0.858 · HGB regression RMSE 11.0, R² 0.55 · k = 5, ARI 0.75 · recommender hit-rate@10 SVD 0.72 vs popularity 0.67 vs random 0.55.
