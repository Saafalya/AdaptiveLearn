# Viva preparation – likely questions

**Why is the dataset synthetic? Isn't that a weakness?**
No public dataset has per-topic scores, behaviour logs *and* resource feedback together. A seeded simulator lets us control and document ground truth, inject realistic missingness/outliers, and check that the pipeline recovers known structure. The weakness is real: absolute numbers don't transfer to real learners. The loader accepts the same column names, so real LMS data can be dropped in.

**Why macro-F1 rather than accuracy?**
Classes are imbalanced (54 / 26 / 20 %). The majority-class baseline already gets 0.54 accuracy but only 0.23 macro-F1, so accuracy would flatter weak models.

**How did you avoid data leakage?**
Imputers, scalers and encoders live inside sklearn `Pipeline`s fitted on training folds only; engineered features are row-wise; the final score and level are never model inputs; the model was selected by 5-fold CV on the training set, not on the test set.

**Random Forest vs Gradient Boosting vs Logistic Regression – which is "best"?**
Top models are within one CV standard deviation (0.80–0.81 macro-F1), so differences are not significant. We chose Random Forest (tuned) on CV score. Logistic Regression is nearly as good and far more interpretable — a valid production choice.

**Why did linear models win for regression?**
The simulator builds the score as a noisy linear combination of the inputs; extra capacity can't help. On real data we'd expect more of a gap for ensembles.

**Why did SVD collaborative filtering lose to popularity?**
Only 15 ratings per student (28 % density) and ratings are driven by the match between mastery and resource difficulty — a feature-level effect, not a latent-taste effect. Content features capture it directly; the low-rank structure SVD needs is mostly absent.

**What about a new student with no history (cold start)?**
The content-based model uses student features, not a user id, so it works immediately; SVD cannot.

**Silhouette is only 0.19 – is clustering meaningless?**
No – it means the population is a continuum. The clusters are still stable (ARI 0.97), interpretable, and align with level, so they are useful *segments*. We also report k = 2 had a higher silhouette (0.28) but gives only "good/bad", which is not actionable; we restricted k to 3–6 and say so.

**How is the learning path produced – is it ML or rules?**
Hybrid. ML predicts level, topic-fit ratings and persona; a transparent rule layer (thresholds, prerequisite graph, checkpoint-advance rule) turns that into an ordered path. Rules keep it explainable to students and teachers.

**How does this support SDG 10?**
Fairness slice by internet access; format-aware recommendations (practice/reading are lighter than video); the same model quality across access groups (0.79–0.85 macro-F1; small n for "limited").

**How would you evaluate it in the real world?**
A/B test: path-followers vs control, measuring learning gain (post – pre score), completion and retention; monitor fairness metrics continuously.

**What would you improve next?**
Knowledge tracing (BKT/DKT) for time-aware mastery, bandits for online learning from feedback, SHAP explanations, real LMS data.

**Know your numbers:** 1,500 students · 22 features · 54 resources · 22,500 ratings · RF macro-F1 0.809 / AUC 0.944 · Lasso RMSE 4.37, R² 0.876 · k = 3, ARI 0.97 · Content GBM RMSE 0.705 vs popularity 0.998.
