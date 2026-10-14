# ESE rubric → evidence map (30 marks)

Source: *Detailed ESE Evaluation Rubric*, TE7483 – Applications and Use Cases of ML.
"Excellent" (100 %) language is quoted from the rubric; the right column says where this project provides it.

| Component | Marks | Rubric asks for ("Excellent") | Evidence in this project |
|---|---|---|---|
| Problem refinement & use-case relevance | 3 | objective, scope, domain, users/stakeholders, real-world relevance | `docs/report.md` §1: objective list, stakeholders, scope boundaries, SDG 4 + 10 reasoning |
| Dataset understanding, preprocessing, features | 4 | source, size, features, target, data types, missing values, preprocessing, feature selection/engineering, validation strategy | §2: provenance (+ honest synthetic note), size table, missing/outlier table, engineered features, permutation importance, CV + hold-out strategy, leakage controls. Files: `data/*.csv`, `results/tables/data_describe.csv`, figs 01–03 |
| Methodology & ML pipeline | 5 | data prep, model selection, training strategy, evaluation method, **justification** of algorithms | §3: pipeline diagram, per-task algorithm table with rationale, evaluation plan, learning-path rules |
| Implementation / execution | 5 | complete, aligned, clear and **reproducible** code & structure | modular `src/adaptivelearn/`, `run_all.py` (one command), seeded, `requirements.txt`, unit tests, Streamlit demo |
| Results, metrics, comparative analysis | 5 | suitable metrics (accuracy, P/R/F1, confusion matrix, RMSE, MAE, ROC-AUC, clustering metrics) | §5: 9 classifiers, 8 regressors, 3 clustering algorithms, 6 recommenders, ablation, fairness; figs 04–12; `results/tables/*.csv` |
| Final report & documentation | 3 | problem, methodology, dataset, results, conclusion, **future scope**, references | `docs/report.md` §1–6 + references; this README |
| Communication & presentation | 2 | clear, structured, time-managed, visuals | `docs/presentation_outline.md` (10 slides, timings) + live app |
| Viva / project defense | 3 | deep **individual** understanding incl. limitations & future scope | `docs/viva_prep.md` – likely questions with answers; every member should be able to answer all |

## Self-check before submission
- [ ] Replace team/PRN placeholders in `docs/report.md`.
- [ ] Convert `docs/report.md` to the required submission format (PDF/DOCX) if the faculty specifies one.
- [ ] Re-run `python run_all.py` on the demo laptop; open the Streamlit app once beforehand.
- [ ] Every member can explain: why synthetic data, why macro-F1, why Random Forest was chosen on CV not test, why SVD lost, what silhouette 0.19 means.
