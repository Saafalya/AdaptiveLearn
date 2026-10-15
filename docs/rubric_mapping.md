# ESE rubric → evidence map (30 marks)

Source: *Detailed ESE Evaluation Rubric*, TE7483 – Applications and Use Cases of ML.

| Component | Marks | Rubric asks for ("Excellent") | Evidence in this project |
|---|---|---|---|
| Problem refinement & use-case relevance | 3 | objective, scope, domain, users/stakeholders, real-world relevance | `docs/report.md` §1: 5 objectives, stakeholders, scope, SDG 4 + 10 reasoning, real attrition problem (40 % fail/withdraw) |
| Dataset understanding, preprocessing, features | 4 | source, size, features, target, data types, missing values, preprocessing, feature selection/engineering, validation strategy | §2: **real OULAD** (cited, CC BY 4.0, MD5-verified), 7-table description, 34 features in 3 groups, targets defined, quality table, grouped CV + hold-out, leakage controls; `results/tables/data_describe.csv`, figs 01–03 |
| Methodology & ML pipeline | 5 | data prep, model selection, training strategy, evaluation method, **justification** | §3: pipeline diagram, algorithm ladder with rationale per task, evaluation plan (incl. temporal hold-out), study-plan rules |
| Implementation / execution | 5 | complete, aligned, clear and **reproducible** code and structure | modular `src/adaptivelearn/`, `run_all.py` one command, seeded, `requirements.txt`, unit tests, Streamlit app, processed tables committed |
| Results, metrics, comparative analysis | 5 | suitable metrics (accuracy, P/R/F1, confusion matrix, RMSE, MAE, ROC-AUC, clustering metrics) | §5: 8 classifiers, 6 regressors, 3 clustering algorithms, 6 recommenders; ablation, early-prediction sweep, fairness audit; figs 04–14; `results/tables/*.csv` |
| Final report & documentation | 3 | problem, methodology, dataset, results, conclusion, **future scope**, references | `docs/report.md` §1–6 + ethics + references; README |
| Communication & presentation | 2 | clear, structured, time-managed, visuals | `docs/presentation_outline.md` (10 slides, timings) + live app |
| Viva / project defense | 3 | deep **individual** understanding incl. limitations and future scope | `docs/viva_prep.md`; every member must be able to answer all |

## Self-check before submission
- [ ] Fill in Member 4's name and PRN in `docs/report.md` and `README.md`.
- [ ] Convert `docs/report.md` to the format faculty require (PDF/DOCX) — ask what that is.
- [ ] Re-run `python run_all.py` on the demo laptop (needs `data/oulad_raw/`) and open the app once beforehand.
- [ ] Every member can explain: why the cut-off day, why grouped splits, why macro-F1, why Withdrawn is hard, why recommender gains are small, what the fairness audit found.
