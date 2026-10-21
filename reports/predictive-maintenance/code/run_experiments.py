"""Runs every experiment in the report and writes results/results.json plus all figures.

Usage: python run_experiments.py   (about 30-40 min on a 4-core CPU)
"""
import json
import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import pdm

torch.set_num_threads(4)
OUT = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG = os.path.join(os.path.dirname(__file__), '..')
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
ABL_SEEDS = [0, 1, 2]
plt.rcParams.update({'font.family': 'serif', 'font.size': 11})


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def baselines(data, seed):
    Xtr, ytr, _, _, Xte, yte = data
    flat = lambda X: X.reshape(len(X), -1)
    out = {}
    t0 = time.time()
    lr = LinearRegression().fit(flat(Xtr), ytr)
    p = np.clip(lr.predict(flat(Xte)), 0, None)
    out['Linear Regression'] = {'rmse': pdm.rmse(p, yte), 'score': pdm.phm_score(p, yte), 'params': int(lr.coef_.size + 1), 'time_s': time.time() - t0}
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=4, random_state=seed).fit(flat(Xtr), ytr)
    p = np.clip(rf.predict(flat(Xte)), 0, None)
    out['Random Forest'] = {'rmse': pdm.rmse(p, yte), 'score': pdm.phm_score(p, yte), 'params': None, 'time_s': time.time() - t0}
    return out


def summarise(runs):
    r = np.array([x['rmse'] for x in runs]); s = np.array([x['score'] for x in runs])
    return {'rmse_mean': r.mean(), 'rmse_std': r.std(), 'score_mean': s.mean(), 'score_std': s.std(),
            'params': runs[0]['params'], 'time_s_mean': float(np.mean([x['time_s'] for x in runs])),
            'epochs_mean': float(np.mean([x.get('epochs', 0) for x in runs])), 'runs': [{'rmse': x['rmse'], 'score': x['score']} for x in runs]}


results = {'main': {}, 'ablation': []}
keep = {}  # FD001 seed-0 proposed model, for figures

# ---------- main comparison ----------
for subset in ['FD001', 'FD003']:
    data = pdm.load(subset)
    log(subset, 'train/val/test windows:', len(data[0]), len(data[2]), len(data[4]))
    per_model = {}
    lr_runs = None
    for seed in SEEDS:
        b = baselines(data, seed)
        if lr_runs is None:
            lr_runs = [b['Linear Regression']]  # deterministic, one run is enough
        per_model.setdefault('Random Forest', []).append(b['Random Forest'])
        log(subset, 'RF seed', seed, round(b['Random Forest']['rmse'], 2))
    per_model['Linear Regression'] = lr_runs
    for name in pdm.MODELS:
        for seed in SEEDS:
            r = pdm.train_dl(name, data, seed)
            log(subset, name, 'seed', seed, 'rmse', round(r['rmse'], 2), 'score', round(r['score'], 1), 'epochs', r['epochs'])
            if subset == 'FD001' and seed == 0 and name == 'CNN-LSTM + Attention':
                keep = {'model': r['model'], 'pred': np.array(r['pred']), 'true': data[5], 'history': r['history'], 'data': data}
            r.pop('model')
            per_model.setdefault(name, []).append(r)
    results['main'][subset] = {k: summarise(v) for k, v in per_model.items()}
    json.dump(results, open(f'{OUT}/results.json', 'w'), indent=1)

# ---------- ablation: window length and RUL cap ----------
for window, cap in [(20, 125), (30, 125), (40, 125), (30, 115), (30, 130), (30, None)]:
    data = pdm.load('FD001', window=window, cap=cap)
    _, _, _, _, Xte, _ = data
    true_uncapped = np.loadtxt(f'{pdm.DATA_DIR}/RUL_FD001.txt')
    runs = []
    for seed in ABL_SEEDS:
        r = pdm.train_dl('CNN-LSTM + Attention', data, seed)
        # Score every setting against the same target (true RUL capped at 125) so rows are comparable.
        ref = np.minimum(true_uncapped, 125)
        p = np.array(r['pred'])
        runs.append({'rmse': pdm.rmse(p, ref), 'score': pdm.phm_score(p, ref), 'params': r['params'], 'time_s': r['time_s'], 'epochs': r['epochs']})
    s = summarise(runs); s.update(window=window, cap=cap)
    results['ablation'].append(s)
    log('ablation', window, cap, round(s['rmse_mean'], 2), round(s['score_mean'], 1))
    json.dump(results, open(f'{OUT}/results.json', 'w'), indent=1)

# ---------- figures (FD001, proposed model, seed 0) ----------
model, pred, true, hist, data = keep['model'], keep['pred'], keep['true'], keep['history'], keep['data']
o = np.argsort(true)
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(true[o], 'o-', ms=3, label='Actual RUL', color='#333')
ax.plot(pred[o], 's', ms=3, label='Predicted RUL (CNN-LSTM-Att)', color='#8B0000', alpha=.7)
ax.set_xlabel('Test engine (sorted by actual RUL)'); ax.set_ylabel('RUL (cycles)'); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f'{FIG}/fig_pred_vs_actual.png', dpi=160)

fig, ax = plt.subplots(figsize=(8, 3.8))
e = np.arange(1, len(hist['train']) + 1)
ax.plot(e, hist['train'], label='Training', color='#333'); ax.plot(e, hist['val'], label='Validation', color='#8B0000')
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE loss'); ax.set_yscale('log'); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f'{FIG}/fig_loss_curves.png', dpi=160)

# Full-life trajectory of one held-out validation engine (validation engines run to failure).
train = pdm.read('train_FD001.txt')
units = train['unit'].unique()
val_units = np.random.default_rng(0).choice(units, int(len(units) * 0.2), replace=False)
Xtr, ytr, Xva, yva, _, _ = data
g = train[train['unit'] == val_units[0]]
n = len(g) - 30 + 1
start = sum(len(train[train['unit'] == u]) - 30 + 1 for u in train['unit'].unique() if u in val_units and u < val_units[0])
model.eval()
with torch.no_grad():
    vp = model(torch.from_numpy(Xva[start:start + n])).numpy()
    model(torch.from_numpy(Xva))  # fills last_attention for every validation window
    att_w = model.last_attention.numpy()
cyc = g['cycle'].values[29:]
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(cyc, yva[start:start + n], label='Actual RUL (capped at 125)', color='#333', lw=2)
ax.plot(cyc, vp, label='Predicted RUL', color='#8B0000', alpha=.85)
ax.set_xlabel('Operating cycle'); ax.set_ylabel('RUL (cycles)'); ax.set_title(f'Validation engine {val_units[0]}', fontsize=10)
ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f'{FIG}/fig_engine_trajectory.png', dpi=160)

mean_att = att_w.mean(0)
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.bar(np.arange(-29, 1), mean_att, color='#8B0000')
ax.set_xlabel('Time step within window (0 = current cycle)'); ax.set_ylabel('Mean attention weight'); ax.grid(alpha=.3, axis='y')
fig.tight_layout(); fig.savefig(f'{FIG}/fig_attention.png', dpi=160)

results['figures'] = {'val_engine': int(val_units[0]), 'attention_last5_share': float(mean_att[-5:].sum()),
                      'attention_mean': mean_att.tolist(),
                      'fd001_seed0_test_abs_err_mean': float(np.mean(np.abs(pred - true))),
                      'fd001_seed0_late_share': float(np.mean(pred > true)),
                      'fd001_seed0_err_high_rul': float(np.sqrt(np.mean((pred[true >= 100] - true[true >= 100]) ** 2))),
                      'fd001_seed0_err_low_rul': float(np.sqrt(np.mean((pred[true < 50] - true[true < 50]) ** 2)))}
json.dump(results, open(f'{OUT}/results.json', 'w'), indent=1)
log('done')
