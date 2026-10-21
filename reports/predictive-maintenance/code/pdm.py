"""Data pipeline, models and training loop for C-MAPSS RUL prediction."""
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DATA_DIR = __file__.rsplit('/code/', 1)[0] + '/data/CMAPSS'
COLS = ['unit', 'cycle', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
# Constant or near-constant in FD001/FD003, so they carry no degradation signal.
DROP = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']
SENSORS = [f's{i}' for i in range(1, 22) if f's{i}' not in DROP]


def read(name):
    return pd.read_csv(f'{DATA_DIR}/{name}', sep=r'\s+', header=None, names=COLS)


def load(subset, window=30, cap=125, val_frac=0.2, seed=0):
    """Returns train/val windows (all positions) and the last window of each test engine."""
    train, test = read(f'train_{subset}.txt'), read(f'test_{subset}.txt')
    rul_test = np.loadtxt(f'{DATA_DIR}/RUL_{subset}.txt')

    train['rul'] = train.groupby('unit')['cycle'].transform('max') - train['cycle']
    if cap is not None:
        train['rul'] = train['rul'].clip(upper=cap)

    units = train['unit'].unique()
    rng = np.random.default_rng(seed)
    val_units = rng.choice(units, int(len(units) * val_frac), replace=False)
    fit = train[~train['unit'].isin(val_units)]

    # Min-max scaling fitted on training engines only.
    lo, hi = fit[SENSORS].min(), fit[SENSORS].max()
    scale = lambda df: ((df[SENSORS] - lo) / (hi - lo + 1e-8)).values.astype(np.float32)

    def windows(df):
        X, y = [], []
        for _, g in df.groupby('unit'):
            v, r = scale(g), g['rul'].values
            for end in range(window, len(g) + 1):
                X.append(v[end - window:end]); y.append(r[end - 1])
        return np.stack(X), np.array(y, np.float32)

    def last_windows(df):
        X = []
        for _, g in df.groupby('unit'):
            v = scale(g)
            if len(v) < window:  # pad short series by repeating the first row
                v = np.vstack([np.repeat(v[:1], window - len(v), 0), v])
            X.append(v[-window:])
        return np.stack(X)

    Xtr, ytr = windows(fit)
    Xva, yva = windows(train[train['unit'].isin(val_units)])
    Xte = last_windows(test)
    yte = rul_test.astype(np.float32)
    if cap is not None:
        yte = np.minimum(yte, cap)
    return Xtr, ytr, Xva, yva, Xte, yte


# ---------------- metrics ----------------
def rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def phm_score(pred, true):
    d = pred - true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


# ---------------- models ----------------
class CNN1D(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_feat, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):  # x: (batch, time, features)
        return self.net(x.transpose(1, 2)).squeeze(-1)


class LSTMNet(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.l1 = nn.LSTM(n_feat, 64, batch_first=True)
        self.l2 = nn.LSTM(64, 32, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        h, _ = self.l1(x)
        h, _ = self.l2(h)
        return self.head(h[:, -1]).squeeze(-1)


class CNNLSTMAttention(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(n_feat, 64, 3, padding=1), nn.ReLU())
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        # Additive (Bahdanau-style) attention over time steps.
        self.att_w = nn.Linear(64, 64)
        self.att_v = nn.Linear(64, 1, bias=False)
        self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
        self.last_attention = None

    def forward(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.lstm(h)
        a = torch.softmax(self.att_v(torch.tanh(self.att_w(h))).squeeze(-1), dim=1)
        self.last_attention = a.detach()
        ctx = (a.unsqueeze(-1) * h).sum(1)
        return self.head(ctx).squeeze(-1)


MODELS = {'1D-CNN': CNN1D, 'LSTM': LSTMNet, 'CNN-LSTM + Attention': CNNLSTMAttention}


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def train_dl(name, data, seed, epochs=100, patience=10, batch=256, lr=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr, Xva, yva, Xte, yte = data
    model = MODELS[name](Xtr.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xva_t = torch.from_numpy(Xva)

    def predict(X):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(X[i:i + 4096]) for i in range(0, len(X), 4096)]).numpy()

    best, best_state, wait, hist = np.inf, None, 0, {'train': [], 'val': []}
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        tot = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        val_mse = float(np.mean((predict(Xva_t) - yva) ** 2))
        hist['train'].append(tot / len(perm)); hist['val'].append(val_mse)
        if val_mse < best:
            best, wait = val_mse, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    pred = np.clip(predict(torch.from_numpy(Xte)), 0, None)
    return {'rmse': rmse(pred, yte), 'score': phm_score(pred, yte), 'pred': pred.tolist(),
            'params': n_params(model), 'epochs': ep + 1, 'time_s': time.time() - t0,
            'history': hist, 'model': model}
