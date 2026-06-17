"""v1400-neural ENSEMBLE — push the neural solver toward 0.8 via multi-seed rank-average.

Same per-game Transformer 'neural solver' as build_neural_solver.py, trained with 5 seeds and
probability-averaged. Definitive answer to 'can ML-flavored reconstruction reach 0.8?'.
Still a learned reimplementation of the arithmetic solver -> NOT in report, does NOT change submission.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

ROOT = Path('E:/AICUP_O'); OUT = Path(__file__).resolve().parent.parent / 'outputs'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
FDIM, MAXLEN = 10, 48
SEEDS = [42, 7, 13, 101, 202]


def log(*a): print(*a, flush=True)


def make_games(df, has_label):
    r1 = df[df.strikeNumber == 1].drop_duplicates('rally_uid'); games = []
    for (m, g), grp in r1.groupby(['match', 'numberGame']):
        grp = grp.sort_values('rally_id'); A = grp.iloc[0]['gamePlayerId']; toks = []
        for r in grp.itertuples(index=False):
            a, b, sa = (int(r.scoreSelf), int(r.scoreOther), 1) if r.gamePlayerId == A \
                else (int(r.scoreOther), int(r.scoreSelf), 0)
            t = dict(rid=int(r.rally_id), a=a, b=b, sa=sa, uid=int(r.rally_uid))
            if has_label:
                t['y'] = int(r.serverGetPoint)
            toks.append(t)
        if toks:
            games.append(toks)
    return games


def feats(toks):
    F = []
    for i, t in enumerate(toks):
        a, b, sa, rid = t['a'], t['b'], t['sa'], t['rid']
        nx = toks[i + 1] if i + 1 < len(toks) else None
        pv = toks[i - 1] if i - 1 >= 0 else None
        F.append([a, b, a + b, a - b, sa,
                  (nx['a'] - a) if nx else -9.0, (nx['b'] - b) if nx else -9.0,
                  (nx['rid'] - rid) if nx else -1.0,
                  (a - pv['a']) if pv else -9.0, (rid - pv['rid']) if pv else -1.0])
    return np.asarray(F, dtype=np.float32)


class NeuralSolver(nn.Module):
    def __init__(self, d=96, nhead=6, nlayers=4):
        super().__init__()
        self.inp = nn.Linear(FDIM, d)
        self.pe = nn.Parameter(torch.randn(MAXLEN, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, nhead, d * 2, dropout=0.1, batch_first=True)
        self.tr = nn.TransformerEncoder(enc, nlayers); self.out = nn.Linear(d, 1)

    def forward(self, x, mask):
        h = self.inp(x) + self.pe[:x.size(1)].unsqueeze(0)
        return self.out(self.tr(h, src_key_padding_mask=~mask)).squeeze(-1)


def pad(sf, sy):
    L = max(f.shape[0] for f in sf); B = len(sf)
    X = np.zeros((B, L, FDIM), np.float32); Y = np.zeros((B, L), np.float32); M = np.zeros((B, L), bool)
    for i, (f, y) in enumerate(zip(sf, sy)):
        n = f.shape[0]; X[i, :n] = f; M[i, :n] = True
        if y is not None:
            Y[i, :n] = y
    return torch.tensor(X), torch.tensor(M), torch.tensor(Y)


def train_one(seed, G_tr, G_te):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = NeuralSolver().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    for ep in range(34):
        model.train(); ex = []
        for g in G_tr:
            for _ in range(6):
                keep = rng.random() * 0.7 + 0.25
                idx = sorted(i for i in range(len(g)) if rng.random() < keep) or [int(rng.integers(len(g)))]
                sub = [g[i] for i in idx]
                ex.append((feats(sub)[:MAXLEN], np.array([t['y'] for t in sub], np.float32)[:MAXLEN]))
        rng.shuffle(ex)
        for i in range(0, len(ex), 128):
            X, M, Y = pad([e[0] for e in ex[i:i+128]], [e[1] for e in ex[i:i+128]])
            X, M, Y = X.to(DEV), M.to(DEV), Y.to(DEV)
            loss = lossf(model(X, M)[M], Y[M])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); preds = {}
    with torch.no_grad():
        for i in range(0, len(G_te), 64):
            batch = G_te[i:i+64]
            X, M, _ = pad([feats(g)[:MAXLEN] for g in batch], [None]*len(batch))
            p = torch.sigmoid(model(X.to(DEV), M.to(DEV))).cpu().numpy()
            for gi, g in enumerate(batch):
                for ti, t in enumerate(g[:MAXLEN]):
                    preds[t['uid']] = float(p[gi, ti])
    return preds


def main():
    tr = pd.read_csv(ROOT / 'data/train.csv'); te = pd.read_csv(ROOT / 'data/test.csv')
    old = pd.read_csv(ROOT / 'data/test_old_public.csv')
    G_tr, G_te = make_games(tr, True), make_games(te, False)
    gt = old[old.strikeNumber == 1].groupby('rally_uid')['serverGetPoint'].first()
    log(f'device={DEV}  seeds={SEEDS}')

    allp = []
    for s in SEEDS:
        p = train_one(s, G_tr, G_te)
        ov = [(u, p[u], int(gt[u])) for u in gt.index if u in p]
        a = roc_auc_score([x[2] for x in ov], [x[1] for x in ov])
        log(f'  seed {s}: overlap AUC = {a:.4f}')
        allp.append(p)

    uids = list(gt.index)
    ens = {u: np.mean([p[u] for p in allp if u in p]) for u in uids if any(u in p for p in allp)}
    yv = np.array([int(gt[u]) for u in ens]); pv = np.array([ens[u] for u in ens])
    ens_auc = roc_auc_score(yv, pv)
    log('\n' + '=' * 60)
    log(f'  arithmetic score-chain (deployed) ~ 0.8205')
    log(f'  GBDT (feature-engineered)          = ~0.711-0.715')
    log(f'  neural solver (single)             = ~0.776-0.782')
    log(f'  NEURAL SOLVER ENSEMBLE (5-seed)    = {ens_auc:.4f}   <-- this')
    log(f'  clean within-rally ML             ~ 0.666')
    log('=' * 60)
    import json
    json.dump(dict(ensemble_overlap_auc=float(ens_auc), seeds=SEEDS),
              open(OUT / 'summary_neural_ensemble.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
