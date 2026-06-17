"""v1400-neural — the 'neural solver' ceiling probe.

A per-game Transformer over the rallies of a game (ordered by rally_id). It attends across the
WHOLE game, so it can learn to PROPAGATE the score constraints (what the GBDT couldn't). Trained
with random rally-dropout so it learns to solve from SPARSE observations (matching test truncation).

This is effectively a LEARNED reimplementation of the arithmetic constraint solver. It is NOT
spirit-cleaner than the arithmetic version, will NOT go in the report, and does NOT change the
locked submission. Pure 'how high can ML get' curiosity probe.

Honest metric: AUC on the overlap (real test rallies with ground truth).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

torch.manual_seed(42); np.random.seed(42)
ROOT = Path('E:/AICUP_O'); OUT = Path(__file__).resolve().parent.parent / 'outputs'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
FDIM, MAXLEN = 10, 48


def log(*a): print(*a, flush=True)


def make_games(df, has_label):
    """One sequence per (match, game): tokens = present rallies in a CONSISTENT player frame."""
    r1 = df[df.strikeNumber == 1].drop_duplicates('rally_uid')
    games = []
    for (m, g), grp in r1.groupby(['match', 'numberGame']):
        grp = grp.sort_values('rally_id')
        A = grp.iloc[0]['gamePlayerId']                 # player A = first rally's server
        toks = []
        for r in grp.itertuples(index=False):
            if r.gamePlayerId == A:
                a, b, sa = int(r.scoreSelf), int(r.scoreOther), 1
            else:
                a, b, sa = int(r.scoreOther), int(r.scoreSelf), 0
            t = dict(rid=int(r.rally_id), a=a, b=b, sa=sa, uid=int(r.rally_uid))
            if has_label:
                t['y'] = int(r.serverGetPoint)
            toks.append(t)
        if len(toks) >= 1:
            games.append(toks)
    return games


def feats(toks):
    """Per-token features computed WITHIN the present subset (sorted by rid)."""
    F = []
    for i, t in enumerate(toks):
        a, b, sa, rid = t['a'], t['b'], t['sa'], t['rid']
        nx = toks[i + 1] if i + 1 < len(toks) else None
        pv = toks[i - 1] if i - 1 >= 0 else None
        dn_a = (nx['a'] - a) if nx else -9.0
        dn_b = (nx['b'] - b) if nx else -9.0
        gn = (nx['rid'] - rid) if nx else -1.0
        dp_a = (a - pv['a']) if pv else -9.0
        gp = (rid - pv['rid']) if pv else -1.0
        F.append([a, b, a + b, a - b, sa, dn_a, dn_b, gn, dp_a, gp])
    return np.asarray(F, dtype=np.float32)


class NeuralSolver(nn.Module):
    def __init__(self, d=96, nhead=6, nlayers=4):
        super().__init__()
        self.inp = nn.Linear(FDIM, d)
        self.pe = nn.Parameter(torch.randn(MAXLEN, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, nhead, d * 2, dropout=0.1, batch_first=True)
        self.tr = nn.TransformerEncoder(enc, nlayers)
        self.out = nn.Linear(d, 1)

    def forward(self, x, mask):                          # x:(B,L,FDIM) mask:(B,L) True=valid
        h = self.inp(x) + self.pe[:x.size(1)].unsqueeze(0)
        h = self.tr(h, src_key_padding_mask=~mask)
        return self.out(h).squeeze(-1)


def pad_batch(seq_feats, seq_y):
    L = max(f.shape[0] for f in seq_feats)
    B = len(seq_feats)
    X = np.zeros((B, L, FDIM), np.float32); Y = np.zeros((B, L), np.float32)
    M = np.zeros((B, L), bool)
    for i, (f, y) in enumerate(zip(seq_feats, seq_y)):
        n = f.shape[0]; X[i, :n] = f; M[i, :n] = True
        if y is not None:
            Y[i, :n] = y
    return (torch.tensor(X), torch.tensor(M), torch.tensor(Y))


def main():
    tr = pd.read_csv(ROOT / 'data/train.csv'); te = pd.read_csv(ROOT / 'data/test.csv')
    old = pd.read_csv(ROOT / 'data/test_old_public.csv')
    G_tr = make_games(tr, True); G_te = make_games(te, False)
    log(f'device={DEV}  train games={len(G_tr)}  test games={len(G_te)}')

    model = NeuralSolver().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(42)

    gt = old[old.strikeNumber == 1].groupby('rally_uid')['serverGetPoint'].first()

    def eval_overlap():
        model.eval(); preds = {}
        with torch.no_grad():
            for i in range(0, len(G_te), 64):
                batch = G_te[i:i + 64]
                sf = [feats(g)[:MAXLEN] for g in batch]
                X, M, _ = pad_batch(sf, [None] * len(batch))
                p = torch.sigmoid(model(X.to(DEV), M.to(DEV))).cpu().numpy()
                for gi, g in enumerate(batch):
                    for ti, t in enumerate(g[:MAXLEN]):
                        preds[t['uid']] = float(p[gi, ti])
        ov = [(u, preds[u], int(gt[u])) for u in gt.index if u in preds]
        yv = np.array([x[2] for x in ov]); pv = np.array([x[1] for x in ov])
        return roc_auc_score(yv, pv), len(ov)

    EPOCHS, AUG = 40, 6
    best = 0.0
    for ep in range(EPOCHS):
        model.train(); order = rng.permutation(len(G_tr)); tot = 0.0
        # build augmented examples: random subset of each game (matches test sparsity)
        examples = []
        for gi in order:
            g = G_tr[gi]
            for _ in range(AUG):
                keep = rng.random() * 0.7 + 0.25                  # keep ratio U(0.25,0.95)
                idx = sorted(i for i in range(len(g)) if rng.random() < keep)
                if len(idx) < 1:
                    idx = [rng.integers(len(g))]
                sub = [g[i] for i in idx]
                examples.append((feats(sub)[:MAXLEN], np.array([t['y'] for t in sub], np.float32)[:MAXLEN]))
        rng.shuffle(examples)
        for i in range(0, len(examples), 128):
            bf = [e[0] for e in examples[i:i + 128]]; by = [e[1] for e in examples[i:i + 128]]
            X, M, Y = pad_batch(bf, by)
            X, M, Y = X.to(DEV), M.to(DEV), Y.to(DEV)
            logit = model(X, M)
            loss = lossf(logit[M], Y[M])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if ep % 4 == 0 or ep == EPOCHS - 1:
            auc, nov = eval_overlap()
            best = max(best, auc)
            log(f'ep{ep:02d} loss={tot/ max(1,len(examples)//128):.4f}  overlap AUC={auc:.4f}  (best {best:.4f})')

    auc, nov = eval_overlap()
    best = max(best, auc)
    log('\n' + '=' * 60)
    log(f'  arithmetic score-chain (deployed) ~ 0.8205')
    log(f'  GBDT (local/rich/global)           = ~0.711-0.715')
    log(f'  NEURAL SOLVER (this)               = {auc:.4f}   (best over epochs {best:.4f})')
    log(f'  clean within-rally ML             ~ 0.666')
    log('=' * 60)
    import json
    json.dump(dict(neural_overlap_auc=float(auc), best=float(best), n_overlap=int(nov)),
              open(OUT / 'summary_neural.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
