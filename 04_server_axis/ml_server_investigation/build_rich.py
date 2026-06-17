"""v1400-rich — push the ML-on-score server as high as GENUINE features allow.

Adds richer score-structure features (up to 3 forward + 2 backward present neighbors, immediate
forward-interval all-win/all-lose flags, parity, near-game-end). Still LEGITIMATE feature
engineering on the legal score columns. HARD LINE: we do NOT feed the deterministic solver's
pinned answer as a feature (that would be arithmetic-in-ML-clothing, not a learned model).

Honest test AUC measured on the overlap (true labels). Compare to score-chain 0.8205 / clean 0.666.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

ROOT = Path('E:/AICUP_O')
OUT = Path(__file__).resolve().parent.parent / 'outputs'

FEATS = ['server_score', 'opp_score', 'point_in_game', 'score_diff', 'serve_parity', 'near_end',
         'delta_next1', 'gap_next1', 'delta_next2', 'gap_next2', 'delta_next3', 'gap_next3',
         'delta_prev1', 'gap_prev1', 'delta_prev2', 'gap_prev2',
         'fwd_all_lose', 'fwd_all_win', 'fwd_gap1']


def log(*a): print(*a, flush=True)


def rally_rows(df):
    return df[df.strikeNumber == 1].drop_duplicates('rally_uid').copy()


def build(d, has_label):
    rows = []
    for (m, g), grp in d.groupby(['match', 'numberGame']):
        recs = grp.sort_values('rally_id').to_dict('records')
        n = len(recs)
        for i, r in enumerate(recs):
            gp = r['gamePlayerId']; ss, so = int(r['scoreSelf']), int(r['scoreOther'])
            f = dict(rally_uid=int(r['rally_uid']), match=int(m),
                     server_score=ss, opp_score=so, point_in_game=ss + so,
                     score_diff=ss - so, serve_parity=((ss + so) // 2) % 2,
                     near_end=int(max(ss, so) >= 9))
            for j in range(1, 4):                          # forward present neighbors
                if i + j < n:
                    nx = recs[i + j]
                    nmy = nx['scoreSelf'] if nx['gamePlayerId'] == gp else nx['scoreOther']
                    f[f'delta_next{j}'] = int(nmy) - ss
                    f[f'gap_next{j}'] = int(nx['rally_id']) - int(r['rally_id'])
                else:
                    f[f'delta_next{j}'], f[f'gap_next{j}'] = -99, -1
            for j in range(1, 3):                          # backward present neighbors
                if i - j >= 0:
                    pv = recs[i - j]
                    pmy = pv['scoreSelf'] if pv['gamePlayerId'] == gp else pv['scoreOther']
                    f[f'delta_prev{j}'] = ss - int(pmy)
                    f[f'gap_prev{j}'] = int(r['rally_id']) - int(pv['rally_id'])
                else:
                    f[f'delta_prev{j}'], f[f'gap_prev{j}'] = -99, -1
            dn, gn = f['delta_next1'], f['gap_next1']       # immediate forward interval flags
            f['fwd_all_lose'] = int(gn > 0 and dn == 0)
            f['fwd_all_win'] = int(gn > 0 and dn == gn)
            f['fwd_gap1'] = int(gn == 1)
            if has_label:
                f['y'] = int(r['serverGetPoint'])
            rows.append(f)
    return pd.DataFrame(rows)


def params():
    base = dict(objective='binary', metric='auc', learning_rate=0.03, num_leaves=63,
                min_child_samples=30, feature_fraction=0.85, bagging_fraction=0.85,
                bagging_freq=1, reg_lambda=2.0, verbose=-1, seed=42)
    try:
        gp = dict(base, device='gpu', gpu_platform_id=0, gpu_device_id=0)
        lgb.train(gp, lgb.Dataset(np.random.rand(50, 3), label=(np.random.rand(50) > .5).astype(int)),
                  num_boost_round=1)
        log('LightGBM GPU: OK'); return gp
    except Exception as e:
        log(f'GPU off ({type(e).__name__}); CPU.'); return base


def main():
    tr = pd.read_csv(ROOT / 'data/train.csv'); te = pd.read_csv(ROOT / 'data/test.csv')
    old = pd.read_csv(ROOT / 'data/test_old_public.csv')
    Ftr = build(rally_rows(tr), True); Fte = build(rally_rows(te), False)
    X = Ftr[FEATS].values.astype(np.float32); y = Ftr['y'].values.astype(int)
    groups = Ftr['match'].values; P = params()

    oof = np.zeros(len(y))
    for tri, vai in GroupKFold(5).split(X, y, groups):
        m = lgb.train(P, lgb.Dataset(X[tri], label=y[tri]), num_boost_round=800,
                      valid_sets=[lgb.Dataset(X[vai], label=y[vai])],
                      callbacks=[lgb.early_stopping(80, verbose=False)])
        oof[vai] = m.predict(X[vai], num_iteration=m.best_iteration)
    oof_auc = roc_auc_score(y, oof)

    mf = lgb.train(P, lgb.Dataset(X, label=y), num_boost_round=500)
    Fte = Fte.assign(pred=mf.predict(Fte[FEATS].values.astype(np.float32)))
    gt = old[old.strikeNumber == 1].groupby('rally_uid')['serverGetPoint'].first()
    ov = Fte[Fte.rally_uid.isin(gt.index)].copy(); ov['yt'] = ov.rally_uid.map(gt).astype(int)
    ov_auc = roc_auc_score(ov.yt.values, ov.pred.values)

    log(f'\nOOF AUC (GroupKFold match) = {oof_auc:.4f}  [optimistic: train games complete]')
    log(f'REAL test AUC on overlap   = {ov_auc:.4f}  [HONEST number]')
    imp = dict(sorted(zip(FEATS, mf.feature_importance('gain').astype(int)), key=lambda x: -x[1]))
    log('top feats: ' + str({k: int(v) for k, v in list(imp.items())[:6]}))
    log('\n' + '=' * 60)
    log(f'  score-chain (deployed, arithmetic) ~ 0.8205')
    log(f'  simple ML (prev run)                = 0.7147')
    log(f'  THIS rich ML                        = {ov_auc:.4f}')
    log(f'  clean within-rally ML              ~ 0.666')
    log('=' * 60)
    import json
    json.dump(dict(oof_auc=float(oof_auc), overlap_test_auc=float(ov_auc),
                   importance={k: int(v) for k, v in imp.items()}),
              open(OUT / 'summary_rich.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
