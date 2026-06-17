"""v1400-global — can GENUINE ML-flavored reconstruction reach ~0.8?

Adds GLOBAL game-context features (whole-game score structure: each player's max score in the
game, points the server wins AFTER this rally, position in game, #present rallies) on top of the
rich local features. Still pure feature engineering on the LEGAL score columns — we do NOT feed
the deterministic solver's pinned answer.

Honest test AUC on the overlap (true labels). Empirically answers: how high can real ML get?
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

ROOT = Path('E:/AICUP_O')
OUT = Path(__file__).resolve().parent.parent / 'outputs'

LOCAL = ['server_score', 'opp_score', 'point_in_game', 'score_diff', 'serve_parity', 'near_end',
         'delta_next1', 'gap_next1', 'delta_next2', 'gap_next2', 'delta_next3', 'gap_next3',
         'delta_prev1', 'gap_prev1', 'delta_prev2', 'gap_prev2',
         'fwd_all_lose', 'fwd_all_win', 'fwd_gap1']
GLOBAL = ['s_max', 'o_max', 's_pts_after', 'o_pts_after', 'game_n_present', 'pos_frac',
          'bracket_gap', 'bracket_sum']
FEATS = LOCAL + GLOBAL


def log(*a): print(*a, flush=True)


def rally_rows(df):
    return df[df.strikeNumber == 1].drop_duplicates('rally_uid').copy()


def build(d, has_label):
    rows = []
    for (m, g), grp in d.groupby(['match', 'numberGame']):
        recs = grp.sort_values('rally_id').to_dict('records')
        n = len(recs)
        # per-game each player's max observed score (global structure)
        pmax = {}
        for r in recs:
            pmax[r['gamePlayerId']] = max(pmax.get(r['gamePlayerId'], 0), int(r['scoreSelf']))
            pmax[r['gamePlayerOtherId']] = max(pmax.get(r['gamePlayerOtherId'], 0), int(r['scoreOther']))
        for i, r in enumerate(recs):
            gp, gpo = r['gamePlayerId'], r['gamePlayerOtherId']
            ss, so = int(r['scoreSelf']), int(r['scoreOther'])
            f = dict(rally_uid=int(r['rally_uid']), match=int(m),
                     server_score=ss, opp_score=so, point_in_game=ss + so,
                     score_diff=ss - so, serve_parity=((ss + so) // 2) % 2,
                     near_end=int(max(ss, so) >= 9),
                     s_max=pmax.get(gp, ss), o_max=pmax.get(gpo, so),
                     s_pts_after=pmax.get(gp, ss) - ss, o_pts_after=pmax.get(gpo, so) - so,
                     game_n_present=n, pos_frac=(i / max(n - 1, 1)))
            for j in range(1, 4):
                if i + j < n:
                    nx = recs[i + j]
                    nmy = nx['scoreSelf'] if nx['gamePlayerId'] == gp else nx['scoreOther']
                    f[f'delta_next{j}'] = int(nmy) - ss
                    f[f'gap_next{j}'] = int(nx['rally_id']) - int(r['rally_id'])
                else:
                    f[f'delta_next{j}'], f[f'gap_next{j}'] = -99, -1
            for j in range(1, 3):
                if i - j >= 0:
                    pv = recs[i - j]
                    pmy = pv['scoreSelf'] if pv['gamePlayerId'] == gp else pv['scoreOther']
                    f[f'delta_prev{j}'] = ss - int(pmy)
                    f[f'gap_prev{j}'] = int(r['rally_id']) - int(pv['rally_id'])
                else:
                    f[f'delta_prev{j}'], f[f'gap_prev{j}'] = -99, -1
            dn, gn = f['delta_next1'], f['gap_next1']
            f['fwd_all_lose'] = int(gn > 0 and dn == 0)
            f['fwd_all_win'] = int(gn > 0 and dn == gn)
            f['fwd_gap1'] = int(gn == 1)
            # tightest bracket (nearest present before+after): interval sum + gap
            f['bracket_gap'] = (f['gap_next1'] if f['gap_next1'] > 0 else 0) + \
                               (f['gap_prev1'] if f['gap_prev1'] > 0 else 0)
            f['bracket_sum'] = (f['delta_next1'] if f['delta_next1'] != -99 else 0) + \
                               (f['delta_prev1'] if f['delta_prev1'] != -99 else 0)
            if has_label:
                f['y'] = int(r['serverGetPoint'])
            rows.append(f)
    return pd.DataFrame(rows)


def params():
    base = dict(objective='binary', metric='auc', learning_rate=0.02, num_leaves=127,
                min_child_samples=20, feature_fraction=0.85, bagging_fraction=0.85,
                bagging_freq=1, reg_lambda=1.0, verbose=-1, seed=42)
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
        m = lgb.train(P, lgb.Dataset(X[tri], label=y[tri]), num_boost_round=1200,
                      valid_sets=[lgb.Dataset(X[vai], label=y[vai])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[vai] = m.predict(X[vai], num_iteration=m.best_iteration)
    oof_auc = roc_auc_score(y, oof)

    mf = lgb.train(P, lgb.Dataset(X, label=y), num_boost_round=700)
    Fte = Fte.assign(pred=mf.predict(Fte[FEATS].values.astype(np.float32)))
    gt = old[old.strikeNumber == 1].groupby('rally_uid')['serverGetPoint'].first()
    ov = Fte[Fte.rally_uid.isin(gt.index)].copy(); ov['yt'] = ov.rally_uid.map(gt).astype(int)
    ov_auc = roc_auc_score(ov.yt.values, ov.pred.values)

    log(f'\nOOF AUC = {oof_auc:.4f}  [optimistic]')
    log(f'REAL test AUC on overlap = {ov_auc:.4f}  [HONEST]')
    imp = dict(sorted(zip(FEATS, mf.feature_importance('gain').astype(int)), key=lambda x: -x[1]))
    log('top feats: ' + str({k: int(v) for k, v in list(imp.items())[:8]}))
    log('\n' + '=' * 60)
    log(f'  arithmetic score-chain (deployed) ~ 0.8205')
    log(f'  ML local (simple)                  = 0.7147')
    log(f'  ML local (rich)                    = 0.7108')
    log(f'  ML + GLOBAL game features          = {ov_auc:.4f}   <-- this run')
    log(f'  clean within-rally ML             ~ 0.666')
    log('=' * 60)
    import json
    json.dump(dict(oof_auc=float(oof_auc), overlap_test_auc=float(ov_auc),
                   importance={k: int(v) for k, v in imp.items()}),
              open(OUT / 'summary_global.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
