"""v3 training — extends v2 ensemble with:
1. Serve-class mask for action (argmax only over classes 0..14).
2. Two-stage point head (binary terminal + 9-class zone).
3. Class-inverse-frequency sample weights on top of context-len weights —
   macro-F1 benefits from up-weighting rare classes.

Models (all GPU where possible): LightGBM (CPU) + XGBoost (CUDA) +
CatBoost (GPU).
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score

from . import config as C
from .features import load_raw, build_targets
from .matchup_features import (
    build_matchup_transitions_v23, compute_matchup_features_v23,
    MT_COL_NAMES,
)
from .postprocess import (
    posthoc_logit_adjust, tune_tau,
    coordinate_descent_thresholds, apply_shifts, macro_f1,
)


SERVER_DROP = ['context_len', 'is_server_turn']


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _pick_random_target_per_rally(rally_ids, strike_ids, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({'r': rally_ids, 's': strike_ids,
                       'idx': np.arange(len(rally_ids))})
    out = []
    for _, sub in df.groupby('r', sort=False):
        out.append(sub['idx'].iloc[rng.integers(0, len(sub))])
    return np.array(out, dtype=int)


def _class_inverse_weights(y, n_classes, base=1.0, power=0.5):
    """Return per-sample weights: sqrt(median_freq / freq) so rare classes
    up-weighted but not extremely. Combined multiplicatively with the
    existing context-len weights."""
    cnts = np.bincount(y, minlength=n_classes).astype(np.float64)
    cnts = np.maximum(cnts, 1)
    inv = (np.median(cnts) / cnts) ** power
    inv = inv / inv.mean()  # mean-1 normalise
    return base * inv[y].astype(np.float32)


# ---------------------------------------------------------------------------
# Single-model trainers (same as v2 but with explicit weights param)
# ---------------------------------------------------------------------------

def _lgb_multiclass(X_tr, y_tr, w_tr, X_va, y_va, w_va, n_class, cat_cols,
                    num_round=None, early_stop=None):
    num_round = num_round or C.LGB_NUM_ROUND
    early_stop = early_stop or C.LGB_EARLY_STOP
    params = dict(C.LGB_MULTI); params['num_class'] = n_class
    dtr = lgb.Dataset(X_tr, label=y_tr, weight=w_tr,
                      categorical_feature=cat_cols, free_raw_data=False)
    dva = lgb.Dataset(X_va, label=y_va, weight=w_va,
                      categorical_feature=cat_cols,
                      reference=dtr, free_raw_data=False)
    return lgb.train(params, dtr, num_boost_round=num_round,
                     valid_sets=[dtr, dva], valid_names=['tr', 'va'],
                     callbacks=[lgb.early_stopping(early_stop, verbose=False),
                                lgb.log_evaluation(C.LGB_LOG_EVERY)])


def _lgb_binary(X_tr, y_tr, w_tr, X_va, y_va, w_va, cat_cols,
                num_round=None, early_stop=None):
    num_round = num_round or C.LGB_NUM_ROUND
    early_stop = early_stop or C.LGB_EARLY_STOP
    params = dict(C.LGB_BIN)
    dtr = lgb.Dataset(X_tr, label=y_tr, weight=w_tr,
                      categorical_feature=cat_cols, free_raw_data=False)
    dva = lgb.Dataset(X_va, label=y_va, weight=w_va,
                      categorical_feature=cat_cols,
                      reference=dtr, free_raw_data=False)
    return lgb.train(params, dtr, num_boost_round=num_round,
                     valid_sets=[dtr, dva], valid_names=['tr', 'va'],
                     callbacks=[lgb.early_stopping(early_stop, verbose=False),
                                lgb.log_evaluation(C.LGB_LOG_EVERY)])


def _xgb_multiclass(X_tr, y_tr, w_tr, X_va, y_va, w_va, n_class,
                    num_round=None, early_stop=None):
    num_round = num_round or C.XGB_NUM_ROUND
    early_stop = early_stop or C.XGB_EARLY_STOP
    params = dict(C.XGB_MULTI); params['num_class'] = n_class
    dtr = xgb.DMatrix(X_tr.values, label=y_tr, weight=w_tr)
    dva = xgb.DMatrix(X_va.values, label=y_va, weight=w_va)
    return xgb.train(params, dtr, num_boost_round=num_round,
                     evals=[(dva, 'va')],
                     early_stopping_rounds=early_stop,
                     verbose_eval=C.XGB_LOG_EVERY)


def _xgb_binary(X_tr, y_tr, w_tr, X_va, y_va, w_va,
                num_round=None, early_stop=None):
    num_round = num_round or C.XGB_NUM_ROUND
    early_stop = early_stop or C.XGB_EARLY_STOP
    params = dict(C.XGB_BIN)
    dtr = xgb.DMatrix(X_tr.values, label=y_tr, weight=w_tr)
    dva = xgb.DMatrix(X_va.values, label=y_va, weight=w_va)
    return xgb.train(params, dtr, num_boost_round=num_round,
                     evals=[(dva, 'va')],
                     early_stopping_rounds=early_stop,
                     verbose_eval=C.XGB_LOG_EVERY)


def _cb_multiclass(X_tr, y_tr, w_tr, X_va, y_va, w_va, cat_feat_idx, n_class,
                   iterations=None, early_stop=None):
    iterations = iterations or C.CB_NUM_ROUND
    early_stop = early_stop or C.CB_EARLY_STOP
    params = dict(C.CB_MULTI); params['iterations'] = iterations
    params['early_stopping_rounds'] = early_stop
    params['classes_count'] = n_class
    tp = Pool(X_tr, y_tr, weight=w_tr, cat_features=cat_feat_idx)
    vp = Pool(X_va, y_va, weight=w_va, cat_features=cat_feat_idx)
    m = CatBoostClassifier(**params); m.fit(tp, eval_set=vp); return m


def _cb_binary(X_tr, y_tr, w_tr, X_va, y_va, w_va, cat_feat_idx,
               iterations=None, early_stop=None):
    iterations = iterations or C.CB_NUM_ROUND
    early_stop = early_stop or C.CB_EARLY_STOP
    params = dict(C.CB_BIN); params['iterations'] = iterations
    params['early_stopping_rounds'] = early_stop
    tp = Pool(X_tr, y_tr, weight=w_tr, cat_features=cat_feat_idx)
    vp = Pool(X_va, y_va, weight=w_va, cat_features=cat_feat_idx)
    m = CatBoostClassifier(**params); m.fit(tp, eval_set=vp); return m


def _pred_lgb(m, X): return m.predict(X).astype(np.float32)
def _pred_xgb_multi(m, X): return m.predict(xgb.DMatrix(X.values),
    iteration_range=(0, m.best_iteration + 1)).astype(np.float32)
def _pred_xgb_binary(m, X): return m.predict(xgb.DMatrix(X.values),
    iteration_range=(0, m.best_iteration + 1)).astype(np.float32)
def _pred_cb_multi(m, X, cf): return m.predict_proba(Pool(X, cat_features=cf)).astype(np.float32)
def _pred_cb_binary(m, X, cf): return m.predict_proba(Pool(X, cat_features=cf))[:, 1].astype(np.float32)


# ---------------------------------------------------------------------------
# Ensemble heads
# ---------------------------------------------------------------------------

def _blend_multi(X_tr, y_tr, w_tr, X_va, y_va, w_va, X_te, n_class,
                 cat_cols, cat_feat_idx, weights, tag=""):
    probs_va = np.zeros((len(X_va), n_class), dtype=np.float32)
    probs_te = np.zeros((len(X_te), n_class), dtype=np.float32)

    t = time.time(); print(f"      >> LGB {tag} ({len(X_tr)}r × {n_class}c)")
    m = _lgb_multiclass(X_tr, y_tr, w_tr, X_va, y_va, w_va, n_class, cat_cols)
    probs_va += weights['lgb'] * _pred_lgb(m, X_va)
    probs_te += weights['lgb'] * _pred_lgb(m, X_te)
    print(f"         LGB done in {time.time()-t:.1f}s (iter {m.best_iteration})")

    t = time.time(); print(f"      >> XGB {tag}")
    m = _xgb_multiclass(X_tr, y_tr, w_tr, X_va, y_va, w_va, n_class)
    probs_va += weights['xgb'] * _pred_xgb_multi(m, X_va)
    probs_te += weights['xgb'] * _pred_xgb_multi(m, X_te)
    print(f"         XGB done in {time.time()-t:.1f}s (iter {m.best_iteration})")

    t = time.time(); print(f"      >> CB  {tag}")
    m = _cb_multiclass(X_tr, y_tr, w_tr, X_va, y_va, w_va, cat_feat_idx, n_class)
    probs_va += weights['cb'] * _pred_cb_multi(m, X_va, cat_feat_idx)
    probs_te += weights['cb'] * _pred_cb_multi(m, X_te, cat_feat_idx)
    print(f"         CB  done in {time.time()-t:.1f}s (iter {m.get_best_iteration()})")

    s = sum(weights.values()); probs_va /= s; probs_te /= s
    return probs_va, probs_te


def _blend_binary(X_tr, y_tr, w_tr, X_va, y_va, w_va, X_te,
                  cat_cols, cat_feat_idx, weights, tag=""):
    probs_va = np.zeros(len(X_va), dtype=np.float32)
    probs_te = np.zeros(len(X_te), dtype=np.float32)

    t = time.time(); print(f"      >> LGB {tag} binary")
    m = _lgb_binary(X_tr, y_tr, w_tr, X_va, y_va, w_va, cat_cols)
    probs_va += weights['lgb'] * _pred_lgb(m, X_va)
    probs_te += weights['lgb'] * _pred_lgb(m, X_te)
    print(f"         LGB done in {time.time()-t:.1f}s (iter {m.best_iteration})")

    t = time.time(); print(f"      >> XGB {tag} binary")
    m = _xgb_binary(X_tr, y_tr, w_tr, X_va, y_va, w_va)
    probs_va += weights['xgb'] * _pred_xgb_binary(m, X_va)
    probs_te += weights['xgb'] * _pred_xgb_binary(m, X_te)
    print(f"         XGB done in {time.time()-t:.1f}s (iter {m.best_iteration})")

    t = time.time(); print(f"      >> CB  {tag} binary")
    m = _cb_binary(X_tr, y_tr, w_tr, X_va, y_va, w_va, cat_feat_idx)
    probs_va += weights['cb'] * _pred_cb_binary(m, X_va, cat_feat_idx)
    probs_te += weights['cb'] * _pred_cb_binary(m, X_te, cat_feat_idx)
    print(f"         CB  done in {time.time()-t:.1f}s (iter {m.get_best_iteration()})")

    s = sum(weights.values()); probs_va /= s; probs_te /= s
    return probs_va, probs_te


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_pipeline(seed: int = C.SEED, n_splits: int = C.N_SPLITS,
                 use_weights: bool = True):
    t0 = time.time()
    print("[v23] loading + feature build (matchup deferred per-fold)")
    train_raw, test_raw = load_raw(C.TRAIN_CSV, C.TEST_CSV)
    data = build_targets(train_raw, test_raw)
    X_train, X_test = data['X_train'], data['X_test']
    y_action, y_point, y_server = data['y_action'], data['y_point'], data['y_server']
    groups = data['groups_match']
    rally_tr, strike_tr = data['rally_train'], data['strike_train']
    rally_te = data['rally_test']
    cat_cols = data['cat_cols']
    ctx_w = data['sample_w'] if use_weights else np.ones(len(X_train), dtype=np.float32)
    train_mt_meta = data['train_mt_meta']
    test_mt_meta = data['test_mt_meta']

    print(f"       X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"       # features: {X_train.shape[1]} | # cat: {len(cat_cols)}")
    print(f"       matchup cols (+26 per fold): {len(MT_COL_NAMES)}")

    # ----- sanity: all y_action should be in [0..14]; remap just in case -----
    bad = (y_action >= 15)
    if bad.any():
        print(f"[warn] {bad.sum()} training rows have action >= 15 (serve); dropping them")
        keep = ~bad
        X_train = X_train[keep].reset_index(drop=True)
        y_action = y_action[keep]; y_point = y_point[keep]; y_server = y_server[keep]
        groups = groups[keep]; rally_tr = rally_tr[keep]; strike_tr = strike_tr[keep]
        ctx_w = ctx_w[keep]
        train_mt_meta = train_mt_meta[keep].reset_index(drop=True)

    cat_feat_idx = [X_train.columns.get_loc(c) for c in cat_cols
                    if c in X_train.columns]

    # class-inverse weights per task
    w_cls_action = _class_inverse_weights(y_action, 15)
    w_cls_point = _class_inverse_weights(y_point, 10)
    # server is binary — small inverse skew only
    w_cls_server = _class_inverse_weights(y_server, 2, power=0.3)

    # composite: context-len × class-inverse
    w_action = (ctx_w * w_cls_action).astype(np.float32)
    w_point = (ctx_w * w_cls_point).astype(np.float32)
    w_server = (ctx_w * w_cls_server).astype(np.float32)

    # Two-stage point: binary terminal (y=0 → 1, else 0) and 9-class zone
    y_terminal = (y_point == 0).astype(int)
    nz_mask = (y_point != 0)
    y_zone = (y_point - 1)  # for rows where nz_mask is True, valid values 0..8
    # inverse class weights for the 9-class head (computed only on non-terminal rows)
    w_zone_cls = np.ones_like(y_point, dtype=np.float32)
    if nz_mask.any():
        w_zone_cls_sub = _class_inverse_weights(y_zone[nz_mask], 9)
        w_zone_cls[nz_mask] = w_zone_cls_sub
    w_zone = (ctx_w * w_zone_cls).astype(np.float32)

    prior_action = np.bincount(y_action, minlength=15) / len(y_action)
    prior_point = np.bincount(y_point, minlength=10) / len(y_point)

    N_ACT = 15
    oof_action = np.zeros((len(X_train), N_ACT), dtype=np.float32)
    # point OOF final 10-class combined prob
    oof_point = np.zeros((len(X_train), 10), dtype=np.float32)
    oof_server = np.zeros(len(X_train), dtype=np.float32)

    test_action_sum = np.zeros((len(X_test), N_ACT), dtype=np.float32)
    test_point_sum = np.zeros((len(X_test), 10), dtype=np.float32)
    test_server_sum = np.zeros(len(X_test), dtype=np.float32)

    X_train_s = X_train.drop(columns=SERVER_DROP, errors='ignore')
    X_test_s = X_test.drop(columns=SERVER_DROP, errors='ignore')
    cat_cols_s = [c for c in cat_cols if c not in SERVER_DROP]
    cat_feat_idx_s = [X_train_s.columns.get_loc(c) for c in cat_cols_s
                      if c in X_train_s.columns]

    W = C.ENSEMBLE_W

    def _attach_mt(X_df, mt_arr):
        """Append 26 matchup float columns to X_df (DataFrame), return new DF."""
        mt_df = pd.DataFrame(mt_arr, columns=MT_COL_NAMES, index=X_df.index)
        return pd.concat([X_df, mt_df], axis=1)

    gkf = GroupKFold(n_splits=n_splits)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_action, groups)):
        t_fold = time.time()
        print(f"\n--- fold {fold+1}/{n_splits} "
              f"(tr={len(tr_idx)}, va={len(va_idx)}) ---")

        # ---- v23: per-fold matchup stats (leak fix) ----
        t_mt = time.time()
        mt_stats = build_matchup_transitions_v23(
            train_mt_meta.iloc[tr_idx].reset_index(drop=True))
        mt_tr = compute_matchup_features_v23(
            train_mt_meta.iloc[tr_idx].reset_index(drop=True), mt_stats)
        mt_va = compute_matchup_features_v23(
            train_mt_meta.iloc[va_idx].reset_index(drop=True), mt_stats)
        mt_te = compute_matchup_features_v23(test_mt_meta, mt_stats)
        print(f"  [matchup] per-fold stats built "
              f"({len(mt_stats['pair_action_trans'])} (srv,rcv,prev1) keys) "
              f"in {time.time()-t_mt:.1f}s")

        X_tr = _attach_mt(X_train.iloc[tr_idx].reset_index(drop=True), mt_tr)
        X_va = _attach_mt(X_train.iloc[va_idx].reset_index(drop=True), mt_va)
        X_te_fold = _attach_mt(X_test.reset_index(drop=True), mt_te)

        X_tr_s_fold = X_tr.drop(columns=SERVER_DROP, errors='ignore')
        X_va_s_fold = X_va.drop(columns=SERVER_DROP, errors='ignore')
        X_te_s_fold = X_te_fold.drop(columns=SERVER_DROP, errors='ignore')

        # cat_cols stays the same — matchup cols are continuous float
        cat_feat_idx_fold = [X_tr.columns.get_loc(c) for c in cat_cols
                              if c in X_tr.columns]
        cat_feat_idx_s_fold = [X_tr_s_fold.columns.get_loc(c) for c in cat_cols_s
                                if c in X_tr_s_fold.columns]

        # ---- action (15-class: 0..14) ----
        print("  [action] 15-class blend (serves masked out of label space)")
        pv_a, pt_a = _blend_multi(
            X_tr, y_action[tr_idx], w_action[tr_idx],
            X_va, y_action[va_idx], w_action[va_idx],
            X_te_fold, N_ACT, cat_cols, cat_feat_idx_fold, W, tag='[A]')
        oof_action[va_idx] = pv_a
        test_action_sum += pt_a

        # ---- point stage 1: terminal binary ----
        print("  [point/stage1] binary terminal (class 0 vs rest)")
        pv_term, pt_term = _blend_binary(
            X_tr, y_terminal[tr_idx], w_point[tr_idx],
            X_va, y_terminal[va_idx], w_point[va_idx],
            X_te_fold, cat_cols, cat_feat_idx_fold, W, tag='[P1]')

        # ---- point stage 2: 9-class zone on non-terminal subset ----
        # Use local-within-fold non-terminal masks
        nz_tr_local = nz_mask[tr_idx]
        nz_va_local = nz_mask[va_idx]
        print(f"  [point/stage2] 9-class zone on {nz_tr_local.sum()}/{len(tr_idx)} "
              f"non-terminal rows")
        pv_z_sub, pt_z = _blend_multi(
            X_tr[nz_tr_local].reset_index(drop=True),
            y_zone[tr_idx][nz_tr_local], w_zone[tr_idx][nz_tr_local],
            X_va[nz_va_local].reset_index(drop=True),
            y_zone[va_idx][nz_va_local], w_zone[va_idx][nz_va_local],
            X_te_fold, 9, cat_cols, cat_feat_idx_fold, W, tag='[P2]')
        # Fill full-length val array: for terminal val rows, fall back to uniform
        pv_z = np.full((len(va_idx), 9), 1.0 / 9, dtype=np.float32)
        nz_in_va = np.where(nz_va_local)[0]
        pv_z[nz_in_va] = pv_z_sub

        # Combine: P(point=0) = P(terminal); P(point=k, k>=1) = (1-P(term)) * P(zone=k-1)
        pv_p = np.zeros((len(va_idx), 10), dtype=np.float32)
        pv_p[:, 0] = pv_term
        nz_prob = (1.0 - pv_term)[:, None]
        pv_p[:, 1:] = nz_prob * pv_z
        oof_point[va_idx] = pv_p

        pt_p = np.zeros((len(X_test), 10), dtype=np.float32)
        pt_p[:, 0] = pt_term
        pt_p[:, 1:] = (1.0 - pt_term)[:, None] * pt_z
        test_point_sum += pt_p

        # ---- server ----
        print("  [server] binary blend (drop parity-leak cols)")
        pv_s, pt_s = _blend_binary(
            X_tr_s_fold, y_server[tr_idx], w_server[tr_idx],
            X_va_s_fold, y_server[va_idx], w_server[va_idx],
            X_te_s_fold, cat_cols_s, cat_feat_idx_s_fold, W, tag='[S]')
        oof_server[va_idx] = pv_s
        test_server_sum += pt_s

        # fold-level OOF eval
        va_rally, va_strike = rally_tr[va_idx], strike_tr[va_idx]
        pick = _pick_random_target_per_rally(va_rally, va_strike, seed=seed+fold)
        ya = y_action[va_idx][pick]
        yp = y_point[va_idx][pick]
        ys = y_server[va_idx][pick]
        f1a = f1_score(ya, pv_a[pick].argmax(1),
                       labels=list(range(N_ACT)), average='macro', zero_division=0)
        f1p = f1_score(yp, pv_p[pick].argmax(1),
                       labels=list(range(10)), average='macro', zero_division=0)
        try: auc = roc_auc_score(ys, pv_s[pick])
        except ValueError: auc = 0.5
        overall = 0.4 * f1a + 0.4 * f1p + 0.2 * auc
        print(f"  [fold {fold+1}] F1(A)={f1a:.4f} F1(P)={f1p:.4f} "
              f"AUC={auc:.4f} overall={overall:.4f} "
              f"elapsed={time.time()-t_fold:.1f}s")

    test_action_sum /= n_splits
    test_point_sum /= n_splits
    test_server_sum /= n_splits

    # ---- v23 OOF eval: RAW argmax (match how submission is built) ----
    pick = _pick_random_target_per_rally(rally_tr, strike_tr, seed=seed)
    yaL = y_action[pick]; ypL = y_point[pick]; ysL = y_server[pick]
    paL = oof_action[pick]; ppL = oof_point[pick]; psL = oof_server[pick]
    print(f"\n[oof] random-target subset: {len(pick)} rallies (RAW argmax, honest)")

    f1a_raw = f1_score(yaL, paL.argmax(1), labels=list(range(N_ACT)),
                       average='macro', zero_division=0)
    f1p_raw = f1_score(ypL, ppL.argmax(1), labels=list(range(10)),
                       average='macro', zero_division=0)
    auc_server = roc_auc_score(ysL, psL)
    overall_oof = 0.4 * f1a_raw + 0.4 * f1p_raw + 0.2 * auc_server
    f1a_post = f1a_raw  # keep var name for 19-class calc below
    f1p_post = f1p_raw

    # Save everything FIRST (before any potentially-failing print),
    # so a stray Unicode issue in the console can't destroy the run.
    np.save(C.OOF_ACTION_NPY, oof_action)
    np.save(C.OOF_POINT_NPY, oof_point)
    np.save(C.OOF_SERVER_NPY, oof_server)
    np.save(C.TEST_ACTION_NPY, test_action_sum)
    np.save(C.TEST_POINT_NPY, test_point_sum)
    np.save(C.TEST_SERVER_NPY, test_server_sum)

    # v23: USE RAW ARGMAX (no tau/shifts - they overfit OOF per LB analysis)
    pred_action = test_action_sum.argmax(1)
    pred_point = test_point_sum.argmax(1)

    # v23: Keep server as RAW float probs (AUC is invariant to monotonic shift).
    # Writing shift_to_mean for backwards compatibility but no longer
    # enforcing any "sweet spot" since LB evidence disproved the 0.62 theory
    # (neighbor team's 0.3118 submission had mean 0.53, also float).
    def shift_to_mean(probs, target_mean, tol=0.001, max_iter=50):
        probs = np.clip(probs, 0.01, 0.99)
        lo, hi = -5.0, 5.0
        for _ in range(max_iter):
            c = (lo + hi) / 2
            p = 1 / (1 + np.exp(-(np.log(probs / (1 - probs)) + c)))
            m = p.mean()
            if abs(m - target_mean) < tol:
                return p
            if m < target_mean: lo = c
            else: hi = c
        return p
    pred_server = test_server_sum.astype(float)  # raw float, AUC-optimal

    sub = pd.DataFrame({
        'rally_uid': rally_te,
        'actionId': pred_action.astype(int),
        'pointId': pred_point.astype(int),
        'serverGetPoint': pred_server,
    }).sort_values('rally_uid').reset_index(drop=True)
    sub.to_csv(C.SUBMISSION_OUT, index=False)
    print(f"  [v23] server mean raw: {pred_server.mean():.3f} (no post-shift)")
    print(f"  action dist: {dict(sub.actionId.value_counts().sort_index())}")
    print(f"  point  dist: {dict(sub.pointId.value_counts().sort_index())}")

    # 19-class equivalent for comparison
    f1a_19 = f1a_post * N_ACT / 19
    overall_19 = 0.4 * f1a_19 + 0.4 * f1p_post + 0.2 * auc_server
    print(f"\n=== v23 OOF (15-class action, RAW argmax) = "
          f"0.4*{f1a_post:.4f} + 0.4*{f1p_post:.4f} + 0.2*{auc_server:.4f} "
          f"= {overall_oof:.4f} ===")
    print(f"     v23 OOF (19-class equiv.) ~= {overall_19:.4f}")
    print(f"\n[save] submission -> {C.SUBMISSION_OUT} ({len(sub)} rows)")
    print(f"[save] tensors -> {C.OUT_DIR}")
    print(f"\n[done] elapsed = {time.time()-t0:.1f}s")
    return overall_oof
