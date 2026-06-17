"""
v2 feature engineering for AICUP 2026.

Design choices:
- Training targets: every stroke with strikeNumber >= 2 (terminal included), so
  pointId=0 is learnable. Context features are built from strokes strictly
  before the target (groupby shift + cumsum).
- Random-truncation alignment: training rows are *reweighted* so their
  context_len distribution matches test's (test visible-length mean 2.9 vs
  train 5.65). This is equivalent to importance sampling without throwing
  away data.
- Rich history: prev1-prev5 lags for key categoricals; running counts of
  every action/point class over visible history; entropy proxy features.
- Player modelling: Dirichlet-smoothed TE for the striking player, the
  opponent, and the (player, opponent) pair. No raw IDs are fed to the model
  (23/63 test players are unseen in train).
- Server target: rally-level y (serverGetPoint) replicated on every target
  row; server head drops context_len/is_server_turn (parity leak at last-NT).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CAT_COLS


# ---------------------------------------------------------------------------
# Basic loading
# ---------------------------------------------------------------------------

def load_raw(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    # Per 2026-04-17 announcement: old test.csv leaks serverGetPoint
    # (constant per rally == true label). The new test.csv will drop this
    # column. Strip it here so no downstream code can accidentally depend
    # on it even while the old file is still in place.
    if 'serverGetPoint' in test.columns:
        test = test.drop(columns=['serverGetPoint'])
        print("[load] stripped leaked `serverGetPoint` column from test.csv")
    return train, test


def sort_rally(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(['rally_uid', 'strikeNumber']).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Synthetic test target rows
# ---------------------------------------------------------------------------

def _append_synthetic_test_targets(test: pd.DataFrame) -> pd.DataFrame:
    test = sort_rally(test)
    last = test.groupby('rally_uid').tail(1).copy()
    orig_main = last['gamePlayerId'].copy()
    last['gamePlayerId'] = last['gamePlayerOtherId']
    last['gamePlayerOtherId'] = orig_main
    last['strikeNumber'] = last['strikeNumber'] + 1
    for c in CAT_COLS:
        last[c] = -1
    last['_is_target'] = 1
    test = test.copy()
    test['_is_target'] = 0
    out = pd.concat([test, last], ignore_index=True)
    return sort_rally(out)


# ---------------------------------------------------------------------------
# Per-stroke context features (same row-level schema for train / test)
# ---------------------------------------------------------------------------

def _cum_equal(df: pd.DataFrame, col: str, val: int, new_name: str):
    ind = (df[col] == val).astype(np.int32)
    df[new_name] = (ind.groupby(df['rally_uid']).cumsum() - ind).astype(np.int32)


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby('rally_uid', sort=False)

    # ----- lag features (strokes before the current row) -----
    for c in CAT_COLS:
        df[f'prev1_{c}'] = g[c].shift(1).fillna(-1).astype(np.int32)
    for c in ['actionId', 'pointId', 'handId', 'spinId', 'positionId', 'strengthId']:
        df[f'prev2_{c}'] = g[c].shift(2).fillna(-1).astype(np.int32)
    for c in ['actionId', 'pointId', 'handId', 'spinId']:
        df[f'prev3_{c}'] = g[c].shift(3).fillna(-1).astype(np.int32)
    for c in ['actionId', 'pointId']:
        df[f'prev4_{c}'] = g[c].shift(4).fillna(-1).astype(np.int32)
        df[f'prev5_{c}'] = g[c].shift(5).fillna(-1).astype(np.int32)

    # ----- opening strokes -----
    for c in ['actionId', 'spinId', 'handId', 'strengthId', 'pointId', 'positionId']:
        df[f'serve_{c}'] = g[c].transform('first').astype(np.int32)

    # stroke 2 (receive) – may coincide with target when strikeNumber == 2
    for c in ['actionId', 'spinId', 'handId', 'pointId', 'positionId']:
        rcv = g[c].transform(lambda s: s.iloc[1] if len(s) >= 2 else -1)
        df[f'rcv_{c}'] = rcv.astype(np.int32)

    # ----- context_len = number of strokes before current -----
    df['context_len'] = (df['strikeNumber'] - 1).astype(np.int32)
    df['is_server_turn'] = (df['strikeNumber'] % 2 == 1).astype(np.int32)

    mask_rcv = df['strikeNumber'] <= 2
    for c in ['actionId', 'spinId', 'handId', 'pointId', 'positionId']:
        df.loc[mask_rcv, f'rcv_{c}'] = -1

    # ----- running counts over visible history (exclusive of current) -----
    for v in range(1, 15):
        _cum_equal(df, 'actionId', v, f'cum_action_eq_{v}')
    for v in range(0, 10):
        _cum_equal(df, 'pointId', v, f'cum_point_eq_{v}')
    for v in [1, 2]:
        _cum_equal(df, 'handId', v, f'cum_hand_eq_{v}')
    for v in [1, 2, 3]:
        _cum_equal(df, 'strengthId', v, f'cum_strength_eq_{v}')
    for v in [1, 2, 3, 4, 5]:
        _cum_equal(df, 'spinId', v, f'cum_spin_eq_{v}')

    # ----- target-player's own previous stroke (2-step lag; turns alternate) -----
    df['tgt_prev_action']   = g['actionId'].shift(2).fillna(-1).astype(np.int32)
    df['tgt_prev_point']    = g['pointId'].shift(2).fillna(-1).astype(np.int32)
    df['tgt_prev_hand']     = g['handId'].shift(2).fillna(-1).astype(np.int32)
    df['tgt_prev_spin']     = g['spinId'].shift(2).fillna(-1).astype(np.int32)
    df['tgt_prev_position'] = g['positionId'].shift(2).fillna(-1).astype(np.int32)
    df['tgt_prev_strength'] = g['strengthId'].shift(2).fillna(-1).astype(np.int32)

    # opponent's last stroke = prev1_*   (already present)
    # opponent's stroke before their last = shift(3) on same rally (alternates)
    df['opp_prev2_action'] = g['actionId'].shift(3).fillna(-1).astype(np.int32)
    df['opp_prev2_point']  = g['pointId'].shift(3).fillna(-1).astype(np.int32)

    # ----- score / match state -----
    df['score_total'] = (df['scoreSelf'] + df['scoreOther']).astype(np.int32)
    df['score_diff']  = (df['scoreSelf'] - df['scoreOther']).astype(np.int32)
    df['score_abs_diff'] = df['score_diff'].abs().astype(np.int32)
    # is_server_leading: depends on which side is server (server is stroke-1 striker)
    df['server_scoreSelf'] = g['scoreSelf'].transform('first').astype(np.int32)
    df['server_scoreOther'] = g['scoreOther'].transform('first').astype(np.int32)
    df['server_score_diff'] = (df['server_scoreSelf']
                                - df['server_scoreOther']).astype(np.int32)

    # game-point indicator: within 11-point rules, player needs 11 (or win-by-2)
    df['at_game_point_self'] = (df['scoreSelf'] >= 10).astype(np.int32)
    df['at_game_point_other'] = (df['scoreOther'] >= 10).astype(np.int32)

    # ----- rally-scale features (count of strokes visible to the target) -----
    # number of strokes before current (same as context_len but kept for clarity)
    df['visible_strokes'] = df['context_len']
    # rally length so far (max strikeNumber minus 1 seen by this row) — but
    # we only know visible strokes, NOT total rally length, so this is
    # identical to context_len at inference.

    # ----- player identifiers kept ONLY for later TE lookups (will be dropped) -----
    # gamePlayerId and gamePlayerOtherId are already present in df.

    return df


# ---------------------------------------------------------------------------
# Target encoding for players
# ---------------------------------------------------------------------------

def _compute_player_stats(train_raw: pd.DataFrame, alpha: float = 25.0,
                          pair_alpha: float = 8.0):
    """Dirichlet-smoothed TE. Computed over NON-TERMINAL strokes for action/point
    (to get the "what does this player tend to play" signal uncontaminated by
    pointId=0 terminal rows), but server win-rate uses rally-level labels."""
    train = sort_rally(train_raw)

    eligible = train[(train.strikeNumber >= 2) & (train.pointId != 0)].copy()

    prior_a = eligible.actionId.value_counts(normalize=True)
    top_a = prior_a.head(12).index.tolist()
    pa = (eligible.groupby('gamePlayerId')['actionId']
          .value_counts().unstack(fill_value=0)
          .reindex(columns=top_a, fill_value=0))
    pa_smooth = (pa + alpha * prior_a.reindex(top_a).values) / \
                (pa.sum(axis=1).values[:, None] + alpha)

    prior_p = eligible.pointId.value_counts(normalize=True)
    top_p = prior_p.head(9).index.tolist()
    pp = (eligible.groupby('gamePlayerId')['pointId']
          .value_counts().unstack(fill_value=0)
          .reindex(columns=top_p, fill_value=0))
    pp_smooth = (pp + alpha * prior_p.reindex(top_p).values) / \
                (pp.sum(axis=1).values[:, None] + alpha)

    # Server win rate at rally level
    svs = train[train.strikeNumber == 1]
    wr_server = svs.groupby('gamePlayerId')['serverGetPoint'].mean()
    wr_recv = svs.groupby('gamePlayerOtherId')['serverGetPoint'].mean()
    prior_wr = train.groupby('rally_uid')['serverGetPoint'].first().mean()
    cnts_s = svs.groupby('gamePlayerId').size()
    cnts_r = svs.groupby('gamePlayerOtherId').size()
    wr_server_sm = (wr_server * cnts_s + prior_wr * alpha) / (cnts_s + alpha)
    wr_recv_sm = (wr_recv * cnts_r + prior_wr * alpha) / (cnts_r + alpha)

    # Pair (player, opponent) win-rate — when player is the server
    pair_wr = svs.groupby(['gamePlayerId', 'gamePlayerOtherId'])['serverGetPoint'].mean()
    pair_cnt = svs.groupby(['gamePlayerId', 'gamePlayerOtherId']).size()
    pair_wr_sm = ((pair_wr * pair_cnt + prior_wr * pair_alpha)
                  / (pair_cnt + pair_alpha))

    # Rally-length by player (as server) — useful game-state feature
    rally_len = train.groupby('rally_uid').agg(
        server=('gamePlayerId', 'first'),
        recv=('gamePlayerOtherId', 'first'),
        L=('strikeNumber', 'max'),
    )
    player_avg_L = rally_len.groupby('server').L.mean()
    prior_L = rally_len.L.mean()
    cnts_L = rally_len.groupby('server').size()
    player_avg_L_sm = ((player_avg_L * cnts_L + prior_L * alpha)
                       / (cnts_L + alpha))

    return {
        'action_cols': top_a,
        'point_cols': top_p,
        'pa': pa_smooth,
        'pp': pp_smooth,
        'wr_server': wr_server_sm,
        'wr_recv': wr_recv_sm,
        'prior_a': prior_a.reindex(top_a).values.astype(np.float32),
        'prior_p': prior_p.reindex(top_p).values.astype(np.float32),
        'prior_wr': float(prior_wr),
        'pair_wr': pair_wr_sm,
        'player_avg_L': player_avg_L_sm,
        'prior_L': float(prior_L),
    }


def _attach_player_te(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    pa, pp = stats['pa'], stats['pp']
    wrs, wrr = stats['wr_server'], stats['wr_recv']
    prior_a, prior_p, prior_wr = stats['prior_a'], stats['prior_p'], stats['prior_wr']

    def _lookup(series, frame, prior_vec, cols_prefix):
        lut = frame.reindex(series.values)
        arr = lut.values
        na_mask = np.isnan(arr).any(axis=1)
        if na_mask.any():
            arr[na_mask] = prior_vec
        res = pd.DataFrame(arr, index=series.index,
                           columns=[f'{cols_prefix}_{int(c)}' for c in frame.columns])
        return res.astype(np.float32)

    tp_pa = _lookup(df['gamePlayerId'], pa, prior_a, 'tp_pa')
    tp_pp = _lookup(df['gamePlayerId'], pp, prior_p, 'tp_pp')
    op_pa = _lookup(df['gamePlayerOtherId'], pa, prior_a, 'op_pa')
    op_pp = _lookup(df['gamePlayerOtherId'], pp, prior_p, 'op_pp')

    def _scalar(series, stat, prior):
        return series.map(stat).fillna(prior).astype(np.float32)

    server_id = df['gamePlayerId'].where(df['is_server_turn'] == 1,
                                         df['gamePlayerOtherId'])
    receiver_id = df['gamePlayerOtherId'].where(df['is_server_turn'] == 1,
                                                df['gamePlayerId'])
    server_wr = _scalar(server_id, wrs, prior_wr)
    receiver_wr = _scalar(receiver_id, wrr, prior_wr)

    # pair (server, receiver) win rate
    pair_wr = stats['pair_wr']
    pair_key = list(zip(server_id.values, receiver_id.values))
    pair_vals = np.array([pair_wr.get(k, prior_wr) for k in pair_key],
                         dtype=np.float32)

    pav = df['gamePlayerId'].map(stats['player_avg_L']).fillna(stats['prior_L']).astype(np.float32)
    oav = df['gamePlayerOtherId'].map(stats['player_avg_L']).fillna(stats['prior_L']).astype(np.float32)

    df = pd.concat([df, tp_pa, tp_pp, op_pa, op_pp], axis=1)
    df['server_wr_te'] = server_wr.values
    df['receiver_wr_te'] = receiver_wr.values
    df['pair_wr_te'] = pair_vals
    df['tp_avg_rally_len'] = pav.values
    df['op_avg_rally_len'] = oav.values
    # wr diff is a strong signal for server AUC
    df['wr_diff_te'] = (server_wr.values - receiver_wr.values).astype(np.float32)
    return df


# ---------------------------------------------------------------------------
# Importance-sampling weights to match test context_len distribution
# ---------------------------------------------------------------------------

def _test_visible_len_pmf(test_raw: pd.DataFrame) -> dict:
    """Probability that a test rally has visible_len == k (= context_len for
    its target row, since target sits at strikeNumber = visible_len+1)."""
    lens = test_raw.groupby('rally_uid').strikeNumber.max().values
    vc = pd.Series(lens).value_counts(normalize=True).to_dict()
    return {int(k): float(v) for k, v in vc.items()}


def _train_context_len_pmf(train_target_rows: pd.DataFrame) -> dict:
    vc = train_target_rows.context_len.value_counts(normalize=True).to_dict()
    return {int(k): float(v) for k, v in vc.items()}


def compute_sample_weights(train_target_rows: pd.DataFrame,
                           test_raw: pd.DataFrame) -> np.ndarray:
    """Reweight training samples so their context_len distribution matches
    the test context_len (= visible_len) distribution.

    Weight = p_test(k) / p_train(k).
    Clip to [0.1, 10] for stability.
    """
    p_te = _test_visible_len_pmf(test_raw)
    p_tr = _train_context_len_pmf(train_target_rows)
    k = train_target_rows.context_len.values
    w = np.array([p_te.get(int(kk), 0.0) / max(p_tr.get(int(kk), 1e-9), 1e-9)
                  for kk in k], dtype=np.float32)
    # Normalise so mean weight = 1
    w = np.clip(w, 0.1, 10.0)
    w = w * (len(w) / w.sum())
    return w


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------

def build_targets(train_raw: pd.DataFrame, test_raw: pd.DataFrame,
                  player_map=None, player_unk=None):
    player_stats = _compute_player_stats(train_raw)

    # ---------- TRAIN ----------
    train = sort_rally(train_raw).copy()
    train['_is_target'] = (train['strikeNumber'] >= 2).astype(int)
    train = add_context_features(train)
    train = _attach_player_te(train, player_stats)

    # ---------- TEST ----------
    test = _append_synthetic_test_targets(test_raw)
    test = add_context_features(test)
    test = _attach_player_te(test, player_stats)

    train_t = train[train['_is_target'] == 1].copy().reset_index(drop=True)
    test_t = test[test['_is_target'] == 1].copy().reset_index(drop=True)

    # v23: Matchup features are NOT attached here (would leak labels
    # across folds, exactly the M1 bug). They are computed PER-FOLD
    # inside train.py using only train-fold rows.
    print(f"[v23] matchup features DEFERRED to per-fold computation (leak fix)")

    y_action = train_t['actionId'].astype(int).values
    y_point = train_t['pointId'].astype(int).values
    y_server = train_t['serverGetPoint'].astype(int).values
    groups_match = train_t['match'].astype(int).values
    rally_train = train_t['rally_uid'].astype(int).values
    strike_train = train_t['strikeNumber'].astype(int).values
    rally_test = test_t['rally_uid'].astype(int).values

    # Raw player IDs never reach the model (23/63 test players unseen).
    # Target-row self-attributes (strikeId/handId/spinId/strengthId/pointId/
    # positionId) are the prediction surface in train but -1 in test: drop.
    drop_cols = {
        'rally_uid', 'match', 'rally_id', 'strikeNumber',
        'serverGetPoint', 'actionId', 'pointId', 'scoreSelf', 'scoreOther',
        'gamePlayerId', 'gamePlayerOtherId', '_is_target',
        'spinId', 'handId', 'strengthId', 'strikeId', 'positionId',
    }
    feature_cols = [c for c in train_t.columns if c not in drop_cols]
    feature_cols = [c for c in feature_cols if c in test_t.columns]

    cat_cols_model = [
        'sex', 'numberGame',
        'prev1_actionId', 'prev1_pointId', 'prev1_spinId', 'prev1_handId',
        'prev1_strengthId', 'prev1_strikeId', 'prev1_positionId',
        'prev2_actionId', 'prev2_pointId', 'prev2_handId',
        'prev2_spinId', 'prev2_positionId', 'prev2_strengthId',
        'prev3_actionId', 'prev3_pointId', 'prev3_handId', 'prev3_spinId',
        'prev4_actionId', 'prev4_pointId',
        'prev5_actionId', 'prev5_pointId',
        'serve_actionId', 'serve_spinId', 'serve_handId',
        'serve_strengthId', 'serve_pointId', 'serve_positionId',
        'rcv_actionId', 'rcv_spinId', 'rcv_handId', 'rcv_pointId',
        'rcv_positionId',
        'tgt_prev_action', 'tgt_prev_point', 'tgt_prev_hand',
        'tgt_prev_spin', 'tgt_prev_position', 'tgt_prev_strength',
        'opp_prev2_action', 'opp_prev2_point',
        'is_server_turn', 'at_game_point_self', 'at_game_point_other',
    ]
    cat_cols_model = [c for c in cat_cols_model if c in feature_cols]

    X_train = train_t[feature_cols].copy()
    X_test = test_t[feature_cols].copy()

    # shift categoricals so -1 becomes a distinct non-negative bucket
    CAT_OFFSET = 2
    for c in cat_cols_model:
        X_train[c] = (X_train[c].astype(np.int32) + CAT_OFFSET).astype(np.int32)
        X_test[c] = (X_test[c].astype(np.int32) + CAT_OFFSET).astype(np.int32)

    # context_len sample weights — align training distribution with test
    ctx_for_weights = train_t[['context_len']].copy()
    sample_w = compute_sample_weights(ctx_for_weights, test_raw)

    # v23: preserve meta columns needed for per-fold matchup computation.
    # These DataFrames retain rally_uid, gamePlayerId, gamePlayerOtherId,
    # prev1_actionId, prev1_pointId, actionId (train only), pointId (train only).
    mt_cols = ['rally_uid', 'gamePlayerId', 'gamePlayerOtherId',
               'prev1_actionId', 'prev1_pointId']
    train_mt_meta = train_t[mt_cols + ['actionId', 'pointId']].copy()
    test_mt_meta = test_t[mt_cols].copy()

    return {
        'X_train': X_train, 'X_test': X_test,
        'y_action': y_action, 'y_point': y_point, 'y_server': y_server,
        'groups_match': groups_match,
        'rally_train': rally_train, 'strike_train': strike_train,
        'rally_test': rally_test,
        'feature_cols': feature_cols, 'cat_cols': cat_cols_model,
        'sample_w': sample_w,
        'train_mt_meta': train_mt_meta, 'test_mt_meta': test_mt_meta,
    }
