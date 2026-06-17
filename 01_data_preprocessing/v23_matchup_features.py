"""v22 — Matchup Transition Features (B2-style).

For each (server_id, receiver_id) pair in TRAIN, compute conditional
distributions P(action_target | prev1_action, pair) and
P(point_target | prev1_point, pair).

At inference, look up these probabilities for each test rally's
(server, receiver) pair. UNK pairs fall back to marginal P(* | prev1).

This captures PLAYER-PAIR interaction: "how does A typically counter
B's topspin?" which is orthogonal to single-player TE.

Critical: compute PER FOLD to avoid leakage. Only use training
portion for the lookup table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_matchup_transitions_v23(train_mt_meta: pd.DataFrame,
                                   alpha: float = 20.0):
    """v23 variant: takes the compact mt_meta frame with columns
    rally_uid, gamePlayerId, gamePlayerOtherId, prev1_actionId,
    prev1_pointId, actionId, pointId. Computes matchup transition stats.

    Caller must pass ONLY training-fold rows to avoid leakage.
    """
    tr = train_mt_meta.copy()
    first = tr.drop_duplicates('rally_uid', keep='first')
    uid_to_srv = dict(zip(first['rally_uid'], first['gamePlayerId']))
    uid_to_rcv = dict(zip(first['rally_uid'], first['gamePlayerOtherId']))
    tr = tr.assign(
        server=tr['rally_uid'].map(uid_to_srv).astype(int),
        receiver=tr['rally_uid'].map(uid_to_rcv).astype(int),
    )
    tr = tr[tr['actionId'].between(0, 14)]

    prior_a = np.bincount(tr['actionId'].astype(int), minlength=15).astype(np.float32)
    prior_a = prior_a / max(prior_a.sum(), 1)
    prior_p = np.bincount(tr['pointId'].astype(int), minlength=10).astype(np.float32)
    prior_p = prior_p / max(prior_p.sum(), 1)

    g = tr.groupby(['server', 'receiver', 'prev1_actionId'])['actionId']
    pivot = g.value_counts().unstack(fill_value=0).reindex(columns=range(15), fill_value=0)
    counts = pivot.values.astype(np.float64)
    totals = counts.sum(axis=1, keepdims=True)
    smoothed = (counts + alpha * prior_a) / (totals + alpha)
    pair_action_trans = {k: smoothed[i].astype(np.float32)
                         for i, k in enumerate(pivot.index.tolist())}

    g = tr.groupby(['server', 'receiver', 'prev1_pointId'])['pointId']
    pivot = g.value_counts().unstack(fill_value=0).reindex(columns=range(10), fill_value=0)
    counts = pivot.values.astype(np.float64)
    totals = counts.sum(axis=1, keepdims=True)
    smoothed = (counts + alpha * prior_p) / (totals + alpha)
    pair_point_trans = {k: smoothed[i].astype(np.float32)
                         for i, k in enumerate(pivot.index.tolist())}

    pair_popularity = tr.groupby(['server', 'receiver']).size().to_dict()

    return {
        'pair_action_trans': pair_action_trans,
        'pair_point_trans': pair_point_trans,
        'prior_a': prior_a, 'prior_p': prior_p,
        'pair_popularity': pair_popularity,
    }


def compute_matchup_features_v23(mt_meta: pd.DataFrame, stats: dict) -> np.ndarray:
    """Return a (N, 26) float array: 15 mt_a + 10 mt_p + 1 mt_pair_pop.
    Uses first-per-rally (server, receiver) lookup keyed on (srv, rcv, prev1).
    """
    df = mt_meta.copy()
    first = df.drop_duplicates('rally_uid', keep='first')
    uid_to_srv = dict(zip(first['rally_uid'], first['gamePlayerId']))
    uid_to_rcv = dict(zip(first['rally_uid'], first['gamePlayerOtherId']))
    srv = df['rally_uid'].map(uid_to_srv).fillna(-1).astype(int).values
    rcv = df['rally_uid'].map(uid_to_rcv).fillna(-1).astype(int).values
    pa_col = df['prev1_actionId'].astype(int).values
    pp_col = df['prev1_pointId'].astype(int).values

    pa = stats['pair_action_trans']
    pp = stats['pair_point_trans']
    prior_a = stats['prior_a']
    prior_p = stats['prior_p']
    pop = stats['pair_popularity']

    n = len(df)
    out = np.zeros((n, 26), dtype=np.float32)
    for i in range(n):
        key_a = (int(srv[i]), int(rcv[i]), int(pa_col[i]))
        out[i, :15] = pa.get(key_a, prior_a)
        key_p = (int(srv[i]), int(rcv[i]), int(pp_col[i]))
        out[i, 15:25] = pp.get(key_p, prior_p)
        out[i, 25] = np.log1p(pop.get((int(srv[i]), int(rcv[i])), 0))
    return out


MT_COL_NAMES = [f'mt_a_{k}' for k in range(15)] + \
               [f'mt_p_{k}' for k in range(10)] + ['mt_pair_pop']


# ---------------------------------------------------------------------------
# Legacy v22 API (unused by v23, retained for reference)
# ---------------------------------------------------------------------------

def build_matchup_transitions(train_target_rows: pd.DataFrame, alpha: float = 20.0):
    """[Legacy v22] Returns dict with pair transition stats. Leaky if
    called on full train_t."""
    tr = train_target_rows.copy()
    # Determine server/receiver based on stroke position
    # gamePlayerId is who HIT the stroke, gamePlayerOtherId is the opponent
    # Server = gamePlayerId at strike 1; maps directly
    # For target row's perspective, server is the earliest striker of the rally
    # We'll use groupby rally_uid to find server per rally.
    serve_rows = tr[['rally_uid', 'gamePlayerId']].drop_duplicates('rally_uid', keep='first')
    uid_to_server = dict(zip(serve_rows['rally_uid'], serve_rows['gamePlayerId']))
    uid_to_receiver_all = tr[['rally_uid', 'gamePlayerOtherId']].drop_duplicates('rally_uid', keep='first')
    uid_to_receiver = dict(zip(uid_to_receiver_all['rally_uid'],
                                uid_to_receiver_all['gamePlayerOtherId']))

    tr = tr.assign(
        server=tr['rally_uid'].map(uid_to_server).astype(int),
        receiver=tr['rally_uid'].map(uid_to_receiver).astype(int),
    )

    # Restrict to valid actions
    tr = tr[tr['actionId'].between(0, 14)]

    # Prior action / point
    prior_a = np.zeros(15, dtype=np.float32)
    cnts = np.bincount(tr['actionId'].astype(int), minlength=15)
    prior_a = cnts / max(cnts.sum(), 1)
    prior_p = np.zeros(10, dtype=np.float32)
    cnts = np.bincount(tr['pointId'].astype(int), minlength=10)
    prior_p = cnts / max(cnts.sum(), 1)

    # P(action | server, receiver, prev1_action)
    g = tr.groupby(['server', 'receiver', 'prev1_actionId'])['actionId']
    pivot = g.value_counts().unstack(fill_value=0).reindex(columns=range(15), fill_value=0)
    counts = pivot.values.astype(np.float64)
    totals = counts.sum(axis=1, keepdims=True)
    smoothed = (counts + alpha * prior_a) / (totals + alpha)
    pair_action_trans = {k: smoothed[i].astype(np.float32)
                          for i, k in enumerate(pivot.index.tolist())}

    # P(point | server, receiver, prev1_point)
    g = tr.groupby(['server', 'receiver', 'prev1_pointId'])['pointId']
    pivot = g.value_counts().unstack(fill_value=0).reindex(columns=range(10), fill_value=0)
    counts = pivot.values.astype(np.float64)
    totals = counts.sum(axis=1, keepdims=True)
    smoothed = (counts + alpha * prior_p) / (totals + alpha)
    pair_point_trans = {k: smoothed[i].astype(np.float32)
                         for i, k in enumerate(pivot.index.tolist())}

    pair_popularity = tr.groupby(['server', 'receiver']).size().to_dict()

    return {
        'pair_action_trans': pair_action_trans,
        'pair_point_trans': pair_point_trans,
        'prior_a': prior_a, 'prior_p': prior_p,
        'pair_popularity': pair_popularity,
    }


def attach_matchup_features(df: pd.DataFrame, stats: dict):
    """Attach matchup transition features:
    - mt_a_0..14 : P(action_k | server, receiver, prev1_action)
    - mt_p_0..9  : P(point_k | server, receiver, prev1_point)
    - mt_pair_popularity : log1p count of this pair in train
    """
    # Identify server & receiver from rally
    # Server = first stroke's gamePlayerId per rally
    g = df.groupby('rally_uid', sort=False)
    df = df.copy()
    df['mt_server'] = g['gamePlayerId'].transform('first').astype(int)
    df['mt_receiver'] = g['gamePlayerOtherId'].transform('first').astype(int)

    pa = stats['pair_action_trans']; pp = stats['pair_point_trans']
    prior_a = stats['prior_a']; prior_p = stats['prior_p']
    pop = stats['pair_popularity']

    n = len(df)
    feat_a = np.zeros((n, 15), dtype=np.float32)
    feat_p = np.zeros((n, 10), dtype=np.float32)
    feat_pop = np.zeros(n, dtype=np.float32)

    srv = df['mt_server'].values
    rcv = df['mt_receiver'].values
    pa_col = df['prev1_actionId'].values
    pp_col = df['prev1_pointId'].values

    for i in range(n):
        key_a = (int(srv[i]), int(rcv[i]), int(pa_col[i]))
        feat_a[i] = pa.get(key_a, prior_a)
        key_p = (int(srv[i]), int(rcv[i]), int(pp_col[i]))
        feat_p[i] = pp.get(key_p, prior_p)
        feat_pop[i] = np.log1p(pop.get((int(srv[i]), int(rcv[i])), 0))

    for k in range(15):
        df[f'mt_a_{k}'] = feat_a[:, k]
    for k in range(10):
        df[f'mt_p_{k}'] = feat_p[:, k]
    df['mt_pair_pop'] = feat_pop
    # drop the helper cols
    df = df.drop(columns=['mt_server', 'mt_receiver'])
    return df
