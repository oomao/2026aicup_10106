"""Within-match transductive POINT feature builder (OOV-safe).

For each stroke row, compute conditional pointId distributions and tendency
features from OTHER strokes of the SAME MATCH (never the target row's own label).
These are OOV-safe: built from the rally's own match, no train-player identity.

Causal/leak rules:
- Within-match stats for a row exclude the row's own (match,target_player,prev) target.
  We use a "sum-then-subtract-self" trick: aggregate all same-match strokes of a
  player into a count vector, then for each row subtract its own one-hot pointId.
  This yields leave-self-out empirical distributions (mimics: at test we never see
  the target stroke's own point, only the other visible strokes).
- Serve-context (serve_spin, serve_point) buckets crossed with player tendency.
- action x spin/point joint, landing geometry interactions.
"""
import numpy as np
import pandas as pd

N_PT = 10
N_ACT = 19  # actionId 0-18
ALPHA = 0.5  # Dirichlet smoothing for within-match conditionals


def _smooth_dist(cnt_2d, prior_1d, alpha=ALPHA):
    """cnt_2d: (n, N_PT) integer counts. prior_1d: (N_PT,) global prior.
    Returns smoothed distribution rows; rows with zero count -> prior."""
    tot = cnt_2d.sum(1, keepdims=True)
    sm = (cnt_2d + alpha * prior_1d[None, :]) / (tot + alpha)
    # where tot==0, fall back to prior exactly
    zero = (tot[:, 0] == 0)
    sm[zero] = prior_1d[None, :]
    return sm.astype(np.float32)


def add_serve_and_seq(df):
    """Add per-rally serve context and previous-stroke sequence features.
    df must be sorted by rally_uid, strikeNumber."""
    df = df.copy()
    g = df.groupby('rally_uid', sort=False)
    # serve = stroke 1 of the rally (the serve). Its spin/point/position describe rally start.
    first = g[['spinId', 'pointId', 'positionId', 'strengthId', 'handId']].transform('first')
    df['serve_spin'] = first['spinId'].astype(int)
    df['serve_point'] = first['pointId'].astype(int)
    df['serve_pos'] = first['positionId'].astype(int)
    # strokes since serve = strikeNumber - 1 ; parity
    df['since_serve'] = (df['strikeNumber'] - 1).astype(int)
    df['since_serve_parity'] = (df['since_serve'] % 2).astype(int)
    # previous strokes (already-known history, legit feature per organizer)
    for k in [1, 2, 3]:
        df[f'prev{k}_pointId'] = g['pointId'].shift(k).fillna(-1).astype(int)
        df[f'prev{k}_actionId'] = g['actionId'].shift(k).fillna(-1).astype(int)
        df[f'prev{k}_spinId'] = g['spinId'].shift(k).fillna(-1).astype(int)
        df[f'prev{k}_positionId'] = g['positionId'].shift(k).fillna(-1).astype(int)
        df[f'prev{k}_handId'] = g['handId'].shift(k).fillna(-1).astype(int)
        df[f'prev{k}_strengthId'] = g['strengthId'].shift(k).fillna(-1).astype(int)
    return df


def build_within_match_conditionals(stats_df, target_df, global_prior, self_subtract=False):
    """Build within-match conditional point distributions for each row of target_df,
    using counts aggregated over stats_df (same-match strokes pool).

    stats_df: pool of strokes to aggregate (must have match, gamePlayerId, pointId,
              serve_spin, serve_point, prev1_positionId, prev1_spinId, since_serve_parity).
    target_df: rows to produce features for. Must contain the SAME key columns,
              evaluated for the TARGET player (gamePlayerId of stats == target player).
    self_subtract: if True, subtract target_df row's own one-hot pointId from the
              matching (match,player,...) aggregate (leave-self-out). Used when
              stats_df == target_df pool (train OOF / train-all).

    Returns dict of (n_target, N_PT) arrays for several conditioning keys.
    """
    out = {}
    pt = stats_df['pointId'].values.astype(int)
    onehot = np.eye(N_PT, dtype=np.float64)[pt]  # (n_stats, N_PT)

    def agg_by(keys_stats, keys_target, name, self_oh=None):
        # Build a DataFrame summing one-hot point by key over stats pool
        kcols = [f'_k{i}' for i in range(len(keys_stats))]
        sdf = pd.DataFrame({c: stats_df[k].values for c, k in zip(kcols, keys_stats)})
        for j in range(N_PT):
            sdf[f'_oh{j}'] = onehot[:, j]
        summ = sdf.groupby(kcols, sort=False).sum()
        # map onto target rows
        tdf = pd.DataFrame({c: target_df[k].values for c, k in zip(kcols, keys_target)})
        merged = tdf.merge(summ, left_on=kcols, right_index=True, how='left')
        cnt = merged[[f'_oh{j}' for j in range(N_PT)]].fillna(0.0).values
        if self_oh is not None:
            cnt = cnt - self_oh  # leave-self-out
            cnt = np.clip(cnt, 0, None)
        out[name] = _smooth_dist(cnt, global_prior)
        out[name + '_n'] = cnt.sum(1).astype(np.float32)  # support count (OOV signal)

    self_oh = onehot if self_subtract else None
    # but self_subtract only valid if target rows are a subset that appears in stats with same index
    # We instead recompute self one-hot from target_df directly:
    if self_subtract:
        tpt = target_df['pointId'].values.astype(int)
        self_oh = np.eye(N_PT, dtype=np.float64)[tpt]

    # 1. player same-match (the player's landing tendency this match)
    agg_by(['match', 'gamePlayerId'], ['match', 'tgt_player'], 'wm_player', self_oh)
    # 2. player x serve-context (serve_spin, serve_point)
    agg_by(['match', 'gamePlayerId', 'serve_spin', 'serve_point'],
           ['match', 'tgt_player', 'serve_spin', 'serve_point'], 'wm_player_serve', self_oh)
    # 3. player x prev_position (landing geometry conditioning)
    agg_by(['match', 'gamePlayerId', 'prev1_positionId'],
           ['match', 'tgt_player', 'prev1_positionId'], 'wm_player_prevpos', self_oh)
    # 4. player x prev_spin
    agg_by(['match', 'gamePlayerId', 'prev1_spinId'],
           ['match', 'tgt_player', 'prev1_spinId'], 'wm_player_prevspin', self_oh)
    # 5. player x since_serve_parity (server/receiver landing differs)
    agg_by(['match', 'gamePlayerId', 'since_serve_parity'],
           ['match', 'tgt_player', 'since_serve_parity'], 'wm_player_parity', self_oh)
    # 6. match-level serve-context (both players, OOV-safe even if player unseen this match)
    agg_by(['match', 'serve_spin', 'serve_point'],
           ['match', 'serve_spin', 'serve_point'], 'wm_match_serve', None)
    return out


def assemble_numeric(df, wm, baseline_point=None):
    """Concatenate engineered numeric features into a single matrix.
    df: row frame (with prev features + serve features).
    wm: dict of within-match conditional arrays.
    baseline_point: optional (n,10) baseline point probs to include as features
                    (lets GBDT learn a residual on top of baseline). OOV-safe (it's a
                    model output, not a player id)."""
    cols = []
    names = []
    # within-match conditional distributions
    for key in ['wm_player', 'wm_player_serve', 'wm_player_prevpos',
                'wm_player_prevspin', 'wm_player_parity', 'wm_match_serve']:
        cols.append(wm[key])
        names += [f'{key}_p{j}' for j in range(N_PT)]
        cols.append(wm[key + '_n'][:, None])
        names.append(f'{key}_n')
    # raw categorical context as ints (GBDT handles as numeric splits)
    raw_int = ['since_serve', 'since_serve_parity', 'serve_spin', 'serve_point', 'serve_pos',
               'spinId', 'strengthId', 'handId', 'positionId',
               'prev1_pointId', 'prev1_actionId', 'prev1_spinId', 'prev1_positionId',
               'prev1_handId', 'prev1_strengthId',
               'prev2_pointId', 'prev2_actionId', 'prev2_spinId', 'prev2_positionId',
               'prev3_pointId', 'prev3_actionId',
               'scoreSelf', 'scoreOther', 'sex', 'numberGame']
    for c in raw_int:
        cols.append(df[c].values.astype(np.float32)[:, None])
        names.append(c)
    # action x spin / point joint encodings (physical coupling)
    df = df.copy()
    df['_act_spin'] = (df['prev1_actionId'].clip(-1, 18) + 1) * 10 + df['prev1_spinId'].clip(-1, 8) + 1
    df['_act_pos'] = (df['prev1_actionId'].clip(-1, 18) + 1) * 12 + df['positionId'].clip(0, 11)
    df['_hand_spin_pos'] = (df['handId'].clip(0, 3) * 9 + df['spinId'].clip(0, 8)) * 12 + df['positionId'].clip(0, 11)
    for c in ['_act_spin', '_act_pos', '_hand_spin_pos']:
        cols.append(df[c].values.astype(np.float32)[:, None])
        names.append(c)
    if baseline_point is not None:
        cols.append(baseline_point.astype(np.float32))
        names += [f'base_p{j}' for j in range(N_PT)]
    X = np.concatenate(cols, axis=1).astype(np.float32)
    return X, names
