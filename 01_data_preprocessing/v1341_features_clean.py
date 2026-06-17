"""v1341 — CLEAN richer within-match transductive POINT features (OOV-safe, NO terminal leak).

Identical to v1340 features EXCEPT:
  - DROPS the leaking `wm_opponent_serve` block.  (Diagnostic: removing it returns OOF
    class-0 f1 from 0.857 -> 0.443 = healthy; it was rally-identifying via fine
    match x opponent x serve_spin x serve_point cells with no leave-self-out -> terminal leak.)
  - Keeps the CLEAN richer blocks (each forward-verified non-leaking):
      wm_player_prevpoint  (zone-TRANSITION prior P(zone | prev1_pointId, striker))
      wm_player_score      (score-conditioned zone prior)
      wm_player_ssbucket   (rally-length(since-serve)-bucket-conditioned zone prior)
      wm_opponent          (opponent's within-match landing tendency, plain)
  - Plus all 6 v701 base blocks + transd_mp_point (truncation-matched prior-rally striker
    point distribution) + raw context + baseline residual.

All transductive; leave-self-out for striker blocks; OOV-safe (built from the rally's OWN
match strokes, no global player-TE).  Target-stroke-own fields masked to -1 (train/test parity).
"""
import numpy as np
import pandas as pd

N_PT = 10
ALPHA = 0.5


def _smooth_dist(cnt_2d, prior_1d, alpha=ALPHA):
    tot = cnt_2d.sum(1, keepdims=True)
    sm = (cnt_2d + alpha * prior_1d[None, :]) / (tot + alpha)
    zero = (tot[:, 0] == 0)
    sm[zero] = prior_1d[None, :]
    return sm.astype(np.float32)


def add_serve_and_seq(df):
    df = df.copy()
    g = df.groupby('rally_uid', sort=False)
    first = g[['spinId', 'pointId', 'positionId', 'strengthId', 'handId']].transform('first')
    df['serve_spin'] = first['spinId'].astype(int)
    df['serve_point'] = first['pointId'].astype(int)
    df['serve_pos'] = first['positionId'].astype(int)
    df['since_serve'] = (df['strikeNumber'] - 1).astype(int)
    df['since_serve_parity'] = (df['since_serve'] % 2).astype(int)
    df['ss_bucket'] = df['since_serve'].clip(0, 4).astype(int)
    smax = np.maximum(df['scoreSelf'].values, df['scoreOther'].values)
    sphase = np.where(smax >= 10, 3, np.where(smax >= 7, 2, np.where(smax >= 3, 1, 0)))
    df['score_phase'] = sphase.astype(int)
    df['score_diff_sign'] = np.sign(df['scoreSelf'] - df['scoreOther']).astype(int)
    for k in [1, 2, 3]:
        df[f'prev{k}_pointId'] = g['pointId'].shift(k).fillna(-1).astype(int)
        df[f'prev{k}_actionId'] = g['actionId'].shift(k).fillna(-1).astype(int)
        df[f'prev{k}_spinId'] = g['spinId'].shift(k).fillna(-1).astype(int)
        df[f'prev{k}_positionId'] = g['positionId'].shift(k).fillna(-1).astype(int)
        df[f'prev{k}_handId'] = g['handId'].shift(k).fillna(-1).astype(int)
        df[f'prev{k}_strengthId'] = g['strengthId'].shift(k).fillna(-1).astype(int)
    return df


def build_within_match_conditionals(stats_df, target_df, global_prior,
                                     self_subtract=False, opp_stats_df=None,
                                     include_blocks=None):
    """CLEAN within-match conditionals.  include_blocks: optional set of block names to
    build (default = all clean blocks; wm_opponent_serve is NEVER built)."""
    if include_blocks is None:
        include_blocks = set(CLEAN_WM_KEYS)
    out = {}
    pt = stats_df['pointId'].values.astype(int)
    onehot = np.eye(N_PT, dtype=np.float64)[pt]

    def agg_by(sdf_src, oh_src, keys_stats, keys_target, name, self_oh=None):
        kcols = [f'_k{i}' for i in range(len(keys_stats))]
        sdf = pd.DataFrame({c: sdf_src[k].values for c, k in zip(kcols, keys_stats)})
        for j in range(N_PT):
            sdf[f'_oh{j}'] = oh_src[:, j]
        summ = sdf.groupby(kcols, sort=False).sum()
        tdf = pd.DataFrame({c: target_df[k].values for c, k in zip(kcols, keys_target)})
        merged = tdf.merge(summ, left_on=kcols, right_index=True, how='left')
        cnt = merged[[f'_oh{j}' for j in range(N_PT)]].fillna(0.0).values
        if self_oh is not None:
            cnt = np.clip(cnt - self_oh, 0, None)
        out[name] = _smooth_dist(cnt, global_prior)
        out[name + '_n'] = cnt.sum(1).astype(np.float32)

    self_oh = None
    if self_subtract:
        tpt = target_df['pointId'].values.astype(int)
        self_oh = np.eye(N_PT, dtype=np.float64)[tpt]

    # ---- v701 base blocks (verbatim) ----
    agg_by(stats_df, onehot, ['match', 'gamePlayerId'],
           ['match', 'tgt_player'], 'wm_player', self_oh)
    agg_by(stats_df, onehot, ['match', 'gamePlayerId', 'serve_spin', 'serve_point'],
           ['match', 'tgt_player', 'serve_spin', 'serve_point'], 'wm_player_serve', self_oh)
    agg_by(stats_df, onehot, ['match', 'gamePlayerId', 'prev1_positionId'],
           ['match', 'tgt_player', 'prev1_positionId'], 'wm_player_prevpos', self_oh)
    agg_by(stats_df, onehot, ['match', 'gamePlayerId', 'prev1_spinId'],
           ['match', 'tgt_player', 'prev1_spinId'], 'wm_player_prevspin', self_oh)
    agg_by(stats_df, onehot, ['match', 'gamePlayerId', 'since_serve_parity'],
           ['match', 'tgt_player', 'since_serve_parity'], 'wm_player_parity', self_oh)
    agg_by(stats_df, onehot, ['match', 'serve_spin', 'serve_point'],
           ['match', 'serve_spin', 'serve_point'], 'wm_match_serve', None)

    # ---- CLEAN richer blocks (forward-verified) ----
    if 'wm_player_prevpoint' in include_blocks:
        agg_by(stats_df, onehot, ['match', 'gamePlayerId', 'prev1_pointId'],
               ['match', 'tgt_player', 'prev1_pointId'], 'wm_player_prevpoint', self_oh)
    if 'wm_player_score' in include_blocks:
        agg_by(stats_df, onehot, ['match', 'gamePlayerId', 'score_phase'],
               ['match', 'tgt_player', 'score_phase'], 'wm_player_score', self_oh)
    if 'wm_player_ssbucket' in include_blocks:
        agg_by(stats_df, onehot, ['match', 'gamePlayerId', 'ss_bucket'],
               ['match', 'tgt_player', 'ss_bucket'], 'wm_player_ssbucket', self_oh)

    # ---- OPPONENT plain (NO serve-cross; that crossed version leaks) ----
    if 'wm_opponent' in include_blocks:
        opp_src = opp_stats_df if opp_stats_df is not None else stats_df
        opp_oh = np.eye(N_PT, dtype=np.float64)[opp_src['pointId'].values.astype(int)]
        agg_by(opp_src, opp_oh, ['match', 'gamePlayerId'],
               ['match', 'tgt_opp'], 'wm_opponent', None)
    # wm_opponent_serve INTENTIONALLY OMITTED (terminal leak)
    return out


# CLEAN block order (NO wm_opponent_serve)
CLEAN_WM_KEYS = ['wm_player', 'wm_player_serve', 'wm_player_prevpos', 'wm_player_prevspin',
                 'wm_player_parity', 'wm_match_serve', 'wm_player_prevpoint',
                 'wm_player_score', 'wm_player_ssbucket', 'wm_opponent']


def assemble_numeric(df, wm, baseline_point=None, transd_block=None, transd_cols=None,
                     wm_keys=None):
    if wm_keys is None:
        wm_keys = [k for k in CLEAN_WM_KEYS if k in wm]
    cols, names = [], []
    for key in wm_keys:
        cols.append(wm[key])
        names += [f'{key}_p{j}' for j in range(N_PT)]
        cols.append(wm[key + '_n'][:, None])
        names.append(f'{key}_n')
    raw_int = ['since_serve', 'since_serve_parity', 'ss_bucket', 'score_phase',
               'score_diff_sign', 'serve_spin', 'serve_point', 'serve_pos',
               'spinId', 'strengthId', 'handId', 'positionId',
               'prev1_pointId', 'prev1_actionId', 'prev1_spinId', 'prev1_positionId',
               'prev1_handId', 'prev1_strengthId',
               'prev2_pointId', 'prev2_actionId', 'prev2_spinId', 'prev2_positionId',
               'prev3_pointId', 'prev3_actionId',
               'scoreSelf', 'scoreOther', 'sex', 'numberGame']
    for c in raw_int:
        cols.append(df[c].values.astype(np.float32)[:, None])
        names.append(c)
    df = df.copy()
    df['_act_spin'] = (df['prev1_actionId'].clip(-1, 18) + 1) * 10 + df['prev1_spinId'].clip(-1, 8) + 1
    df['_act_pos'] = (df['prev1_actionId'].clip(-1, 18) + 1) * 12 + df['positionId'].clip(0, 11)
    df['_prevpt_prevpos'] = (df['prev1_pointId'].clip(-1, 9) + 1) * 12 + df['prev1_positionId'].clip(0, 11)
    for c in ['_act_spin', '_act_pos', '_prevpt_prevpos']:
        cols.append(df[c].values.astype(np.float32)[:, None])
        names.append(c)
    if transd_block is not None:
        cols.append(transd_block.astype(np.float32))
        names += list(transd_cols)
    if baseline_point is not None:
        cols.append(baseline_point.astype(np.float32))
        names += [f'base_p{j}' for j in range(N_PT)]
    X = np.concatenate(cols, axis=1).astype(np.float32)
    return X, names
