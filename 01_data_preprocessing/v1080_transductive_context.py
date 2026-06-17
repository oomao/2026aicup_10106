"""Truncation-matched transductive context vectors for v1080.

Produces, per TARGET ROW, a 16-dim vector:
    mp_action_0 .. mp_action_14   (striker's same-match PRIOR-rally action proportions)
    mp_has                        (1 if the striker has >=1 prior-rally stroke, else 0)

This is the EXACT transductive signal that the v1075 GBDT (player-grouped, OOV-honest)
used to reach f1a 0.3795 (+0.05 over the agnostic floor 0.3283). It is OOV-safe (no
player identity is used as a feature; only the striker's OWN prior in-match behaviour),
truncation-matched (each prior rally is truncated to a visible prefix K_vis ~ test K-vis
PMF, multi-seed averaged), and computable at test time for ~76% of rallies.

Two builders:
  build_train_transductive(): aligned bit-exactly to v1054 canonical_target_rows order
      (= v1075 target_row_order.csv). Reuses v1075.build_prior_style(truncate=True).
  build_test_transductive():  for the 1845 test rallies. Test matches are DISJOINT from
      train, so prior rallies come from test.csv itself. Each test rally is ALREADY
      truncated to its visible length (strikeNumber<=K_vis), so the striker's prior-rally
      action proportions are computed directly from the visible prior rallies (no extra
      truncation needed -- the test prefix IS the truncation). Returns vectors aligned to
      the test rally_uid order produced by infer-test (sorted by rally_uid).

ANTI-LEAK:
  - train: cumulative-EXCLUSIVE over rally_id within (match, gamePlayerId) -> a target
    row at rally_id r only sees the striker's rallies with rally_id < r. Prior rallies
    truncation-matched. Proportions, never counts. Never the target row's own attributes.
  - test: prior rallies = same (match, striker) with rally_id < current. Uses only the
    visible strokes of those prior rallies. serverGetPoint never used.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path("E:/AICUP_O")
V1075_SRC = ROOT / "models/v1075_deconfound_transductive/src"
sys.path.insert(0, str(V1075_SRC))

MP_COLS = [f"mp_action_{c}" for c in range(15)] + ["mp_has"]
N_TRANSDUCTIVE = len(MP_COLS)  # 16
TRUNC_SEEDS = [101, 202, 303, 404, 505]


# ---------------------------------------------------------------------------
# TRAIN: reuse v1075's exact truncation-matched prior-style builder + merge on
# (match, gamePlayerId, rally_id) onto the canonical target-row order.
# ---------------------------------------------------------------------------
def build_train_transductive(target_order_df: pd.DataFrame, test_kvis_pmf: pd.Series):
    """target_order_df: columns [rally_uid, strikeNumber, match, rally_id, gamePlayerId]
    in the exact canonical_target_rows order. Returns (N,16) float32 array."""
    import main as v1075  # v1075/src/main.py (build_prior_style, TRUNC_SEEDS)
    prop_tr, feat_cols = v1075.build_prior_style(truncate=True, test_kvis_pmf=test_kvis_pmf,
                                                 seeds=TRUNC_SEEDS)
    assert feat_cols == MP_COLS, f"feat col mismatch: {feat_cols[:3]}..{feat_cols[-1]}"
    keyed = target_order_df[["match", "gamePlayerId", "rally_id"]].copy()
    merged = keyed.merge(prop_tr, on=["match", "gamePlayerId", "rally_id"], how="left")
    X = merged[MP_COLS].fillna(0.0).astype(np.float32).to_numpy()
    # mp_has is already 0/1; ensure no NaN leaked
    cov = float((X[:, -1] > 0.5).mean())
    return X, cov


# ---------------------------------------------------------------------------
# TEST: derive the K+1 striker per rally (parity rule, identical to v1075
# test_time_coverage), then build the striker's same-match prior-rally action
# proportions from test.csv's VISIBLE strokes (prior rallies already truncated).
# Returned in sorted-by-rally_uid order (matches infer-test output order).
# ---------------------------------------------------------------------------
def _test_striker_per_rally(test: pd.DataFrame):
    """Return DataFrame: rally_uid, match, rally_id, kvis, striker (Int64)."""
    meta = test.groupby("rally_uid").agg(
        match=("match", "first"), rally_id=("rally_id", "first"),
        kvis=("strikeNumber", "max")).reset_index()
    t = test.copy()
    t["parity"] = t["strikeNumber"] % 2
    par_pid = t.groupby(["rally_uid", "parity"])["gamePlayerId"].first().unstack()
    other_of = test.groupby("rally_uid")[["gamePlayerId", "gamePlayerOtherId"]].first()

    def striker_pid(row):
        u, kv = row["rally_uid"], row["kvis"]
        tp = (kv + 1) % 2
        if u in par_pid.index and tp in par_pid.columns and not pd.isna(par_pid.loc[u, tp]):
            return par_pid.loc[u, tp]
        # target striker not visible: it's the OTHER of the server (stroke1,parity1)
        return other_of.loc[u, "gamePlayerOtherId"] if tp == 0 else other_of.loc[u, "gamePlayerId"]

    meta["striker"] = meta.apply(striker_pid, axis=1).astype("Int64")
    return meta


def build_test_transductive(test_csv: Path = ROOT / "data/test.csv"):
    """Returns (uids_sorted, X[len,16] float32, coverage_float).

    For each test rally's K+1 striker, aggregate that striker's action distribution over
    the VISIBLE strokes of all PRIOR rallies (same match, rally_id < current). Test prior
    rallies are already truncated (they are real test rallies with their own K_vis), so the
    visible strokes ARE the truncation-matched sample -- no extra truncation needed.
    """
    test = pd.read_csv(test_csv).sort_values(["rally_uid", "strikeNumber"]).reset_index(drop=True)
    meta = _test_striker_per_rally(test)

    # striker action rows from VISIBLE test strokes (actionId<15 = real action classes;
    # serves stroke1 are actionId 15-18 and excluded, matching the train builder which
    # filters actionId<15).
    strokes = test[test["actionId"] < 15][
        ["match", "rally_id", "gamePlayerId", "actionId"]
    ].copy()
    # count of each action per (match, striker, rally_id)
    cnt = (strokes.groupby(["match", "gamePlayerId", "rally_id"])["actionId"]
           .value_counts().unstack(fill_value=0))
    cnt = cnt.reindex(columns=list(range(15)), fill_value=0)

    # For each target rally we need the SUM over prior rallies (rally_id < current) of the
    # striker's action counts in this match. Build per (match, striker) cumulative arrays.
    rows = []
    # group prior-count table by (match, striker) -> dict rally_id -> count-vector
    grp = {}
    for (m, g, rid), r in cnt.iterrows():
        grp.setdefault((m, g), {})[rid] = r.to_numpy().astype(np.float64)

    for _, mr in meta.iterrows():
        m = mr["match"]; g = mr["striker"]; rid = mr["rally_id"]
        vec = np.zeros(15, dtype=np.float64)
        has = 0
        if pd.notna(g):
            d = grp.get((m, int(g)))
            if d is not None:
                for r_id, cvec in d.items():
                    if r_id < rid:
                        vec += cvec
                tot = vec.sum()
                if tot > 0:
                    vec = vec / tot
                    has = 1
                else:
                    vec[:] = 0.0
        rows.append(np.concatenate([vec, [float(has)]]))
    X = np.array(rows, dtype=np.float32)

    uids = meta["rally_uid"].to_numpy()
    order = np.argsort(uids)
    uids_s = uids[order]; X = X[order]
    cov = float((X[:, -1] > 0.5).mean())
    return uids_s, X, cov


if __name__ == "__main__":
    # quick self-test
    test = pd.read_csv(ROOT / "data/test.csv")
    kvis = test.groupby("rally_uid")["strikeNumber"].max()
    pmf = kvis.value_counts(normalize=True).sort_index()
    order = pd.read_csv(ROOT / "models/v1075_deconfound_transductive/outputs/target_row_order.csv")
    Xtr, cov_tr = build_train_transductive(order, pmf)
    print(f"TRAIN transductive: shape={Xtr.shape} coverage(mp_has)={cov_tr:.3f}")
    uids, Xte, cov_te = build_test_transductive()
    print(f"TEST  transductive: n={len(uids)} shape={Xte.shape} coverage(mp_has)={cov_te:.3f}")
    print(f"  expect train cov ~0.40-0.55, test cov ~0.76 (v1075 reported 0.762)")
    print(f"  sample test rows mp_has sum check: {Xte[:, -1].sum():.0f}/{len(uids)}")
