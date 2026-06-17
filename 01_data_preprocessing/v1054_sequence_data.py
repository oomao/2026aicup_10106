"""Data prep for v1054 serious transformer.

Reuses the v1021 anti-leak prefix construction + canonical NT ordering (bit-exact
with v260/y_action.npy & v681/y_point.npy), but enriches per-stroke tokens and
numeric scalars and adds an OOV-safe "target context" vector (role parity of the
K+1 striker, sex) that is appended to the pooled prefix representation.

Anti-leak (verified): for a target row at strikeNumber=s, the encoder sees ONLY
strokes 1..(s-1) of the same rally. Target stroke (K+1) and anything after it
NEVER enter the encoder. serverGetPoint is NEVER used as input.

Canonical NT target rows: strike>=2 & actionId<15, sorted by (rally_uid,
strikeNumber). n=69710.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("E:/AICUP_O")

PAD = 0
MASK = 1
OFFSET = 2  # reserved [PAD, MASK]

# per-stroke categorical fields -> raw max (min >=0)
FIELDS = {
    "actionId": 18,
    "pointId": 9,
    "spinId": 5,
    "strengthId": 3,
    "handId": 2,
    "strikeId": 4,
    "positionId": 3,
}
VOCAB = {f: (mx + 1 + OFFSET) for f, mx in FIELDS.items()}
FIELD_LIST = list(FIELDS.keys())

MAX_LEN = 24  # cap visible prefix length (rallies longer truncated to last MAX_LEN)

# numeric per-stroke scalars
#   parity (strike%2), score diff/10, score sum/20, is_serve, sex(0/1),
#   strike_norm (strikeNumber/20), is_last_visible (filled at batch time? -> no,
#   computed per-prefix later). We keep 6 stroke-level numerics here.
N_NUM = 6


def load_both():
    tr = pd.read_csv(ROOT / "data/train.csv")
    te = pd.read_csv(ROOT / "data/test.csv")
    tr = tr.sort_values(["rally_uid", "strikeNumber"]).reset_index(drop=True)
    te = te.sort_values(["rally_uid", "strikeNumber"]).reset_index(drop=True)
    return tr, te


def _stroke_token_matrix(df: pd.DataFrame):
    toks = {}
    for f in FIELDS:
        toks[f] = (df[f].to_numpy().astype(np.int64) + OFFSET)
    sn = df["strikeNumber"].to_numpy()
    parity = (sn % 2).astype(np.float32)
    sdiff = (df["scoreSelf"].to_numpy() - df["scoreOther"].to_numpy()).astype(np.float32) / 10.0
    ssum = (df["scoreSelf"].to_numpy() + df["scoreOther"].to_numpy()).astype(np.float32) / 20.0
    is_serve = (sn == 1).astype(np.float32)
    sexn = (df["sex"].to_numpy() - 1).astype(np.float32)
    strike_norm = np.clip(sn / 20.0, 0, 2).astype(np.float32)
    num = np.stack([parity, sdiff, ssum, is_serve, sexn, strike_norm], axis=1)  # (N,6)
    return toks, num


def build_sequences():
    tr, te = load_both()
    out = {}
    for name, df in [("train", tr), ("test", te)]:
        toks, num = _stroke_token_matrix(df)
        field_arr = np.stack([toks[f] for f in FIELDS], axis=1)  # (N, n_field)
        out[name] = dict(
            field=field_arr, num=num.astype(np.float32),
            rally=df["rally_uid"].to_numpy(),
            strike=df["strikeNumber"].to_numpy().astype(np.int64),
            striker=df["gamePlayerId"].to_numpy().astype(np.int64),
            opp=df["gamePlayerOtherId"].to_numpy().astype(np.int64),
            match=df["match"].to_numpy(),
            sex=(df["sex"].to_numpy() - 1).astype(np.float32),
            df=df,
        )
    return out


def canonical_target_rows(tr_pack):
    strike = tr_pack["strike"]
    field = tr_pack["field"]
    action_raw = field[:, FIELD_LIST.index("actionId")] - OFFSET
    mask = (strike >= 2) & (action_raw < 15)
    return np.where(mask)[0]


def build_rally_index(pack):
    rally = pack["rally"]
    idx = {}
    if len(rally) == 0:
        return idx
    starts = np.r_[0, np.where(rally[1:] != rally[:-1])[0] + 1]
    ends = np.r_[starts[1:], len(rally)]
    for s, e in zip(starts, ends):
        idx[rally[s]] = (s, e)
    return idx


def target_context(pack, rows_global):
    """OOV-safe context for the K+1 target stroke: [parity_of_target, sex].

    parity = strikeNumber(target) % 2 ; this is the role (server/receiver) of the
    K+1 striker -- legitimate (we know whose turn the next stroke is) and OOV-safe
    (no player identity). Returns (B, 2) float.
    """
    strike = pack["strike"]
    sex = pack["sex"]
    par = (strike[rows_global] % 2).astype(np.float32)
    sx = sex[rows_global].astype(np.float32)
    return np.stack([par, sx], axis=1)


N_CTX = 2


def make_prefix_batch(pack, rally_idx, rows_global, max_len=MAX_LEN):
    """For each target row i at strikeNumber s, build prefix = strokes 1..(s-1).

    Right-aligned (most recent stroke at the LAST position), PAD=0 at the front.
    Returns field_seq (B,max_len,n_field) int, num_seq (B,max_len,N_NUM) float,
    lengths (B,) int.
    """
    field = pack["field"]; num = pack["num"]; strike = pack["strike"]; rally = pack["rally"]
    n_field = field.shape[1]
    B = len(rows_global)
    fseq = np.zeros((B, max_len, n_field), dtype=np.int64)
    nseq = np.zeros((B, max_len, N_NUM), dtype=np.float32)
    lengths = np.zeros(B, dtype=np.int64)
    for b, gi in enumerate(rows_global):
        ru = rally[gi]
        s, e = rally_idx[ru]
        s_strike = strike[gi]
        slc_strike = strike[s:e]
        pidx = np.where(slc_strike < s_strike)[0] + s
        if len(pidx) == 0:
            lengths[b] = 0
            continue
        pidx = pidx[-max_len:]
        L = len(pidx)
        fseq[b, max_len - L:] = field[pidx]
        nseq[b, max_len - L:] = num[pidx]
        lengths[b] = L
    return fseq, nseq, lengths


def player_disjoint_folds(strikers, n_splits=5, seed=42):
    rng = np.random.default_rng(seed)
    players = np.unique(strikers)
    rng.shuffle(players)
    pl_fold = {p: (i % n_splits) for i, p in enumerate(players)}
    return np.array([pl_fold[p] for p in strikers], dtype=np.int64)
