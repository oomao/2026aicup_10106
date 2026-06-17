"""v701 WITHIN-MATCH-CONDITIONED POINT model with PLAYER-held-out + MATCH-held-out CV.

Core question: does within-match transductive conditioning lift point macro-F1 and
the low-support zones (1,3,4,5) UNDER PLAYER-held-out (the OOV-honest test)?

Design:
- Canonical alignment n=69710 train_NT (strikeNumber>=2 & actionId<15, sorted rally+strike).
  OOF directly blendable with v85_NEW / agree1 baseline.
- Within-match conditionals built PER-MATCH, leave-self-out (mimics test where matches
  are 100% disjoint from train, so conditioning comes from the rally's own match).
- Baseline point probs (v85_NEW) included as features -> GBDT learns a residual.
- Two CV schemes: GroupKFold by MATCH and GroupKFold by PLAYER.
- Test conditionals built from TEST's OWN match strokes (OOV-safe; 0 train-match overlap).
"""
import sys, time, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import lightgbm as lgb

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import features as F

ROOT = Path('E:/AICUP_O')
OUT = SRC.parent / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)
N_PT = 10
DRAG = [1, 3, 4, 5]
SEED = 42

def log(*a):
    print(*a, flush=True)

t0 = time.time()
log('=== v701 within-match point | start ===')

# ---------- Load ----------
train = pd.read_csv(ROOT / 'data/train.csv').sort_values(['rally_uid', 'strikeNumber']).reset_index(drop=True)
test = pd.read_csv(ROOT / 'data/test.csv').sort_values(['rally_uid', 'strikeNumber']).reset_index(drop=True)
if 'serverGetPoint' in test.columns:
    test = test.drop(columns=['serverGetPoint'])  # anti-leak

train = F.add_serve_and_seq(train)
test = F.add_serve_and_seq(test)

# Canonical train_NT
train_NT = train[(train.strikeNumber >= 2) & (train.actionId < 15)].copy().reset_index(drop=True)
y = train_NT['pointId'].values.astype(int)
log(f'train_NT rows={len(train_NT)}  (canonical n=69710)')
assert len(train_NT) == 69710, 'alignment mismatch!'

global_prior = np.bincount(y, minlength=N_PT).astype(np.float64)
global_prior = global_prior / global_prior.sum()

# ---------- Baseline point (for residual features + final compare) ----------
base_oof = np.load(ROOT / 'models/v85_NEW/outputs/oof_point.npy').astype(np.float32)
base_test = np.load(ROOT / 'models/v85_NEW/outputs/test_point.npy').astype(np.float32)
base_arg = base_oof.argmax(1)
base_f1 = f1_score(y, base_arg, labels=list(range(N_PT)), average='macro', zero_division=0)
base_pc = f1_score(y, base_arg, labels=list(range(N_PT)), average=None, zero_division=0)
log(f'BASELINE v85_NEW point: macroF1={base_f1:.4f}  drag zones ' +
    ' '.join(f'z{c}={base_pc[c]:.3f}' for c in DRAG))

# ---------- Within-match conditional pool ----------
# For TRAIN: pool = ALL train NT-or-terminal strokes (we condition on whoever strikes;
#   the "player landing tendency" pool should include all of that player's strokes in
#   the match where they are the striker). Use the full train (all strokes) as the pool
#   but only NT-or-terminal valid pointId rows. Each row's gamePlayerId IS the striker.
train_pool = train[(train.strikeNumber >= 1)].copy()  # all strokes (incl serve) have pointId
# We aggregate point by (match, gamePlayerId, ...) over the pool. The target row's player
# is gamePlayerId for train_NT (the striker of this NT stroke).
train_NT['tgt_player'] = train_NT['gamePlayerId'].values

# Build the within-match conditionals for train_NT (leave-self-out).
# NOTE: leave-self-out only removes the row's OWN one-hot; other strokes of same
# (match,player,...) remain. This exactly mimics test (we see all OTHER same-match strokes).
log('Building train within-match conditionals (leave-self-out)...')
# Pool used for aggregation = train_NT itself (NT strokes with valid landing).
# Including serves (point at serve) would add the serve landing; keep it simple and
# consistent with target distribution: aggregate over NT strokes only.
wm_train = F.build_within_match_conditionals(
    stats_df=train_NT.assign(gamePlayerId=train_NT['tgt_player']),
    target_df=train_NT,
    global_prior=global_prior,
    self_subtract=True,
)
X_train_extra, feat_names = F.assemble_numeric(train_NT, wm_train, baseline_point=base_oof)
log(f'train feature matrix: {X_train_extra.shape}, {len(feat_names)} features')

# ---------- Test within-match conditionals (from TEST's own match strokes) ----------
test_NT_pool = test[(test.strikeNumber >= 2) & (test.actionId < 15)].copy()
# target rows = last visible stroke per rally; target player = gamePlayerOtherId
last = test.sort_values('strikeNumber').groupby('rally_uid', sort=True).tail(1).copy()
last = last.sort_values('rally_uid').reset_index(drop=True)
last['tgt_player'] = last['gamePlayerOtherId'].astype(int)
# For test conditional pool, aggregate point over ALL test NT strokes keyed by the
# STRIKER (gamePlayerId). Then target rows look up by tgt_player.
log('Building test within-match conditionals (from test own match)...')
wm_test = F.build_within_match_conditionals(
    stats_df=test_NT_pool,  # striker = gamePlayerId, pointId = that stroke's landing
    target_df=last,
    global_prior=global_prior,
    self_subtract=False,  # target stroke is HIDDEN at test, nothing to subtract
)
# assemble test features: 'last' rows describe the LAST VISIBLE stroke; the prev* features
# for the target (K+1) need shifting by one relative to 'last'. We approximate the K+1
# context using the last visible stroke as prev1 — build a synthetic target frame.
# Simpler+correct: the model was trained on NT rows where the ROW itself is the target
# stroke with its prev1..3 = preceding strokes. At test we don't have the target stroke's
# own spin/position (hidden). We must DROP target-stroke-own features at inference and
# rely on prev* + within-match. To keep train/test parity, retrain feature set using
# only PRE-target-known fields. -> see build_target_frame below.
test_uids = last['rally_uid'].values.astype(np.int64)

# ---- Build a TARGET FRAME for test: features known BEFORE the target stroke ----
# prev1 = last visible stroke, prev2 = second-last, etc. serve_* from rally. position of
# target is unknown -> set positionId/spinId/strengthId/handId to -1 for the target row.
def build_test_target_frame(test_df, last_rows):
    rows = []
    g = test_df.groupby('rally_uid', sort=False)
    grp = {rid: sub.sort_values('strikeNumber') for rid, sub in g}
    for r in last_rows.itertuples(index=False):
        rid = r.rally_uid
        sub = grp[rid]
        vis = sub  # visible strokes
        n = len(vis)
        rec = {
            'rally_uid': rid, 'match': int(r.match), 'sex': int(r.sex),
            'numberGame': int(r.numberGame), 'scoreSelf': int(r.scoreSelf),
            'scoreOther': int(r.scoreOther),
            'gamePlayerId': int(r.tgt_player), 'tgt_player': int(r.tgt_player),
            'serve_spin': int(r.serve_spin), 'serve_point': int(r.serve_point),
            'serve_pos': int(r.serve_pos),
            'since_serve': int(vis.strikeNumber.max()),  # target is K+1 -> since_serve=K
            'since_serve_parity': int(vis.strikeNumber.max() % 2),
            # target-stroke-own fields unknown:
            'spinId': -1, 'strengthId': -1, 'handId': -1, 'positionId': -1,
        }
        vv = vis.reset_index(drop=True)
        for k in [1, 2, 3]:
            idx = n - k  # prev1 = last visible
            if idx >= 0:
                rec[f'prev{k}_pointId'] = int(vv.loc[idx, 'pointId'])
                rec[f'prev{k}_actionId'] = int(vv.loc[idx, 'actionId'])
                rec[f'prev{k}_spinId'] = int(vv.loc[idx, 'spinId'])
                rec[f'prev{k}_positionId'] = int(vv.loc[idx, 'positionId'])
                rec[f'prev{k}_handId'] = int(vv.loc[idx, 'handId'])
                rec[f'prev{k}_strengthId'] = int(vv.loc[idx, 'strengthId'])
            else:
                for f_ in ['pointId', 'actionId', 'spinId', 'positionId', 'handId', 'strengthId']:
                    rec[f'prev{k}_{f_}'] = -1
        rows.append(rec)
    return pd.DataFrame(rows)

test_tf = build_test_target_frame(test, last)
# prev1_positionId / prev1_spinId needed by within-match conditional lookup on test:
# wm_test was built with target_df=last whose prev1_* came from last's own shift. For
# consistency, rebuild wm_test using test_tf keys (prev1 = last visible stroke).
wm_test = F.build_within_match_conditionals(
    stats_df=test_NT_pool,
    target_df=test_tf,
    global_prior=global_prior,
    self_subtract=False,
)
X_test_extra, feat_names_t = F.assemble_numeric(test_tf, wm_test, baseline_point=base_test)
assert feat_names == feat_names_t, 'feature name mismatch train/test'
log(f'test feature matrix: {X_test_extra.shape}')

# ---- IMPORTANT: train/test parity. At test the target-stroke-own fields are -1.
# So we must also MASK these fields to -1 in TRAIN (the model never sees the target's
# own spin/pos/hand/strength — only prev* + within-match). Rebuild train features with
# target-own fields masked. ----
train_masked = train_NT.copy()
for c in ['spinId', 'strengthId', 'handId', 'positionId']:
    train_masked[c] = -1
X_train_extra, feat_names = F.assemble_numeric(train_masked, wm_train, baseline_point=base_oof)
log('Re-assembled TRAIN features with target-own fields masked (train/test parity).')

LGB_PARAMS = dict(
    objective='multiclass', num_class=N_PT, n_estimators=600, learning_rate=0.03,
    num_leaves=63, max_depth=-1, min_child_samples=80, subsample=0.8,
    subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0, reg_alpha=1.0,
    class_weight='balanced', device='gpu', gpu_platform_id=0, gpu_device_id=0,
    random_state=SEED, verbose=-1, max_bin=255,
)

def run_cv(groups, scheme_name):
    log(f'\n===== CV scheme: {scheme_name} =====')
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros((len(train_NT), N_PT), dtype=np.float32)
    for fold, (tr, va) in enumerate(gkf.split(X_train_extra, y, groups=groups)):
        clf = lgb.LGBMClassifier(**LGB_PARAMS)
        clf.fit(X_train_extra[tr], y[tr])
        oof[va] = clf.predict_proba(X_train_extra[va]).astype(np.float32)
        log(f'  fold {fold+1}/5 trained ({len(tr)} tr / {len(va)} va)')
    arg = oof.argmax(1)
    f1 = f1_score(y, arg, labels=list(range(N_PT)), average='macro', zero_division=0)
    pc = f1_score(y, arg, labels=list(range(N_PT)), average=None, zero_division=0)
    log(f'  {scheme_name} standalone macroF1={f1:.4f}  drag ' +
        ' '.join(f'z{c}={pc[c]:.3f}' for c in DRAG))
    return oof, f1, pc

# match-held-out
oof_match, f1_match, pc_match = run_cv(train_NT['match'].values, 'MATCH-held-out')
# player-held-out (group by striker = target player)
oof_player, f1_player, pc_player = run_cv(train_NT['gamePlayerId'].values, 'PLAYER-held-out')

# Use player-held-out OOF as the honest production OOF
np.save(OUT / 'oof_point.npy', oof_player)
np.save(OUT / 'oof_point_match.npy', oof_match)

# ---------- Test inference: train on ALL train_NT ----------
log('\nTraining FINAL model on all train_NT for test inference...')
final = lgb.LGBMClassifier(**LGB_PARAMS)
final.fit(X_train_extra, y)
test_point = final.predict_proba(X_test_extra).astype(np.float32)
np.save(OUT / 'test_point.npy', test_point)
np.save(OUT / 'test_rally_uids.npy', test_uids)

# feature importance
imp = sorted(zip(feat_names, final.feature_importances_), key=lambda x: -x[1])[:20]
log('Top-20 features: ' + ', '.join(f'{n}={v}' for n, v in imp))

# ---------- Blend evals (within-match as residual on baseline) ----------
def eval_blend(base_o, model_o, yv, base_t, model_t):
    res = {}
    bb = f1_score(yv, base_o.argmax(1), labels=list(range(N_PT)), average='macro', zero_division=0)
    for a in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        bl = (1 - a) * base_o + a * model_o
        f1 = f1_score(yv, bl.argmax(1), labels=list(range(N_PT)), average='macro', zero_division=0)
        pc = f1_score(yv, bl.argmax(1), labels=list(range(N_PT)), average=None, zero_division=0)
        bl_t = (1 - a) * base_t + a * model_t
        p0 = (bl_t.argmax(1) == 0).mean()
        res[a] = dict(f1=float(f1), drag={c: float(pc[c]) for c in DRAG}, p0_rate=float(p0))
    res['base_f1'] = float(bb)
    return res

blend_player = eval_blend(base_oof, oof_player, y, base_test, test_point)
log('\nBlend (PLAYER-held-out OOF) base+α·within-match:')
log(f'  base_f1={blend_player["base_f1"]:.4f}')
for a in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
    r = blend_player[a]
    log(f'  α={a:.1f} f1={r["f1"]:.4f} Δ={r["f1"]-blend_player["base_f1"]:+.4f} '
        f'p0={r["p0_rate"]:.3f} drag ' + ' '.join(f'z{c}={r["drag"][c]:.3f}' for c in DRAG))

# ---------- Save summary ----------
summary = dict(
    n_train=int(len(train_NT)), n_test=int(len(test_uids)),
    baseline_macroF1=float(base_f1),
    baseline_drag={c: float(base_pc[c]) for c in DRAG},
    standalone_within_match_macroF1_MATCH=float(f1_match),
    standalone_within_match_macroF1_PLAYER=float(f1_player),
    drag_MATCH={c: float(pc_match[c]) for c in DRAG},
    drag_PLAYER={c: float(pc_player[c]) for c in DRAG},
    blend_player=blend_player,
    top_features=[(n, int(v)) for n, v in imp],
    elapsed_sec=round(time.time() - t0, 1),
)
with open(OUT / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
log(f'\nSaved summary.json. Elapsed {summary["elapsed_sec"]}s')
log('=== build done ===')
