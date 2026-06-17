"""Selective POINT overlay on agree1 baseline (rule 101) using v701 within-match model.

Mechanism: flip point cells where the within-match model confidently disagrees with the
baseline AND the flip is INTO a low-support drag zone (1,3,4,5). Validated FIRST on the
PLAYER-held-out OOF (does the flip rule improve macro-F1?), then applied to test.

Constraints: cap <= 100 cells, p0_rate in [0.235, 0.290], rule-96 drift check.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score

ROOT = Path('E:/AICUP_O')
OUT = ROOT / 'models/v701_withinmatch_point/outputs'
N_PT = 10
DRAG = {1, 3, 4, 5}

def log(*a): print(*a, flush=True)

# ---- Load OOF (player-held-out) + baseline OOF + y ----
oof = np.load(OUT / 'oof_point.npy')              # within-match player-held-out OOF
base_oof = np.load(ROOT / 'models/v85_NEW/outputs/oof_point.npy').astype(np.float32)
y = np.load(OUT.parent.parent / 'v700_point_transductive/outputs/y_point.npy')  # canonical y
assert len(y) == len(oof) == len(base_oof)

base_arg = base_oof.argmax(1)
base_f1 = f1_score(y, base_arg, labels=list(range(N_PT)), average='macro', zero_division=0)

# blend candidate (alpha tuned on player-held-out: 0.2 was best macroF1)
def blend(a, bo, mo):
    bl = (1 - a) * bo + a * mo
    return bl / bl.sum(1, keepdims=True)

# ---- Tune the selective FLIP rule on OOF, AT THE CAPPED FLIP BUDGET ----
# The test overlay is capped to 100 cells (rule 101). To make the OOF-predicted delta
# HONEST, tune at the SAME flip fraction: 100/1845 = 5.42% of rallies. On the 69710-row
# OOF that is ~3779 top-confidence flips. We select TOP-margin flips into the allowed zone
# and measure macro-F1 delta there (mirrors the capped test overlay exactly).
TEST_N = 1845
CAP = 100
oof_budget = int(round(CAP / TEST_N * len(y)))  # ~3779
log('=== OOF flip-rule tuning (PLAYER-held-out, capped budget) ===')
log(f'baseline OOF macroF1 = {base_f1:.4f}   OOF flip budget = {oof_budget} (= {CAP}/{TEST_N} fraction)')
best = None
full_rule_delta = None  # also record the UNCAPPED best rule for reference
for a in [0.2, 0.3, 0.5]:
    bl = blend(a, base_oof, oof)
    bl_arg = bl.argmax(1)
    wm_sorted = np.sort(oof, 1)
    wm_margin = wm_sorted[:, -1] - wm_sorted[:, -2]
    for allow_z0 in [False, True]:
        target_set = DRAG | ({0} if allow_z0 else set())
        cand = (bl_arg != base_arg) & np.isin(bl_arg, list(target_set)) & (oof.argmax(1) == bl_arg)
        if cand.sum() == 0:
            continue
        # full (uncapped) delta for reference
        nf = base_arg.copy(); nf[cand] = bl_arg[cand]
        f1_full = f1_score(y, nf, labels=list(range(N_PT)), average='macro', zero_division=0)
        # capped: take top-`oof_budget` by within-match margin among candidates
        conf = wm_margin.copy(); conf[~cand] = -1
        topk = np.argsort(-conf)[:oof_budget]
        cap_mask = np.zeros(len(y), bool); cap_mask[topk] = True; cap_mask &= cand
        ncap = base_arg.copy(); ncap[cap_mask] = bl_arg[cap_mask]
        f1_cap = f1_score(y, ncap, labels=list(range(N_PT)), average='macro', zero_division=0)
        # implied margin threshold at the cap (for applying to test)
        margin_at_cap = float(conf[topk].min()) if len(topk) else 1.0
        rec = dict(alpha=a, allow_z0=allow_z0, margin=margin_at_cap,
                   n_flip_capped=int(cap_mask.sum()), n_flip_full=int(cand.sum()),
                   f1_capped=float(f1_cap), delta_capped=float(f1_cap - base_f1),
                   f1_full=float(f1_full), delta_full=float(f1_full - base_f1))
        if best is None or f1_cap > best['f1_capped']:
            best = rec
log('Best OOF flip rule (capped):', json.dumps(best, indent=2))
# Map fields used downstream
best['f1'] = best['f1_capped']; best['delta'] = best['delta_capped']; best['n_flip'] = best['n_flip_capped']

# Per-zone F1 at best CAPPED rule
a, margin, allow_z0 = best['alpha'], best['margin'], best['allow_z0']
bl = blend(a, base_oof, oof); bl_arg = bl.argmax(1)
wm_sorted = np.sort(oof, 1); wm_margin = wm_sorted[:, -1] - wm_sorted[:, -2]
target_set = DRAG | ({0} if allow_z0 else set())
cand = (bl_arg != base_arg) & np.isin(bl_arg, list(target_set)) & (oof.argmax(1) == bl_arg)
conf = wm_margin.copy(); conf[~cand] = -1
topk = np.argsort(-conf)[:oof_budget]
flip_mask = np.zeros(len(y), bool); flip_mask[topk] = True; flip_mask &= cand
new_arg = base_arg.copy(); new_arg[flip_mask] = bl_arg[flip_mask]
pc_b = f1_score(y, base_arg, labels=list(range(N_PT)), average=None, zero_division=0)
pc_n = f1_score(y, new_arg, labels=list(range(N_PT)), average=None, zero_division=0)
log('Per-zone F1 (base -> overlay):')
for c in range(N_PT):
    flag = ' <DRAG>' if c in DRAG else ''
    log(f'  z{c}: {pc_b[c]:.4f} -> {pc_n[c]:.4f}  ({pc_n[c]-pc_b[c]:+.4f}){flag}')

# ---- Apply SAME rule to TEST against agree1 baseline submission ----
log('\n=== Applying overlay to TEST (agree1 baseline) ===')
sub = pd.read_csv(ROOT / '_NEW_PUBLIC/result/lb_history/0.4132214_day39_slot1_point_ext_agree1/sub_day39_point_ext_agree1_m0175.csv')
sub = sub.sort_values('rally_uid').reset_index(drop=True)
test_uids = np.load(OUT / 'test_rally_uids.npy')
assert np.array_equal(sub.rally_uid.values, np.sort(test_uids)), 'uid order mismatch'

wm_test = np.load(OUT / 'test_point.npy')
base_test = np.load(ROOT / 'models/v85_NEW/outputs/test_point.npy').astype(np.float32)
# align test_point order to sorted uids (it was saved in sorted-uid order already)
order = np.argsort(test_uids)
wm_test = wm_test[order]
# base_test is in v85_NEW order = sorted test rally order; verify length
assert len(base_test) == len(sub) == len(wm_test)

bl_t = blend(a, base_test, wm_test); bl_t_arg = bl_t.argmax(1)
wm_t_sorted = np.sort(wm_test, 1); wm_t_margin = wm_t_sorted[:, -1] - wm_t_sorted[:, -2]
base_t_arg = sub.pointId.values  # agree1 baseline point argmax
flip_t = (bl_t_arg != base_t_arg) & (wm_t_margin >= margin) & np.isin(bl_t_arg, list(target_set)) & (wm_test.argmax(1) == bl_t_arg)

# action=0 -> point=0 constraint: never flip those rows away from 0
a0_mask = sub.actionId.values == 0
flip_t = flip_t & (~a0_mask)

# Cap at 100 cells: keep highest-confidence flips
if flip_t.sum() > 100:
    conf = wm_t_margin.copy(); conf[~flip_t] = -1
    keep_idx = np.argsort(-conf)[:100]
    new_flip = np.zeros_like(flip_t); new_flip[keep_idx] = True
    flip_t = new_flip & flip_t

new_pt = base_t_arg.copy()
new_pt[flip_t] = bl_t_arg[flip_t]
p0_rate = (new_pt == 0).mean()
n_flip_t = int(flip_t.sum())
log(f'Test flips: {n_flip_t}, p0_rate {p0_rate:.4f} (band [0.235,0.290])')
log('Flip target-zone distribution: ' + str(pd.Series(new_pt[flip_t]).value_counts().sort_index().to_dict()))

# rule-96 drift: corr(base_test, wm_test) vs corr(base_oof, oof) on argmax agreement proxy
def soft_corr(p, q):
    return np.corrcoef(p.ravel(), q.ravel())[0, 1]
corr_oof = soft_corr(base_oof, oof)
corr_test = soft_corr(base_test, wm_test)
log(f'rule-96 drift: corr_test={corr_test:.4f}  corr_oof={corr_oof:.4f}  diff={corr_test-corr_oof:+.4f} '
    f'(SAFE if diff < +0.03)')

# ---- write overlay submission ----
out_sub = sub.copy()
out_sub['pointId'] = new_pt.astype(int)
# verify a0->p0 still holds
viol = int(((out_sub.actionId == 0) & (out_sub.pointId != 0)).sum())
log(f'action0->point0 violations: {viol}')
out_path = OUT / 'sub_v701_withinmatch_point_overlay.csv'
out_sub.to_csv(out_path, index=False)
log(f'Saved overlay sub -> {out_path}')

# predicted composite LB delta (point axis only changes)
# OOF flip delta on macroF1_point translates to composite via 0.4 weight; LB transfer for
# point overlays historically ~0.4-0.6 of OOF (rule 101 Day38 +0.008 from ~40 cells).
oof_delta = best['delta']
log(f'\nOOF point macroF1 delta (overlay rule) = {oof_delta:+.4f}')
log(f'Composite contribution (0.4 x point delta) = {0.4*oof_delta:+.4f} (OOF-side, full-flip-rule)')
log('NOTE: test overlay is CAPPED/selective -> realized composite delta scales with n_flip_t/n_flip_oof.')

summary = json.load(open(OUT / 'summary.json'))
summary['overlay'] = dict(
    best_rule=best, n_test_flips=n_flip_t, test_p0_rate=float(p0_rate),
    corr_oof=float(corr_oof), corr_test=float(corr_test),
    drift=float(corr_test - corr_oof), a0_violations=viol,
    per_zone_oof={f'z{c}': [float(pc_b[c]), float(pc_n[c])] for c in range(N_PT)},
)
json.dump(summary, open(OUT / 'summary.json', 'w'), indent=2)
log('Updated summary.json with overlay block.')
