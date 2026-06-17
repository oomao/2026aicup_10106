"""Selective POINT overlay on the RECORD sub (sub_day48_v1080_OOVgated.csv) using the
v1341 CLEAN within-match point model (rule 101).  Action + server kept byte-identical.

Flip rule tuned on the PLAYER-held-out OOF at the capped budget (mirrors test cap), then
applied to test with margin gate + p0 band + rule-96 + action0->point0 constraint.

Two flip strategies evaluated (pick the one with best OOF capped delta):
  A) DRAG-only (flip only into drag zones {1,3,4,5}; v701 style)
  B) ALL-zone (flip into any zone where the clean model confidently disagrees), still gated
     by margin + p0 band  (richer model may correct over-predicted common zones too).
"""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score

ROOT = Path('E:/AICUP_O')
OUT = ROOT / 'models/v1341_transd_point_clean/outputs'
N_PT = 10
DRAG = {1, 3, 4, 5}
TEST_N = 1845
CAP = 100
P0_LO, P0_HI = 0.18, 0.27  # flat region (record sub p0 measured below; keep within)

RECORD_SUB = ROOT / 'result/staging_day48/sub_day48_v1080_OOVgated.csv'
STAGE_OUT = ROOT / 'result/staging_day48/sub_day48_SPRINT_point.csv'


def log(*a): print(*a, flush=True)


def main(variant):
    log(f'=== v1341 overlay on RECORD sub | variant={variant} ===')
    oof = np.load(OUT / f'oof_point_{variant}.npy').astype(np.float32)
    test_probs = np.load(OUT / f'test_point_{variant}.npy').astype(np.float32)
    test_uids = np.load(OUT / 'test_rally_uids.npy')
    y = np.load(OUT / 'y_point.npy')
    base_oof = np.load(ROOT / 'models/v85_NEW/outputs/oof_point.npy').astype(np.float32)
    base_test = np.load(ROOT / 'models/v85_NEW/outputs/test_point.npy').astype(np.float32)
    v701_oof = np.load(ROOT / 'models/v701_withinmatch_point/outputs/oof_point.npy').astype(np.float32)

    sub = pd.read_csv(RECORD_SUB).sort_values('rally_uid').reset_index(drop=True)
    assert np.array_equal(sub.rally_uid.values, np.sort(test_uids)), 'uid order mismatch'
    # align our test_probs to the sorted-uid (sub) order
    order = np.argsort(test_uids)
    test_probs_s = test_probs[order]
    base_test_s = base_test  # v85_NEW saved in sorted-uid order; verify length
    assert len(base_test_s) == len(sub) == len(test_probs_s)

    base_arg = base_oof.argmax(1)
    base_f1 = f1_score(y, base_arg, labels=list(range(N_PT)), average='macro', zero_division=0)
    # what the RECORD sub's point macroF1 looks like on OOF is not directly available (it's a
    # different baseline). We tune the overlay on the v85_NEW OOF (the residual base the model
    # learned from), then APPLY to the record sub. Report OOF macroF1 delta over v85_NEW base.

    def blend(a, bo, mo):
        bl = (1 - a) * bo + a * mo
        return bl / bl.sum(1, keepdims=True)

    oof_budget = int(round(CAP / TEST_N * len(y)))
    log(f'v85_NEW OOF base macroF1={base_f1:.4f}; OOF flip budget={oof_budget} (={CAP}/{TEST_N})')

    best = None
    wm_sorted = np.sort(oof, 1)
    wm_margin = wm_sorted[:, -1] - wm_sorted[:, -2]
    for a in [0.2, 0.3, 0.5, 0.7]:
        bl = blend(a, base_oof, oof)
        bl_arg = bl.argmax(1)
        for strat, zones in [('DRAG', DRAG), ('ALL', set(range(1, N_PT)))]:
            # never flip INTO terminal 0 (preserve p0); only flip among non-terminal landing zones
            cand = (bl_arg != base_arg) & np.isin(bl_arg, list(zones)) & (oof.argmax(1) == bl_arg)
            if cand.sum() == 0:
                continue
            conf = wm_margin.copy(); conf[~cand] = -1
            topk = np.argsort(-conf)[:oof_budget]
            cap_mask = np.zeros(len(y), bool); cap_mask[topk] = True; cap_mask &= cand
            ncap = base_arg.copy(); ncap[cap_mask] = bl_arg[cap_mask]
            f1_cap = f1_score(y, ncap, labels=list(range(N_PT)), average='macro', zero_division=0)
            margin_at_cap = float(conf[topk].min()) if len(topk) else 1.0
            margin_at_cap = max(margin_at_cap, 0.10)  # rule: margin floor 0.10
            rec = dict(alpha=a, strat=strat, margin=margin_at_cap, zones=sorted(zones),
                       n_flip_capped=int(cap_mask.sum()), n_flip_full=int(cand.sum()),
                       f1_capped=float(f1_cap), delta_capped=float(f1_cap - base_f1))
            if best is None or f1_cap > best['f1_capped']:
                best = rec
    log('Best OOF flip rule (capped): ' + json.dumps(best))

    # per-zone f1 at best rule
    a, margin = best['alpha'], best['margin']
    zones = set(best['zones'])
    bl = blend(a, base_oof, oof); bl_arg = bl.argmax(1)
    cand = (bl_arg != base_arg) & np.isin(bl_arg, list(zones)) & (oof.argmax(1) == bl_arg)
    conf = wm_margin.copy(); conf[~cand] = -1
    topk = np.argsort(-conf)[:oof_budget]
    flip_mask = np.zeros(len(y), bool); flip_mask[topk] = True; flip_mask &= cand
    new_arg = base_arg.copy(); new_arg[flip_mask] = bl_arg[flip_mask]
    pc_b = f1_score(y, base_arg, labels=list(range(N_PT)), average=None, zero_division=0)
    pc_n = f1_score(y, new_arg, labels=list(range(N_PT)), average=None, zero_division=0)
    log('Per-zone f1 (v85 base -> overlay):')
    for c in range(N_PT):
        flag = ' <DRAG>' if c in DRAG else ''
        log(f'  z{c}: {pc_b[c]:.4f} -> {pc_n[c]:.4f} ({pc_n[c]-pc_b[c]:+.4f}){flag}')

    # ---- apply to TEST against the RECORD sub ----
    bl_t = blend(a, base_test_s, test_probs_s); bl_t_arg = bl_t.argmax(1)
    wm_t_sorted = np.sort(test_probs_s, 1); wm_t_margin = wm_t_sorted[:, -1] - wm_t_sorted[:, -2]
    base_t_arg = sub.pointId.values
    flip_t = ((bl_t_arg != base_t_arg) & (wm_t_margin >= margin) &
              np.isin(bl_t_arg, list(zones)) & (test_probs_s.argmax(1) == bl_t_arg))
    a0_mask = sub.actionId.values == 0
    flip_t = flip_t & (~a0_mask)
    if flip_t.sum() > CAP:
        conf = wm_t_margin.copy(); conf[~flip_t] = -1
        keep = np.argsort(-conf)[:CAP]
        nf = np.zeros_like(flip_t); nf[keep] = True
        flip_t = nf & flip_t
    new_pt = base_t_arg.copy()
    new_pt[flip_t] = bl_t_arg[flip_t]
    p0 = float((new_pt == 0).mean())
    n_flip = int(flip_t.sum())
    log(f'\nTEST flips={n_flip}, p0_rate={p0:.4f} (band [{P0_LO},{P0_HI}])')
    log('Flip into-zone dist: ' + str(pd.Series(new_pt[flip_t]).value_counts().sort_index().to_dict()))
    log('Flip from-zone dist: ' + str(pd.Series(base_t_arg[flip_t]).value_counts().sort_index().to_dict()))

    # rule-96 (vs v85_NEW, the residual base)
    corr_oof = float(np.corrcoef(base_oof.ravel(), oof.ravel())[0, 1])
    corr_test = float(np.corrcoef(base_test_s.ravel(), test_probs_s.ravel())[0, 1])
    log(f'rule-96: corr_test={corr_test:.4f} corr_oof={corr_oof:.4f} diff={corr_test-corr_oof:+.4f} '
        f'(SAFE if < +0.03)')

    out_sub = sub.copy()
    out_sub['pointId'] = new_pt.astype(int)
    viol = int(((out_sub.actionId == 0) & (out_sub.pointId != 0)).sum())
    log(f'action0->point0 violations: {viol}')
    # action + server byte-identical check
    rec_sub = pd.read_csv(RECORD_SUB).sort_values('rally_uid').reset_index(drop=True)
    assert np.array_equal(out_sub.actionId.values, rec_sub.actionId.values), 'action changed!'
    assert np.array_equal(out_sub.serverGetPoint.values, rec_sub.serverGetPoint.values), 'server changed!'

    p0_ok = P0_LO <= p0 <= P0_HI
    rule96_ok = (corr_test - corr_oof) < 0.03
    deploy_ok = p0_ok and rule96_ok and viol == 0 and n_flip > 0
    log(f'\nDEPLOY-READY: {deploy_ok} (p0_ok={p0_ok}, rule96_ok={rule96_ok}, viol={viol}, n_flip={n_flip})')
    if deploy_ok:
        out_sub.to_csv(STAGE_OUT, index=False)
        log(f'STAGED -> {STAGE_OUT}')
    else:
        log('NOT staged (gate failed).')

    return dict(variant=variant, best_rule=best, n_flip=n_flip, p0=p0,
                corr_oof=corr_oof, corr_test=corr_test, rule96_diff=corr_test - corr_oof,
                a0_violations=viol, oof_delta=best['delta_capped'], deploy_ready=deploy_ok,
                per_zone={f'z{c}': [float(pc_b[c]), float(pc_n[c])] for c in range(N_PT)})


if __name__ == '__main__':
    # DEPLOYED variant = BAL (overlay_BAL.json: alpha=0.3, margin=0.286, 41 flips, p0=0.2499).
    # This is the variant that produced the submitted sub_day48_SPRINT_point.csv.
    # (NOBAL gives only 35 flips and does NOT reproduce the submission.)
    # For a SELF-CONTAINED, package-only verified reproduction see
    # 05_assembly_and_outputs/reproduce_final.py.
    variant = sys.argv[1] if len(sys.argv) > 1 else 'BAL'
    res = main(variant)
    json.dump(res, open(OUT / f'overlay_{variant}.json', 'w'), indent=2)
    log('done.')
