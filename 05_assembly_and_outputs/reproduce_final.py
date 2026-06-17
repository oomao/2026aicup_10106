"""SELF-CONTAINED reproduction of the final submission (Public 0.4225088 / Private 0.3643962).

Rebuilds the full 3-overlay assembly chain using ONLY files bundled inside this report
package (no E:/AICUP_O/models or _NEW_PUBLIC paths), then asserts every stage matches the
bundled reference CSV byte-for-byte. This is what an organizer re-running our package gets.

  base0 (agree1 production)  --v701 point overlay-->  stage1
  stage1                     --v1080 OOV-gated action->  stage2 (= RECORD sub)
  stage2                     --v1341 drag-zone point overlay-->  FINAL

Inputs (all bundled):
  data/train.csv, data/test.csv
  code/05_assembly_and_outputs/reproduction_npy/{v85_NEW,v701,v1341,v1080}/*.npy
  code/05_assembly_and_outputs/base0_*.csv  (production base anchor)
References asserted against:
  code/05_assembly_and_outputs/{stage1_*,stage2_*,FINAL_*}.csv

Run:  py aicup_final_report/code/05_assembly_and_outputs/reproduce_final.py
Exit code 0 = exact reproduction; nonzero = mismatch.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score

ASM = Path(__file__).resolve().parent          # code/05_assembly_and_outputs/
PKG = ASM.parent.parent                        # aicup_final_report/
NPY = ASM / 'reproduction_npy'
DATA = PKG / 'data'

BASE0  = ASM / 'base0_agree1_D27action_G14v951server_0.4132214.csv'
STAGE1 = ASM / 'stage1_v701_point_overlay_0.4141329.csv'
STAGE2 = ASM / 'stage2_v1080_OOVgated_action_0.4207553.csv'
FINAL  = ASM / 'FINAL_sub_day48_SPRINT_point_0.4225088.csv'

N_PT, TEST_N, CAP = 10, 1845, 100
DRAG = {1, 3, 4, 5}
# The DEPLOYED v1341 point overlay used the BAL (class-balanced) variant
# (overlay_BAL.json: alpha=0.3, margin=0.286, 41 flips, p0=0.2499). NOBAL gives 35 flips
# and does NOT reproduce the submission.
VARIANT = 'BAL'
ok_all = True


def log(*a): print(*a, flush=True)


def blend(a, bo, mo):
    bl = (1 - a) * bo + a * mo
    return bl / bl.sum(1, keepdims=True)


def load_sub(p):
    return pd.read_csv(p).sort_values('rally_uid').reset_index(drop=True)


def assert_match(stage, got, ref_path):
    """got: DataFrame[rally_uid,actionId,pointId,serverGetPoint]; compare to reference CSV."""
    global ok_all
    ref = load_sub(ref_path)
    got = got.sort_values('rally_uid').reset_index(drop=True)
    uid_ok = np.array_equal(got.rally_uid.values, ref.rally_uid.values)
    act_ok = np.array_equal(got.actionId.values.astype(int), ref.actionId.values.astype(int))
    pt_ok  = np.array_equal(got.pointId.values.astype(int),  ref.pointId.values.astype(int))
    srv_ok = np.allclose(got.serverGetPoint.values.astype(float),
                         ref.serverGetPoint.values.astype(float), atol=0, rtol=0)
    n_act = int((got.actionId.values.astype(int) != ref.actionId.values.astype(int)).sum())
    n_pt  = int((got.pointId.values.astype(int)  != ref.pointId.values.astype(int)).sum())
    n_srv = int((~np.isclose(got.serverGetPoint.values.astype(float),
                             ref.serverGetPoint.values.astype(float))).sum())
    good = uid_ok and act_ok and pt_ok and srv_ok
    ok_all = ok_all and good
    log(f'  [{stage}] vs {ref_path.name}: '
        f'uid={uid_ok} action={act_ok}({n_act} diff) point={pt_ok}({n_pt} diff) '
        f'server={srv_ok}({n_srv} diff)  -> {"MATCH" if good else "*** MISMATCH ***"}')
    return ref


# ===== STAGE 1: v701 within-match point overlay on base0 =====================
def stage1_v701(base0):
    oof = np.load(NPY / 'v701/oof_point.npy')
    base_oof = np.load(NPY / 'v85_NEW/oof_point.npy').astype(np.float32)
    y = np.load(NPY / 'v1341/y_point.npy')          # == v700 canonical y (verified equal)
    base_arg = base_oof.argmax(1)
    base_f1 = f1_score(y, base_arg, labels=list(range(N_PT)), average='macro', zero_division=0)
    budget = int(round(CAP / TEST_N * len(y)))
    best = None
    wm_sorted = np.sort(oof, 1); wm_margin = wm_sorted[:, -1] - wm_sorted[:, -2]
    for a in [0.2, 0.3, 0.5]:
        bl_arg = blend(a, base_oof, oof).argmax(1)
        for allow_z0 in [False, True]:
            tset = DRAG | ({0} if allow_z0 else set())
            cand = (bl_arg != base_arg) & np.isin(bl_arg, list(tset)) & (oof.argmax(1) == bl_arg)
            if cand.sum() == 0:
                continue
            conf = wm_margin.copy(); conf[~cand] = -1
            topk = np.argsort(-conf)[:budget]
            cap_mask = np.zeros(len(y), bool); cap_mask[topk] = True; cap_mask &= cand
            ncap = base_arg.copy(); ncap[cap_mask] = bl_arg[cap_mask]
            f1c = f1_score(y, ncap, labels=list(range(N_PT)), average='macro', zero_division=0)
            rec = dict(alpha=a, allow_z0=allow_z0,
                       margin=float(conf[topk].min()) if len(topk) else 1.0, f1=float(f1c))
            if best is None or f1c > best['f1']:
                best = rec
    a, margin, allow_z0 = best['alpha'], best['margin'], best['allow_z0']
    tset = DRAG | ({0} if allow_z0 else set())
    log(f'  stage1 OOF rule: alpha={a} allow_z0={allow_z0} margin={margin:.4f} '
        f'(base OOF f1p={base_f1:.4f} -> {best["f1"]:.4f})')

    tu = np.load(NPY / 'v701/test_rally_uids.npy')
    assert np.array_equal(base0.rally_uid.values, np.sort(tu))
    wm_t = np.load(NPY / 'v701/test_point.npy')[np.argsort(tu)]
    base_t = np.load(NPY / 'v85_NEW/test_point.npy').astype(np.float32)
    bl_t_arg = blend(a, base_t, wm_t).argmax(1)
    mt = np.sort(wm_t, 1); margin_t = mt[:, -1] - mt[:, -2]
    base_pt = base0.pointId.values
    flip = ((bl_t_arg != base_pt) & (margin_t >= margin) &
            np.isin(bl_t_arg, list(tset)) & (wm_t.argmax(1) == bl_t_arg) &
            (base0.actionId.values != 0))
    if flip.sum() > CAP:
        c = margin_t.copy(); c[~flip] = -1
        keep = np.argsort(-c)[:CAP]; nf = np.zeros_like(flip); nf[keep] = True; flip = nf & flip
    new_pt = base_pt.copy(); new_pt[flip] = bl_t_arg[flip]
    log(f'  stage1 TEST flips={int(flip.sum())} p0={float((new_pt==0).mean()):.4f}')
    out = base0.copy(); out['pointId'] = new_pt.astype(int)
    return out


# ===== STAGE 2: v1080 transformer OOV-gated action on stage1 =================
def stage2_v1080(stage1):
    ta = np.load(NPY / 'v1080/test_action.npy')
    tu = np.load(NPY / 'v1080/test_rally_uids.npy')
    o = np.argsort(tu); tu = tu[o]; ta = ta[o]; tf = ta.argmax(1)
    s1 = stage1.sort_values('rally_uid').reset_index(drop=True)
    assert np.array_equal(s1.rally_uid.values, tu)
    prod = s1.actionId.values
    train = pd.read_csv(DATA / 'train.csv'); test = pd.read_csv(DATA / 'test.csv')
    trpl = set(train.gamePlayerId.unique()) | set(train.gamePlayerOtherId.unique())
    t1 = test[test.strikeNumber == 1].drop_duplicates('rally_uid').set_index('rally_uid')
    oov = np.array([(t1.loc[u, 'gamePlayerId'] not in trpl) or
                    (t1.loc[u, 'gamePlayerOtherId'] not in trpl) for u in tu])
    act = np.where(oov, tf, prod)
    pt = s1.pointId.values.copy(); pt[act == 0] = 0
    log(f'  stage2 OOV rallies={int(oov.sum())} action cells changed={int((act!=prod).sum())}')
    return pd.DataFrame({'rally_uid': tu, 'actionId': act.astype(int),
                         'pointId': pt.astype(int), 'serverGetPoint': s1.serverGetPoint.values})


# ===== STAGE 3: v1341 clean drag-zone point overlay on stage2 (RECORD) =======
def stage3_v1341(stage2):
    oof = np.load(NPY / f'v1341/oof_point_{VARIANT}.npy').astype(np.float32)
    test_p = np.load(NPY / f'v1341/test_point_{VARIANT}.npy').astype(np.float32)
    tu = np.load(NPY / 'v1341/test_rally_uids.npy')
    y = np.load(NPY / 'v1341/y_point.npy')
    base_oof = np.load(NPY / 'v85_NEW/oof_point.npy').astype(np.float32)
    base_t = np.load(NPY / 'v85_NEW/test_point.npy').astype(np.float32)
    base_arg = base_oof.argmax(1)
    base_f1 = f1_score(y, base_arg, labels=list(range(N_PT)), average='macro', zero_division=0)
    budget = int(round(CAP / TEST_N * len(y)))
    wm_sorted = np.sort(oof, 1); wm_margin = wm_sorted[:, -1] - wm_sorted[:, -2]
    best = None
    for a in [0.2, 0.3, 0.5, 0.7]:
        bl_arg = blend(a, base_oof, oof).argmax(1)
        for strat, zones in [('DRAG', DRAG), ('ALL', set(range(1, N_PT)))]:
            cand = (bl_arg != base_arg) & np.isin(bl_arg, list(zones)) & (oof.argmax(1) == bl_arg)
            if cand.sum() == 0:
                continue
            conf = wm_margin.copy(); conf[~cand] = -1
            topk = np.argsort(-conf)[:budget]
            cap_mask = np.zeros(len(y), bool); cap_mask[topk] = True; cap_mask &= cand
            ncap = base_arg.copy(); ncap[cap_mask] = bl_arg[cap_mask]
            f1c = f1_score(y, ncap, labels=list(range(N_PT)), average='macro', zero_division=0)
            margin = max(float(conf[topk].min()) if len(topk) else 1.0, 0.10)
            rec = dict(alpha=a, strat=strat, zones=sorted(zones), margin=margin, f1=float(f1c))
            if best is None or f1c > best['f1']:
                best = rec
    a, margin, zones = best['alpha'], best['margin'], set(best['zones'])
    log(f'  stage3 OOF rule: alpha={a} strat={best["strat"]} margin={margin:.4f} '
        f'(base OOF f1p={base_f1:.4f} -> {best["f1"]:.4f})')

    sub = stage2.sort_values('rally_uid').reset_index(drop=True)
    assert np.array_equal(sub.rally_uid.values, np.sort(tu))
    test_p_s = test_p[np.argsort(tu)]
    bl_t_arg = blend(a, base_t, test_p_s).argmax(1)
    mt = np.sort(test_p_s, 1); margin_t = mt[:, -1] - mt[:, -2]
    base_pt = sub.pointId.values
    flip = ((bl_t_arg != base_pt) & (margin_t >= margin) &
            np.isin(bl_t_arg, list(zones)) & (test_p_s.argmax(1) == bl_t_arg) &
            (sub.actionId.values != 0))
    if flip.sum() > CAP:
        c = margin_t.copy(); c[~flip] = -1
        keep = np.argsort(-c)[:CAP]; nf = np.zeros_like(flip); nf[keep] = True; flip = nf & flip
    new_pt = base_pt.copy(); new_pt[flip] = bl_t_arg[flip]
    log(f'  stage3 TEST flips={int(flip.sum())} p0={float((new_pt==0).mean()):.4f} '
        f'into={pd.Series(new_pt[flip]).value_counts().sort_index().to_dict()}')
    out = sub.copy(); out['pointId'] = new_pt.astype(int)
    return out


def main():
    log('=== SELF-CONTAINED reproduction of final submission (package-only inputs) ===\n')
    base0 = load_sub(BASE0)
    log('STAGE 1 — v701 within-match point overlay')
    s1 = stage1_v701(base0);          assert_match('stage1', s1, STAGE1)
    log('STAGE 2 — v1080 transformer OOV-gated action')
    s2 = stage2_v1080(s1);            assert_match('stage2', s2, STAGE2)
    log('STAGE 3 — v1341 clean drag-zone point overlay (FINAL)')
    s3 = stage3_v1341(s2);            ref = assert_match('FINAL', s3, FINAL)

    # write reproduced FINAL and byte-diff against bundled FINAL
    out_path = ASM / 'reproduced_FINAL.csv'
    s3.sort_values('rally_uid').reset_index(drop=True).to_csv(out_path, index=False)
    b_repro = out_path.read_bytes()
    b_ref = FINAL.read_bytes()
    byte_ok = (b_repro == b_ref)
    log(f'\nByte-identical to {FINAL.name}: {byte_ok} '
        f'({len(b_repro)} vs {len(b_ref)} bytes)')
    if byte_ok:
        out_path.unlink()  # clean up; identical to bundled FINAL

    log('\n' + ('=' * 64))
    log(f'RESULT: {"ALL STAGES REPRODUCE EXACTLY — submission is verifiable." if ok_all else "MISMATCH — investigate above."}')
    log('=' * 64)
    sys.exit(0 if ok_all else 1)


if __name__ == '__main__':
    main()
