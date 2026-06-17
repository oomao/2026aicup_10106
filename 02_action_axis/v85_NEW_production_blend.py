"""v85: NM convex blend search — v38 anchor + v79_post.

v85a: action-only blend (v38 point+server kept pure)
  - v79_post test_point over-predicts p=0 (41.7% vs v38 20.4%) → point blending rejected
  - Only blend action probs; keep v38 point/server for clean distribution

Strategy: search blend weights on STANDARD action f1, validate on mandatory gates:
  1. STANDARD OOF >= v38
  2. test_pmf weighted OOF >= v38
  3. rare_player_AND_pair OOF >= v38  <-- KEY new LB proxy
  4. action disagreement_value >= 0

Outputs:
  outputs/blend_results.md  — all blend candidates + gate status
  outputs/v85_submission.csv — best passing blend (if any)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path("E:/AICUP_O")
sys.path.insert(0, str(ROOT / "models/_archive_old/v76_dashboard"))
sys.path.insert(0, str(ROOT / "models/_archive_old/v84_validation"))

from build_dashboard import _load_bundle, normalize_action
from harness import get_extended_buckets, get_fold_aware_buckets

OUT = ROOT / "models/v85_NEW/outputs"
OUT.mkdir(parents=True, exist_ok=True)

# ── loaders ──────────────────────────────────────────────────────────────────
def load_v38():
    return (normalize_action(np.load(ROOT/"models/_active/v36/outputs/oof_action.npy")),
            np.load(ROOT/"models/_active/v38/outputs/oof_point.npy"),
            np.load(ROOT/"models/_active/v38/outputs/oof_server.npy"),
            np.load(ROOT/"models/_active/v36/outputs/test_action.npy"),
            np.load(ROOT/"models/_active/v38/outputs/test_point.npy"),
            np.load(ROOT/"models/_active/v38/outputs/test_server.npy"))

def load_v79_post():
    P = ROOT/"models/v79_NEW_post/outputs"
    return (normalize_action(np.load(P/"oof_action.npy")),
            np.load(P/"oof_point.npy"),
            np.load(P/"oof_server.npy"),
            np.load(ROOT/"models/v79_NEW/outputs/test_action.npy"),
            np.load(ROOT/"models/v79_NEW/outputs/test_point.npy"),
            np.load(ROOT/"models/v79_NEW/outputs/test_server.npy"))

# ── metric helpers ────────────────────────────────────────────────────────────
def compute_overall(y_a, y_p, y_s, oa, op, os_, sample_w=None):
    pa = oa.argmax(1)
    pp_p = op.copy(); pp_p[pa==0]=0; pp_p[pa==0,0]=1
    pp = pp_p.argmax(1)
    kw = dict(labels=list(range(15)), average='macro', zero_division=0)
    f1a = f1_score(y_a, pa, **kw)
    f1p = f1_score(y_p, pp, labels=list(range(10)), average='macro', zero_division=0)
    try: auc = roc_auc_score(y_s, os_)
    except: auc = 0.5
    return 0.4*f1a + 0.4*f1p + 0.2*auc

def compute_overall_w(y_a, y_p, y_s, oa, op, os_, sample_w):
    pa = oa.argmax(1)
    pp_p = op.copy(); pp_p[pa==0]=0; pp_p[pa==0,0]=1
    pp = pp_p.argmax(1)
    f1a = f1_score(y_a, pa, labels=list(range(15)), average='macro',
                   zero_division=0, sample_weight=sample_w)
    f1p = f1_score(y_p, pp, labels=list(range(10)), average='macro',
                   zero_division=0, sample_weight=sample_w)
    try: auc = roc_auc_score(y_s, os_, sample_weight=sample_w)
    except: auc = 0.5
    return 0.4*f1a + 0.4*f1p + 0.2*auc

def overall_masked(y_a, y_p, y_s, oa, op, os_, mask):
    return compute_overall(y_a[mask], y_p[mask], y_s[mask],
                           oa[mask], op[mask], os_[mask])

# ── NM optimizer ─────────────────────────────────────────────────────────────
def nm_blend(sources_oa, sources_op, sources_os, y_a, y_p, y_s, n_restarts=30):
    n = len(sources_oa)
    rng = np.random.default_rng(42)

    def neg_f1(w_raw):
        w = np.abs(w_raw); w /= w.sum()
        oa = sum(w[i]*sources_oa[i] for i in range(n))
        op = sum(w[i]*sources_op[i] for i in range(n))
        os_ = sum(w[i]*sources_os[i] for i in range(n))
        # optimise standard OOF
        return -compute_overall(y_a, y_p, y_s, oa, op, os_)

    best_w, best_val = None, 1e9
    for _ in range(n_restarts):
        w0 = rng.dirichlet(np.ones(n))
        r = minimize(neg_f1, w0, method='Nelder-Mead',
                     options={'maxiter': 600, 'xatol': 1e-5, 'fatol': 1e-5})
        if r.fun < best_val:
            best_val = r.fun; best_w = np.abs(r.x)/np.abs(r.x).sum()
    return best_w, -best_val

# ── mandatory gate ────────────────────────────────────────────────────────────
def gate_check(y_a, y_p, y_s, oa, op, os_, anc_oa, anc_op, anc_os,
               sample_w, rare_mask):
    results = {}
    std = compute_overall(y_a, y_p, y_s, oa, op, os_)
    anc_std = compute_overall(y_a, y_p, y_s, anc_oa, anc_op, anc_os)
    results['standard_overall'] = (std, anc_std, std >= anc_std - 0.001)

    wt = compute_overall_w(y_a, y_p, y_s, oa, op, os_, sample_w)
    anc_wt = compute_overall_w(y_a, y_p, y_s, anc_oa, anc_op, anc_os, sample_w)
    results['test_pmf_weighted'] = (wt, anc_wt, wt >= anc_wt - 0.001)

    rare = overall_masked(y_a, y_p, y_s, oa, op, os_, rare_mask)
    anc_rare = overall_masked(y_a, y_p, y_s, anc_oa, anc_op, anc_os, rare_mask)
    results['rare_player_AND_pair'] = (rare, anc_rare, rare >= anc_rare)

    # disagreement value (action, test_pmf weighted)
    pa_cand = oa.argmax(1); pa_anc = anc_oa.argmax(1)
    disagree = pa_cand != pa_anc
    if disagree.sum() > 0:
        w = sample_w
        w_dis = w[disagree]
        cand_acc = (((pa_cand == y_a) & disagree).astype(float) * w).sum() / w_dis.sum() if w_dis.sum() > 0 else 0.0
        anc_acc  = (((pa_anc  == y_a) & disagree).astype(float) * w).sum() / w_dis.sum() if w_dis.sum() > 0 else 0.0
        dis_val_a = (cand_acc - anc_acc) * float(w_dis.sum() / w.sum())
    else:
        dis_val_a = 0.0
    results['action_disagree_value'] = (dis_val_a, 0.0, dis_val_a >= 0)

    n_pass = sum(1 for v in results.values() if v[2])
    return results, n_pass

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    bundle = _load_bundle()
    y_a, y_p, y_s = bundle['y_a'], bundle['y_p'], bundle['y_s']
    sample_w = bundle['sample_w']

    all_fam = {**get_extended_buckets(bundle), **get_fold_aware_buckets(bundle)}
    rare_mask = all_fam['unseen_proxy']['rare_player_AND_pair']

    print("[v85] loading models...", flush=True)
    v38_oa, v38_op, v38_os, v38_ta, v38_tp, v38_ts = load_v38()
    v79_oa, v79_op, v79_os, v79_ta, v79_tp, v79_ts = load_v79_post()

    # v79_post test_point over-predicts p=0 (41.7% vs v38 20.4%) => point blending rejected
    # Action-only blend: optimise on action f1, keep v38 point+server pure
    print("[v85] NM blend search ACTION-ONLY (v38 + v79_post)...", flush=True)
    w, val = nm_blend([v38_oa, v79_oa], [v38_op, v38_op], [v38_os, v38_os],
                      y_a, y_p, y_s)
    print(f"  best action weights: v38={w[0]:.3f} v79={w[1]:.3f}  OOF={val:.6f}", flush=True)

    # Build blend: action blended, point+server pure v38
    oa_b = w[0]*v38_oa + w[1]*v79_oa
    op_b = v38_op   # pure v38 point
    os_b = v38_os   # pure v38 server

    # Gate check
    gate, n_pass = gate_check(y_a, y_p, y_s, oa_b, op_b, os_b,
                               v38_oa, v38_op, v38_os, sample_w, rare_mask)
    print(f"[v85] Gate: {n_pass}/4 pass", flush=True)
    for k, (cand, anc, ok) in gate.items():
        print(f"  {'OK' if ok else 'XX'} {k}: cand={cand:.4f} anc={anc:.4f}", flush=True)

    # Also try fixed weights for reference
    results_all = []
    for v38_w in [1.0, 0.9, 0.85, 0.80, 0.75, 0.70, w[0]]:
        v79_w = 1.0 - v38_w
        oa_t = v38_w*v38_oa + v79_w*v79_oa
        op_t = v38_op   # pure v38 point
        os_t = v38_os   # pure v38 server
        std = compute_overall(y_a, y_p, y_s, oa_t, op_t, os_t)
        wt  = compute_overall_w(y_a, y_p, y_s, oa_t, op_t, os_t, sample_w)
        rare= overall_masked(y_a, y_p, y_s, oa_t, op_t, os_t, rare_mask)
        label = f"v38={v38_w:.2f}" if v38_w != w[0] else f"NM(v38={w[0]:.3f})"
        results_all.append(dict(label=label, std=std, test_pmf=wt, rare=rare))
    df_r = pd.DataFrame(results_all)
    print("\n[v85] Blend grid:", flush=True)
    print(df_r.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)

    # Save results md
    lines = ["# v85 Blend Results (v38 + v79_post)\n"]
    lines.append(f"NM optimal: v38={w[0]:.3f} v79_post={w[1]:.3f}  standard_OOF={val:.4f}\n")
    lines.append(f"Gate: {n_pass}/4 pass ({'SUBMIT' if n_pass >= 3 else 'REJECT'})\n")
    for k, (c, a, ok) in gate.items():
        lines.append(f"- {'OK' if ok else 'XX'} {k}: cand={c:.4f} anchor={a:.4f}")
    lines.append("\n## Blend grid (v38 weight sweep)")
    cols = list(df_r.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for _, row in df_r.iterrows():
        lines.append("| " + " | ".join(
            f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c])
            for c in cols) + " |")
    (OUT / "blend_results.md").write_text("\n".join(lines), encoding='utf-8')

    # ALWAYS save OOF + TEST so downstream day22_evaluator can use them regardless of gate
    ta_b = w[0]*normalize_action(v38_ta) + w[1]*normalize_action(v79_ta)
    tp_b = v38_tp.copy()  # pure v38 point
    ts_b = v38_ts.copy()  # pure v38 server
    pa_save = ta_b.argmax(1)
    tp_b[pa_save==0] = 0; tp_b[pa_save==0,0] = 1
    np.save(OUT/"oof_action.npy", oa_b)
    np.save(OUT/"oof_point.npy", op_b)
    np.save(OUT/"oof_server.npy", os_b)
    np.save(OUT/"test_action.npy", ta_b)
    np.save(OUT/"test_point.npy", tp_b)
    np.save(OUT/"test_server.npy", ts_b)
    print(f"[v85] saved OOF + TEST (gate {n_pass}/4)", flush=True)
    # Generate submission only if gate passes (>=3/4)
    if n_pass >= 3:
        _gen_submission(oa_b, op_b, os_b, v38_ta, v38_tp, v38_ts,
                        v79_ta, v79_tp, v79_ts, w, n_pass)
    else:
        print("[v85] Gate < 3/4 — no v85_submission.csv generated. OOF/TEST still saved for downstream.", flush=True)

def _gen_submission(oa_b, op_b, os_b, v38_ta, v38_tp, v38_ts,
                    v79_ta, v79_tp, v79_ts, w, n_pass):
    test_raw = pd.read_csv(ROOT/"data/test.csv")
    rally_uids = sorted(test_raw['rally_uid'].unique())

    ta_b = w[0]*normalize_action(v38_ta) + w[1]*normalize_action(v79_ta)
    tp_b = v38_tp   # pure v38 point
    ts_b = v38_ts   # pure v38 server

    # action=0 -> point=0
    pa = ta_b.argmax(1)
    tp_b[pa==0] = 0; tp_b[pa==0,0] = 1

    sub = pd.DataFrame({
        'rally_uid': rally_uids,
        'actionId': pa.astype(int),
        'pointId': tp_b.argmax(1).astype(int),
        'serverGetPoint': ts_b.astype(float),
    }).sort_values('rally_uid').reset_index(drop=True)

    # sanity
    a0_rate = (sub['actionId']==0).mean()
    p0_rate = (sub['pointId']==0).mean()
    sv_mean = sub['serverGetPoint'].mean()
    print(f"[v85] sub sanity: action=0 {a0_rate:.3f} point=0 {p0_rate:.3f} server {sv_mean:.3f}", flush=True)

    path = OUT/"v85_submission.csv"
    sub.to_csv(path, index=False)
    print(f"[v85] saved {path}  (gate {n_pass}/4 PASS)", flush=True)
    np.save(OUT/"oof_action.npy", oa_b)
    np.save(OUT/"oof_point.npy", op_b)
    np.save(OUT/"oof_server.npy", os_b)
    np.save(OUT/"test_action.npy", ta_b)
    np.save(OUT/"test_point.npy", tp_b)
    np.save(OUT/"test_server.npy", ts_b)

if __name__ == "__main__":
    main()
