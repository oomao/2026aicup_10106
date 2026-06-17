"""v1341 — CLEAN richer within-match transductive POINT GBDT (LGB GPU).

Beat v701 (OOV f1p 0.2728) DEPLOYABLY without the v1340 terminal leak (wm_opponent_serve).

Method:
- v701's 6 within-match blocks + CLEAN richer blocks (wm_player_prevpoint zone-transition,
  wm_player_score, wm_player_ssbucket, wm_opponent plain) + truncation-matched prior-rally
  striker point distribution (transd_mp_point).  wm_opponent_serve DROPPED (the leak).
- Player-grouped StratifiedGroupKFold(5) by striker, stratified by pointId (OOV-honest).
  Also MATCH-held-out for context.
- TWO class_weight variants: BAL (balanced, v701-style) + NOBAL (class_weight=None).
- Multi-seed OOF (3 seeds avg) + 5-seed test refit.  GPU LightGBM (rule O4).

ANTI-LEAK GATES (must hold):
- OOF class-0 (terminal) f1 HEALTHY (~0.40, NOT >0.7 which = leak).
- |OOF p0 - TEST p0| < 0.05 (p0 drift; v1340 had +0.117 = leak).
- rule-96: corr_test - corr_oof < +0.03.

VERDICT: STAGE if a CLEAN variant beats v701 OOV by > +0.003 AND deployable (test p0 in
[0.18,0.27]) AND rule-96 safe AND no terminal leak (OOF class-0 f1 < 0.5).  Else NO_GAIN/LEAK.
"""
import sys, time, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.metrics import f1_score, recall_score
import lightgbm as lgb

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import features_clean as F
# reuse v1340's transd_prior (truncation-matched prior-rally striker point dist) verbatim
sys.path.insert(0, str(Path('E:/AICUP_O/models/v1340_transd_point_v2/src')))
import transd_prior as TP

ROOT = Path('E:/AICUP_O')
OUT = SRC.parent / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)
N_PT = 10
DRAG = [1, 3, 4, 5]
PT_RARE = [1, 2, 3]
SEED = 42
OOF_SEEDS = [42, 7, 13]
REFIT_SEEDS = [42, 7, 13, 99, 123]
TRUNC_SEEDS = [101, 202, 303, 404, 505]
V701_OOV = 0.2728

_LOGF = open(OUT / 'run.log', 'w', buffering=1)
def log(*a):
    print(*a, flush=True)
    print(*a, file=_LOGF, flush=True)


def gpu_ok():
    try:
        X = np.random.rand(64, 4).astype(np.float32); y = np.random.randint(0, 3, 64)
        lgb.LGBMClassifier(objective='multiclass', num_class=3, n_estimators=5,
                           device='gpu', gpu_platform_id=0, gpu_device_id=0,
                           verbose=-1).fit(X, y)
        return True
    except Exception as e:
        log(f'[gpu-probe] GPU failed ({type(e).__name__}: {str(e)[:120]}) -> CPU')
        return False


import os
# Rule O4 = GPU by default. OVERRIDE only when an external GPU job makes shared-GPU LGB
# slower than CPU for this small data (measured 84s GPU vs 17.6s CPU/fit under contention).
FORCE_CPU = os.environ.get('V1341_FORCE_CPU', '0') == '1'
GPU = (not FORCE_CPU) and gpu_ok()
log(f'GPU LightGBM available: {GPU} (FORCE_CPU={FORCE_CPU})')


def lgb_params(class_weight, seed):
    p = dict(objective='multiclass', num_class=N_PT, n_estimators=700, learning_rate=0.03,
             num_leaves=63, max_depth=-1, min_child_samples=80, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0, reg_alpha=1.0,
             class_weight=class_weight, random_state=seed, verbose=-1, max_bin=255)
    if GPU:
        p.update(device='gpu', gpu_platform_id=0, gpu_device_id=0)
    else:
        p.update(device='cpu', n_jobs=-1)
    return p


t0 = time.time()
log('=== v1341 CLEAN richer transductive point | start ===')

train = pd.read_csv(ROOT / 'data/train.csv').sort_values(['rally_uid', 'strikeNumber']).reset_index(drop=True)
test = pd.read_csv(ROOT / 'data/test.csv').sort_values(['rally_uid', 'strikeNumber']).reset_index(drop=True)
if 'serverGetPoint' in test.columns:
    test = test.drop(columns=['serverGetPoint'])

train = F.add_serve_and_seq(train)
test = F.add_serve_and_seq(test)

train_NT = train[(train.strikeNumber >= 2) & (train.actionId < 15)].copy().reset_index(drop=True)
y = train_NT['pointId'].values.astype(int)
assert len(train_NT) == 69710, f'alignment mismatch {len(train_NT)}'
log(f'train_NT rows={len(train_NT)} (canonical n=69710)')

global_prior = np.bincount(y, minlength=N_PT).astype(np.float64)
global_prior = global_prior / global_prior.sum()

base_oof = np.load(ROOT / 'models/v85_NEW/outputs/oof_point.npy').astype(np.float32)
base_test = np.load(ROOT / 'models/v85_NEW/outputs/test_point.npy').astype(np.float32)
v701_oof = np.load(ROOT / 'models/v701_withinmatch_point/outputs/oof_point.npy').astype(np.float32)
v701_test = np.load(ROOT / 'models/v701_withinmatch_point/outputs/test_point.npy').astype(np.float32)
v701_uids = np.load(ROOT / 'models/v701_withinmatch_point/outputs/test_rally_uids.npy')

base_f1 = f1_score(y, base_oof.argmax(1), labels=list(range(N_PT)), average='macro', zero_division=0)
v701_f1 = f1_score(y, v701_oof.argmax(1), labels=list(range(N_PT)), average='macro', zero_division=0)
log(f'BASELINE v85_NEW point macroF1={base_f1:.4f} | v701 OOF macroF1(player-held-out)={v701_f1:.4f}')

train_NT['tgt_player'] = train_NT['gamePlayerId'].astype(int)
train_NT['tgt_opp'] = train_NT['gamePlayerOtherId'].astype(int)

log('Building TRAIN within-match conditionals (leave-self-out, CLEAN blocks)...')
stats_train = train_NT.assign(gamePlayerId=train_NT['tgt_player'])
wm_train = F.build_within_match_conditionals(stats_train, train_NT, global_prior, self_subtract=True)

test_raw = pd.read_csv(ROOT / 'data/test.csv')
kvis_test = test_raw.groupby('rally_uid')['strikeNumber'].max()
test_kvis_pmf = kvis_test.value_counts(normalize=True).sort_index()
log(f'test K_vis PMF mean={kvis_test.mean():.3f}; building TRAIN transductive prior block...')
prop_tr, tp_cols = TP.build_point_prior_style(ROOT / 'data/train.csv', truncate=True,
                                              test_kvis_pmf=test_kvis_pmf, seeds=TRUNC_SEEDS, log=log)
train_NT['rally_id'] = train_NT['rally_id'].astype(int)
transd_train = TP.attach(train_NT, prop_tr, tp_cols)

train_masked = train_NT.copy()
for c in ['spinId', 'strengthId', 'handId', 'positionId']:
    train_masked[c] = -1
X_train, feat_names = F.assemble_numeric(train_masked, wm_train, baseline_point=base_oof,
                                         transd_block=transd_train, transd_cols=tp_cols)
log(f'TRAIN feature matrix: {X_train.shape}, {len(feat_names)} features')
# anti-leak: ensure NO wm_opponent_serve feature exists
assert not any(n.startswith('wm_opponent_serve') for n in feat_names), 'LEAK FEATURE PRESENT!'

groups_player = train_NT['gamePlayerId'].astype(int).values
groups_match = train_NT['match'].astype(int).values

# ============================================================================
# TEST target frame
# ============================================================================
test = pd.read_csv(ROOT / 'data/test.csv').sort_values(['rally_uid', 'strikeNumber']).reset_index(drop=True)
if 'serverGetPoint' in test.columns:
    test = test.drop(columns=['serverGetPoint'])
test = F.add_serve_and_seq(test)
test_NT_pool = test[(test.strikeNumber >= 2) & (test.actionId < 15)].copy()
last = test.sort_values('strikeNumber').groupby('rally_uid', sort=True).tail(1).copy()
last = last.sort_values('rally_uid').reset_index(drop=True)
last['tgt_player'] = last['gamePlayerOtherId'].astype(int)
last['tgt_opp'] = last['gamePlayerId'].astype(int)


def build_test_target_frame(test_df, last_rows):
    grp = {rid: sub.sort_values('strikeNumber') for rid, sub in test_df.groupby('rally_uid', sort=False)}
    rows = []
    for r in last_rows.itertuples(index=False):
        rid = r.rally_uid
        vis = grp[rid].reset_index(drop=True)
        n = len(vis)
        ss = int(vis.strikeNumber.max())
        smax = max(int(r.scoreSelf), int(r.scoreOther))
        sphase = 3 if smax >= 10 else (2 if smax >= 7 else (1 if smax >= 3 else 0))
        rec = {
            'rally_uid': rid, 'rally_id': int(r.rally_id), 'match': int(r.match),
            'sex': int(r.sex), 'numberGame': int(r.numberGame),
            'scoreSelf': int(r.scoreSelf), 'scoreOther': int(r.scoreOther),
            'gamePlayerId': int(r.tgt_player), 'tgt_player': int(r.tgt_player),
            'tgt_opp': int(r.tgt_opp),
            'serve_spin': int(r.serve_spin), 'serve_point': int(r.serve_point),
            'serve_pos': int(r.serve_pos),
            'since_serve': ss, 'since_serve_parity': ss % 2,
            'ss_bucket': min(ss, 4), 'score_phase': sphase,
            'score_diff_sign': int(np.sign(int(r.scoreSelf) - int(r.scoreOther))),
            'spinId': -1, 'strengthId': -1, 'handId': -1, 'positionId': -1,
        }
        for k in [1, 2, 3]:
            idx = n - k
            if idx >= 0:
                for f_ in ['pointId', 'actionId', 'spinId', 'positionId', 'handId', 'strengthId']:
                    rec[f'prev{k}_{f_}'] = int(vis.loc[idx, f_])
            else:
                for f_ in ['pointId', 'actionId', 'spinId', 'positionId', 'handId', 'strengthId']:
                    rec[f'prev{k}_{f_}'] = -1
        rows.append(rec)
    return pd.DataFrame(rows)


test_tf = build_test_target_frame(test, last)
log('Building TEST within-match conditionals (from test own match, CLEAN blocks)...')
wm_test = F.build_within_match_conditionals(test_NT_pool, test_tf, global_prior, self_subtract=False)
prop_tr_test, _ = TP.build_point_prior_style(ROOT / 'data/test.csv', truncate=True,
                                             test_kvis_pmf=test_kvis_pmf, seeds=TRUNC_SEEDS, log=log)
transd_test = TP.attach(test_tf, prop_tr_test, tp_cols)
X_test, feat_names_t = F.assemble_numeric(test_tf, wm_test, baseline_point=base_test,
                                          transd_block=transd_test, transd_cols=tp_cols)
assert feat_names == feat_names_t, 'feature name mismatch'
test_uids = test_tf['rally_uid'].values.astype(np.int64)
test_cov = float((wm_test['wm_player_n'] > 0).mean())
transd_cov = float((transd_test[:, -1] > 0).mean())
log(f'TEST feature matrix: {X_test.shape} | WM coverage={test_cov:.3f} | transd coverage={transd_cov:.3f}')

order = {u: i for i, u in enumerate(v701_uids)}
v701_test_aligned = v701_test[[order[u] for u in test_uids]]


def run_variant(class_weight, tag):
    log(f'\n========== VARIANT {tag} (class_weight={class_weight}) ==========')
    res = {}
    for scheme, grp in [('PLAYER', groups_player), ('MATCH', groups_match)]:
        if scheme == 'PLAYER':
            sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
            folds = list(sgkf.split(X_train, y, groups=grp))
            for fi, (tr, va) in enumerate(folds):
                assert len(set(grp[tr]) & set(grp[va])) == 0, f'fold {fi} not disjoint'
        else:
            folds = list(GroupKFold(n_splits=5).split(X_train, y, groups=grp))
        oof = np.zeros((len(train_NT), N_PT), np.float32)
        for fi, (tr, va) in enumerate(folds):
            acc = np.zeros((len(va), N_PT), np.float32)
            for sd in OOF_SEEDS:
                clf = lgb.LGBMClassifier(**lgb_params(class_weight, sd))
                clf.fit(X_train[tr], y[tr])
                acc += clf.predict_proba(X_train[va]).astype(np.float32)
            oof[va] = acc / len(OOF_SEEDS)
            log(f'  [{scheme}] fold {fi+1}/5 done')
        arg = oof.argmax(1)
        f1 = f1_score(y, arg, labels=list(range(N_PT)), average='macro', zero_division=0)
        pc = f1_score(y, arg, labels=list(range(N_PT)), average=None, zero_division=0)
        rec0 = recall_score(y, arg, labels=[0], average='macro', zero_division=0)
        rare = float(np.mean([pc[c] for c in PT_RARE]))
        oof_p0 = float((arg == 0).mean())
        log(f'  [{scheme}] macroF1p={f1:.4f} rare={rare:.4f} class0_f1={pc[0]:.4f} '
            f'class0_recall={rec0:.4f} oof_p0={oof_p0:.4f}')
        log(f'  [{scheme}] per-class f1p: ' + ' '.join(f'c{c}={pc[c]:.3f}' for c in range(N_PT)))
        res[scheme] = dict(oof=oof, f1=float(f1), pc=[float(x) for x in pc],
                           rare=rare, class0_f1=float(pc[0]), class0_recall=float(rec0),
                           oof_p0=oof_p0)
    test_probs = np.zeros((len(X_test), N_PT), np.float32)
    for sd in REFIT_SEEDS:
        clf = lgb.LGBMClassifier(**lgb_params(class_weight, sd))
        clf.fit(X_train, y)
        test_probs += clf.predict_proba(X_test).astype(np.float32)
    test_probs /= len(REFIT_SEEDS)
    test_p0 = float((test_probs.argmax(1) == 0).mean())
    log(f'  [TEST] raw point0_rate={test_p0:.4f} | argmax dist '
        f'{np.bincount(test_probs.argmax(1), minlength=N_PT)}')
    res['test_probs'] = test_probs
    res['test_p0'] = test_p0
    return res


variants = {'BAL': run_variant('balanced', 'BAL'),
            'NOBAL': run_variant(None, 'NOBAL')}


def corr_with_v701(oof_arr, test_arr):
    co = np.corrcoef(oof_arr.ravel(), v701_oof.ravel())[0, 1]
    ct = np.corrcoef(test_arr.ravel(), v701_test_aligned.ravel())[0, 1]
    return float(co), float(ct)


def softvote_with_v701(oof_arr, test_arr):
    """Best soft-vote w with v701; report OVERALL and OOV-segment f1p (player-held-out OOF)."""
    best = dict(w=0.0, f1=v701_f1, p0=None)
    for w in np.linspace(0.0, 1.0, 21):
        bl = (1 - w) * v701_oof + w * oof_arr
        f1 = f1_score(y, bl.argmax(1), labels=list(range(N_PT)), average='macro', zero_division=0)
        bt = (1 - w) * v701_test_aligned + w * test_arr
        p0 = float((bt.argmax(1) == 0).mean())
        if f1 > best['f1']:
            best = dict(w=float(w), f1=float(f1), p0=p0)
    return best


summary = dict(model='v1341_transd_point_clean', gpu=GPU, n_train=int(len(train_NT)),
               n_test=int(len(test_uids)), n_features=len(feat_names),
               dropped_leak_block='wm_opponent_serve',
               ruler='StratifiedGroupKFold(5) by striker, stratified by pointId',
               oof_seeds=OOF_SEEDS, refit_seeds=REFIT_SEEDS, trunc_seeds=TRUNC_SEEDS,
               baseline_v85_macroF1=float(base_f1), v701_oof_player_macroF1=float(v701_f1),
               v701_target_oov_f1p=V701_OOV, test_wm_coverage=test_cov,
               test_transd_coverage=transd_cov, variants={})

for tag, r in variants.items():
    co, ct = corr_with_v701(r['PLAYER']['oof'], r['test_probs'])
    sv = softvote_with_v701(r['PLAYER']['oof'], r['test_probs'])
    p0_drift = r['test_p0'] - r['PLAYER']['oof_p0']
    terminal_leak = r['PLAYER']['class0_f1'] >= 0.5
    rule96_safe = (ct - co) < 0.03
    deployable = 0.18 <= r['test_p0'] <= 0.27
    beats = r['PLAYER']['f1'] > V701_OOV + 0.003
    log(f'\n[{tag}] OOV f1p={r["PLAYER"]["f1"]:.4f} (vs v701 {V701_OOV}, beats+0.003={beats}) | '
        f'MATCH f1p={r["MATCH"]["f1"]:.4f}')
    log(f'[{tag}] raw test p0={r["test_p0"]:.4f} (deployable {deployable}) | OOF p0={r["PLAYER"]["oof_p0"]:.4f} '
        f'| p0_drift={p0_drift:+.4f} (leak if >0.05)')
    log(f'[{tag}] class0_f1={r["PLAYER"]["class0_f1"]:.4f} (terminal_leak={terminal_leak}) | '
        f'class0_recall={r["PLAYER"]["class0_recall"]:.4f}')
    log(f'[{tag}] rule96 corr_oof={co:.4f} corr_test={ct:.4f} diff={ct-co:+.4f} (safe={rule96_safe})')
    log(f'[{tag}] softvote w/ v701: best w={sv["w"]:.2f} f1p={sv["f1"]:.4f} '
        f'(Δ vs v701 {sv["f1"]-v701_f1:+.4f}) p0={sv["p0"]}')
    summary['variants'][tag] = dict(
        oov_f1p=r['PLAYER']['f1'], match_f1p=r['MATCH']['f1'],
        per_class_f1p_PLAYER=r['PLAYER']['pc'], rare_f1p=r['PLAYER']['rare'],
        class0_f1=r['PLAYER']['class0_f1'], class0_recall=r['PLAYER']['class0_recall'],
        oof_p0=r['PLAYER']['oof_p0'], test_p0=r['test_p0'], p0_drift=float(p0_drift),
        corr_oof=co, corr_test=ct, rule96_diff=float(ct - co), rule96_safe=bool(rule96_safe),
        terminal_leak=bool(terminal_leak), deployable_p0=bool(deployable),
        beats_v701_standalone=bool(beats), softvote_v701=sv,
        clean_deployable=bool(beats and deployable and rule96_safe and not terminal_leak
                              and abs(p0_drift) < 0.05),
    )

dep_tag = 'NOBAL' if summary['variants']['NOBAL']['clean_deployable'] else \
          ('BAL' if summary['variants']['BAL']['clean_deployable'] else
           ('NOBAL' if summary['variants']['NOBAL']['deployable_p0'] else 'BAL'))
np.save(OUT / 'oof_point.npy', variants[dep_tag]['PLAYER']['oof'])
np.save(OUT / 'oof_point_match.npy', variants[dep_tag]['MATCH']['oof'])
np.save(OUT / 'test_point.npy', variants[dep_tag]['test_probs'])
np.save(OUT / 'oof_point_BAL.npy', variants['BAL']['PLAYER']['oof'])
np.save(OUT / 'test_point_BAL.npy', variants['BAL']['test_probs'])
np.save(OUT / 'oof_point_NOBAL.npy', variants['NOBAL']['PLAYER']['oof'])
np.save(OUT / 'test_point_NOBAL.npy', variants['NOBAL']['test_probs'])
np.save(OUT / 'test_rally_uids.npy', test_uids)
np.save(OUT / 'y_point.npy', y)
summary['deployment_variant'] = dep_tag

any_clean = any(v['clean_deployable'] for v in summary['variants'].values())
any_sv = any(v['softvote_v701']['f1'] > v701_f1 + 1e-4 for v in summary['variants'].values())
if any_clean:
    verdict = 'STAGE'
elif any_sv:
    verdict = 'PARTIAL_SOFTVOTE'
else:
    verdict = 'NO_GAIN'
summary['verdict'] = verdict

with open(OUT / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
log(f'\n=== VERDICT: {verdict} === (deployment variant={dep_tag}) elapsed {time.time()-t0:.0f}s')
log('=== build done ===')
