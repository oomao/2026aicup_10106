"""v1341 LEAN — CLEAN richer within-match transductive POINT GBDT (LGB GPU).

Faster verdict variant of build.py (GPU is shared with another job, so we cut cost):
- PLAYER-held-out OOF ONLY (the OOV-honest metric; MATCH scheme dropped — was context-only).
- 400 trees / lr 0.05 (diag-validated as representative of the 700/0.03 config).
- 3 OOF seeds + 3 refit seeds.
- Identical CLEAN features (wm_opponent_serve dropped = the v1340 leak) + same anti-leak gates
  + overlay-compatible outputs.
"""
import sys, time, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, recall_score
import lightgbm as lgb

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import features_clean as F
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
REFIT_SEEDS = [42, 7, 13]
TRUNC_SEEDS = [101, 202, 303, 404, 505]
V701_OOV = 0.2728

_LOGF = open(OUT / 'run_lean.log', 'w', buffering=1)
def log(*a):
    print(*a, flush=True)
    print(*a, file=_LOGF, flush=True)


def gpu_ok():
    try:
        X = np.random.rand(64, 4).astype(np.float32); y = np.random.randint(0, 3, 64)
        lgb.LGBMClassifier(objective='multiclass', num_class=3, n_estimators=5,
                           device='gpu', gpu_platform_id=0, gpu_device_id=0, verbose=-1).fit(X, y)
        return True
    except Exception as e:
        log(f'[gpu-probe] GPU failed ({type(e).__name__}) -> CPU'); return False


GPU = gpu_ok()
log(f'GPU LightGBM available: {GPU}')


def lgb_params(class_weight, seed):
    p = dict(objective='multiclass', num_class=N_PT, n_estimators=400, learning_rate=0.05,
             num_leaves=63, max_depth=-1, min_child_samples=80, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0, reg_alpha=1.0,
             class_weight=class_weight, random_state=seed, verbose=-1, max_bin=255)
    if GPU:
        p.update(device='gpu', gpu_platform_id=0, gpu_device_id=0)
    else:
        p.update(device='cpu', n_jobs=-1)
    return p


t0 = time.time()
log('=== v1341 LEAN clean richer transductive point | start ===')
train = pd.read_csv(ROOT / 'data/train.csv').sort_values(['rally_uid', 'strikeNumber']).reset_index(drop=True)
test = pd.read_csv(ROOT / 'data/test.csv').sort_values(['rally_uid', 'strikeNumber']).reset_index(drop=True)
if 'serverGetPoint' in test.columns:
    test = test.drop(columns=['serverGetPoint'])
train = F.add_serve_and_seq(train)
test = F.add_serve_and_seq(test)
train_NT = train[(train.strikeNumber >= 2) & (train.actionId < 15)].copy().reset_index(drop=True)
y = train_NT['pointId'].values.astype(int)
assert len(train_NT) == 69710
log(f'train_NT rows={len(train_NT)}')
global_prior = np.bincount(y, minlength=N_PT).astype(np.float64); global_prior /= global_prior.sum()
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
log('Building TRAIN within-match conditionals (CLEAN)...')
stats_train = train_NT.assign(gamePlayerId=train_NT['tgt_player'])
wm_train = F.build_within_match_conditionals(stats_train, train_NT, global_prior, self_subtract=True)
test_raw = pd.read_csv(ROOT / 'data/test.csv')
kvis_test = test_raw.groupby('rally_uid')['strikeNumber'].max()
test_kvis_pmf = kvis_test.value_counts(normalize=True).sort_index()
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
assert not any(n.startswith('wm_opponent_serve') for n in feat_names), 'LEAK FEATURE PRESENT!'
groups_player = train_NT['gamePlayerId'].astype(int).values

# ---- TEST frame ----
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
        n = len(vis); ss = int(vis.strikeNumber.max())
        smax = max(int(r.scoreSelf), int(r.scoreOther))
        sphase = 3 if smax >= 10 else (2 if smax >= 7 else (1 if smax >= 3 else 0))
        rec = {'rally_uid': rid, 'rally_id': int(r.rally_id), 'match': int(r.match),
               'sex': int(r.sex), 'numberGame': int(r.numberGame),
               'scoreSelf': int(r.scoreSelf), 'scoreOther': int(r.scoreOther),
               'gamePlayerId': int(r.tgt_player), 'tgt_player': int(r.tgt_player), 'tgt_opp': int(r.tgt_opp),
               'serve_spin': int(r.serve_spin), 'serve_point': int(r.serve_point), 'serve_pos': int(r.serve_pos),
               'since_serve': ss, 'since_serve_parity': ss % 2, 'ss_bucket': min(ss, 4), 'score_phase': sphase,
               'score_diff_sign': int(np.sign(int(r.scoreSelf) - int(r.scoreOther))),
               'spinId': -1, 'strengthId': -1, 'handId': -1, 'positionId': -1}
        for k in [1, 2, 3]:
            idx = n - k
            for f_ in ['pointId', 'actionId', 'spinId', 'positionId', 'handId', 'strengthId']:
                rec[f'prev{k}_{f_}'] = int(vis.loc[idx, f_]) if idx >= 0 else -1
        rows.append(rec)
    return pd.DataFrame(rows)


test_tf = build_test_target_frame(test, last)
log('Building TEST within-match conditionals (CLEAN)...')
wm_test = F.build_within_match_conditionals(test_NT_pool, test_tf, global_prior, self_subtract=False)
prop_tr_test, _ = TP.build_point_prior_style(ROOT / 'data/test.csv', truncate=True,
                                             test_kvis_pmf=test_kvis_pmf, seeds=TRUNC_SEEDS, log=log)
transd_test = TP.attach(test_tf, prop_tr_test, tp_cols)
X_test, feat_names_t = F.assemble_numeric(test_tf, wm_test, baseline_point=base_test,
                                          transd_block=transd_test, transd_cols=tp_cols)
assert feat_names == feat_names_t
test_uids = test_tf['rally_uid'].values.astype(np.int64)
test_cov = float((wm_test['wm_player_n'] > 0).mean())
transd_cov = float((transd_test[:, -1] > 0).mean())
log(f'TEST feature matrix: {X_test.shape} | WM cov={test_cov:.3f} | transd cov={transd_cov:.3f}')
order = {u: i for i, u in enumerate(v701_uids)}
v701_test_aligned = v701_test[[order[u] for u in test_uids]]


def run_variant(class_weight, tag):
    log(f'\n========== VARIANT {tag} (class_weight={class_weight}) ==========')
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(sgkf.split(X_train, y, groups=groups_player))
    for fi, (tr, va) in enumerate(folds):
        assert len(set(groups_player[tr]) & set(groups_player[va])) == 0
    oof = np.zeros((len(train_NT), N_PT), np.float32)
    for fi, (tr, va) in enumerate(folds):
        acc = np.zeros((len(va), N_PT), np.float32)
        for sd in OOF_SEEDS:
            clf = lgb.LGBMClassifier(**lgb_params(class_weight, sd))
            clf.fit(X_train[tr], y[tr])
            acc += clf.predict_proba(X_train[va]).astype(np.float32)
        oof[va] = acc / len(OOF_SEEDS)
        log(f'  [PLAYER] fold {fi+1}/5 done  ({time.time()-t0:.0f}s)')
    arg = oof.argmax(1)
    f1 = f1_score(y, arg, labels=list(range(N_PT)), average='macro', zero_division=0)
    pc = f1_score(y, arg, labels=list(range(N_PT)), average=None, zero_division=0)
    rec0 = recall_score(y, arg, labels=[0], average='macro', zero_division=0)
    rare = float(np.mean([pc[c] for c in PT_RARE]))
    oof_p0 = float((arg == 0).mean())
    log(f'  [PLAYER] macroF1p={f1:.4f} rare={rare:.4f} class0_f1={pc[0]:.4f} class0_recall={rec0:.4f} oof_p0={oof_p0:.4f}')
    log(f'  [PLAYER] per-class f1p: ' + ' '.join(f'c{c}={pc[c]:.3f}' for c in range(N_PT)))
    test_probs = np.zeros((len(X_test), N_PT), np.float32)
    for sd in REFIT_SEEDS:
        clf = lgb.LGBMClassifier(**lgb_params(class_weight, sd))
        clf.fit(X_train, y)
        test_probs += clf.predict_proba(X_test).astype(np.float32)
    test_probs /= len(REFIT_SEEDS)
    test_p0 = float((test_probs.argmax(1) == 0).mean())
    log(f'  [TEST] raw point0_rate={test_p0:.4f} | argmax dist {np.bincount(test_probs.argmax(1), minlength=N_PT)}')
    return dict(oof=oof, f1=float(f1), pc=[float(x) for x in pc], rare=rare,
                class0_f1=float(pc[0]), class0_recall=float(rec0), oof_p0=oof_p0,
                test_probs=test_probs, test_p0=test_p0)


variants = {'BAL': run_variant('balanced', 'BAL'), 'NOBAL': run_variant(None, 'NOBAL')}


def softvote_with_v701(oof_arr, test_arr):
    best = dict(w=0.0, f1=v701_f1, p0=None)
    for w in np.linspace(0.0, 1.0, 21):
        bl = (1 - w) * v701_oof + w * oof_arr
        f1 = f1_score(y, bl.argmax(1), labels=list(range(N_PT)), average='macro', zero_division=0)
        bt = (1 - w) * v701_test_aligned + w * test_arr
        if f1 > best['f1']:
            best = dict(w=float(w), f1=float(f1), p0=float((bt.argmax(1) == 0).mean()))
    return best


summary = dict(model='v1341_transd_point_clean_LEAN', gpu=GPU, n_train=int(len(train_NT)),
               n_test=int(len(test_uids)), n_features=len(feat_names),
               dropped_leak_block='wm_opponent_serve',
               oof_seeds=OOF_SEEDS, refit_seeds=REFIT_SEEDS,
               baseline_v85_macroF1=float(base_f1), v701_oof_player_macroF1=float(v701_f1),
               v701_target_oov_f1p=V701_OOV, test_wm_coverage=test_cov,
               test_transd_coverage=transd_cov, variants={})
for tag, r in variants.items():
    co = float(np.corrcoef(r['oof'].ravel(), v701_oof.ravel())[0, 1])
    ct = float(np.corrcoef(r['test_probs'].ravel(), v701_test_aligned.ravel())[0, 1])
    sv = softvote_with_v701(r['oof'], r['test_probs'])
    p0_drift = r['test_p0'] - r['oof_p0']
    terminal_leak = r['class0_f1'] >= 0.5
    rule96_safe = (ct - co) < 0.03
    deployable = 0.18 <= r['test_p0'] <= 0.27
    beats = r['f1'] > V701_OOV + 0.003
    clean_dep = bool(beats and deployable and rule96_safe and not terminal_leak and abs(p0_drift) < 0.05)
    log(f'\n[{tag}] OOV f1p={r["f1"]:.4f} (vs v701 {V701_OOV}, beats+0.003={beats})')
    log(f'[{tag}] raw test p0={r["test_p0"]:.4f} (deployable {deployable}) | OOF p0={r["oof_p0"]:.4f} | p0_drift={p0_drift:+.4f}')
    log(f'[{tag}] class0_f1={r["class0_f1"]:.4f} (terminal_leak={terminal_leak}) | class0_recall={r["class0_recall"]:.4f}')
    log(f'[{tag}] rule96 corr_oof={co:.4f} corr_test={ct:.4f} diff={ct-co:+.4f} (safe={rule96_safe})')
    log(f'[{tag}] softvote w/ v701: best w={sv["w"]:.2f} f1p={sv["f1"]:.4f} (Δ vs v701 {sv["f1"]-v701_f1:+.4f}) p0={sv["p0"]}')
    log(f'[{tag}] CLEAN_DEPLOYABLE={clean_dep}')
    summary['variants'][tag] = dict(
        oov_f1p=r['f1'], per_class_f1p=r['pc'], rare_f1p=r['rare'],
        class0_f1=r['class0_f1'], class0_recall=r['class0_recall'],
        oof_p0=r['oof_p0'], test_p0=r['test_p0'], p0_drift=float(p0_drift),
        corr_oof=co, corr_test=ct, rule96_diff=float(ct - co), rule96_safe=bool(rule96_safe),
        terminal_leak=bool(terminal_leak), deployable_p0=bool(deployable),
        beats_v701=bool(beats), softvote_v701=sv, clean_deployable=clean_dep)

dep_tag = ('NOBAL' if summary['variants']['NOBAL']['clean_deployable'] else
           ('BAL' if summary['variants']['BAL']['clean_deployable'] else
            ('NOBAL' if summary['variants']['NOBAL']['deployable_p0'] else 'BAL')))
for tg in ('BAL', 'NOBAL'):
    np.save(OUT / f'oof_point_{tg}.npy', variants[tg]['oof'])
    np.save(OUT / f'test_point_{tg}.npy', variants[tg]['test_probs'])
np.save(OUT / 'oof_point.npy', variants[dep_tag]['oof'])
np.save(OUT / 'test_point.npy', variants[dep_tag]['test_probs'])
np.save(OUT / 'test_rally_uids.npy', test_uids)
np.save(OUT / 'y_point.npy', y)
summary['deployment_variant'] = dep_tag
any_clean = any(v['clean_deployable'] for v in summary['variants'].values())
any_sv = any(v['softvote_v701']['f1'] > v701_f1 + 1e-4 for v in summary['variants'].values())
summary['verdict'] = 'STAGE' if any_clean else ('PARTIAL_SOFTVOTE' if any_sv else 'NO_GAIN')
json.dump(summary, open(OUT / 'summary_lean.json', 'w'), indent=2)
log(f'\n=== VERDICT: {summary["verdict"]} === (dep variant={dep_tag}) elapsed {time.time()-t0:.0f}s')
log('=== build done ===')
