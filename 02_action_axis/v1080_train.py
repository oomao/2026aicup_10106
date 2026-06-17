"""v1080 — Transformer + truncation-matched transductive context, player-agnostic,
player-grouped CV. The whole-chain (action+point+server) build toward LB 0.46.

Three changes vs v1054 finetune_player:
  1. TRANSDUCTIVE context (16-dim, truncation-matched, multi-seed) REPLACES player-id
     embeddings. Plumbed via a parallel projection concatenated into pooled.
  2. Player-grouped StratifiedGroupKFold(5) by TARGET STRIKER, stratified by actionId,
     as the ONLY ruler. Folds asserted player-disjoint. Inner early-stop split is ALSO
     player-disjoint (carved from train players) -> no seen-player model-selection mirage.
  3. Whole-chain multi-task: action(15)+point(10)+server(1), class-weighted CE + label
     smoothing 0.05, truncation augmentation (test K-vis PMF).

Produces player-grouped OOF (aligned to v1075 target_row_order.csv) AND 1845-row test
predictions (refit on ALL train, sorted by rally_uid). Encoder pretrained (loaded, not
re-pretrained). GPU (rule O4).

Anti-leak: encoder sees only strokes 1..K; transductive uses only the striker's PRIOR
in-match rallies (truncation-matched, proportions); serverGetPoint never input.
"""
from __future__ import annotations
import sys, time, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = Path("E:/AICUP_O")
V1054_SRC = ROOT / "models/v1054_serious_transformer/src"
sys.path.insert(0, str(V1054_SRC))

import data as D                       # v1054 data (build_sequences, prefix batch, etc.)
import transductive as T
from model_v1080 import TransductiveMultiTask, N_TRANSDUCTIVE, TransformerEncoder

OUT = HERE.parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
V1054_OUT = ROOT / "models/v1054_serious_transformer/outputs"
V1075_OUT = ROOT / "models/v1075_deconfound_transductive/outputs"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
N_SPLITS = 5
EPOCHS = 40
BATCH = 256
LR = 1.0e-3
WD = 0.05
PAT = 8
W_SERVER = 0.3
LABEL_SMOOTH = 0.05
RARE_A = [5, 7, 8, 9, 14]
RARE_P = [1, 2, 3]
TEST_K_PMF_AUG = {1: 0.2753, 2: 0.2575, 3: 0.1669, 4: 0.1014, 5: 0.0743, 6: 0.0412, 7: 0.0835}

_LOG = open(OUT / "run.log", "a", buffering=1)


def log(*a):
    print(*a, flush=True)
    print(*a, file=_LOG, flush=True)


def f1a(y, p):
    return f1_score(y, p, labels=list(range(15)), average="macro", zero_division=0)


def f1p(y, p):
    return f1_score(y, p, labels=list(range(10)), average="macro", zero_division=0)


def trunc_augment(fseq_b, nseq_b, len_b, rng, p_aug=0.5):
    B, Lm, nf = fseq_b.shape
    fb = fseq_b.clone(); nb = nseq_b.clone(); lb = len_b.clone()
    keys = np.array(list(TEST_K_PMF_AUG.keys()))
    probs = np.array([TEST_K_PMF_AUG[k] for k in keys]); probs = probs / probs.sum()
    lb_np = len_b.cpu().numpy()
    for b in range(B):
        L = int(lb_np[b])
        if L <= 1 or rng.random() > p_aug:
            continue
        kk = min(int(rng.choice(keys, p=probs)), L)
        if kk == L:
            continue
        cut = Lm - kk
        fb[b, :cut] = 0; nb[b, :cut] = 0.0; lb[b] = kk
    return fb.to(DEV), nb.to(DEV), lb.to(DEV)


def build_train_pack():
    """Canonical train target rows + prefix batch + ctx + transductive (aligned to
    v1075 target_row_order.csv = bit-exact with canonical_target_rows)."""
    packs = D.build_sequences()
    tr = packs["train"]
    target_idx = D.canonical_target_rows(tr)
    ai = D.FIELD_LIST.index("actionId"); pi = D.FIELD_LIST.index("pointId")
    y_action = (tr["field"][:, ai] - D.OFFSET)[target_idx].astype(np.int64)
    y_point = (tr["field"][:, pi] - D.OFFSET)[target_idx].astype(np.int64)
    sgp = tr["df"]["serverGetPoint"].to_numpy().astype(np.float32)[target_idx]
    striker = tr["striker"][target_idx]
    ridx = D.build_rally_index(tr)
    fseq, nseq, lengths = D.make_prefix_batch(tr, ridx, target_idx, max_len=D.MAX_LEN)
    ctx = D.target_context(tr, target_idx)

    # transductive (reuse v1075 builder), aligned to canonical order
    order_df = pd.read_csv(V1075_OUT / "target_row_order.csv")
    # bit-exact alignment sanity
    assert len(order_df) == len(target_idx)
    assert bool((order_df["rally_uid"].values == tr["rally"][target_idx]).all())
    assert bool((order_df["strikeNumber"].values == tr["strike"][target_idx]).all())
    test = pd.read_csv(ROOT / "data/test.csv")
    kvis = test.groupby("rally_uid")["strikeNumber"].max()
    pmf = kvis.value_counts(normalize=True).sort_index()
    transd, cov = T.build_train_transductive(order_df, pmf)
    log(f"[train] n={len(target_idx)} transductive mp_has frac={cov:.3f}")

    # cross-check y_action against v1075 saved labels
    y75 = np.load(V1075_OUT / "y_action.npy")
    assert bool((y75 == y_action).all()), "y_action mismatch vs v1075!"
    return dict(target_idx=target_idx, y_action=y_action, y_point=y_point, sgp=sgp,
                striker=striker, fseq=fseq, nseq=nseq, lengths=lengths, ctx=ctx,
                transd=transd)


def build_test_pack():
    """1845 test rallies: prefix batch (from infer_test logic) + ctx + transductive."""
    import infer_test as IT  # v1054 infer_test (test_target_rows)
    packs = D.build_sequences()
    te = packs["test"]
    uids, t_fseq, t_nseq, t_len, t_ctx, _, _ = IT.test_target_rows(te)
    # transductive for test (prior rallies from test.csv), sorted-by-rally_uid order
    uids_t, transd_te, cov_te = T.build_test_transductive()
    # align transd to the infer order (uids may be unsorted here; sort both)
    order = np.argsort(uids)
    uids_s = uids[order]
    t_fseq = t_fseq[order]; t_nseq = t_nseq[order]; t_len = t_len[order]; t_ctx = t_ctx[order]
    assert bool((uids_s == uids_t).all()), "test rally_uid order mismatch (infer vs transductive)"
    log(f"[test] n={len(uids_s)} transductive mp_has frac={cov_te:.3f}")
    return uids_s, t_fseq, t_nseq, t_len, t_ctx, transd_te


def make_net():
    ckpt = torch.load(V1054_OUT / "pretrained_encoder.pt", map_location="cpu")
    cfg = ckpt["cfg"]
    enc = TransformerEncoder(cfg["d_model"], cfg["n_layers"], cfg["n_heads"],
                             dropout=cfg["dropout"], emb_drop=cfg["emb_drop"]).to(DEV)
    enc.load_state_dict(ckpt["state"])
    net = TransductiveMultiTask(enc, use_server=True).to(DEV)
    return net, cfg


def class_weights(y, n):
    c = np.bincount(y, minlength=n).astype(np.float32)
    w = torch.tensor(1.0 / np.sqrt(c + 1.0), device=DEV)
    return w / w.mean()


def train_one(net, idx_tr, idx_va, data, rng, tensors, epochs=EPOCHS, pat=PAT,
              tag="", seed=SEED):
    fseq_t, nseq_t, len_t, ctx_t, tr_t, ya_t, yp_t, ys_t = tensors
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    wa = class_weights(data["y_action"][idx_tr], 15)
    wp = class_weights(data["y_point"][idx_tr], 10)
    cea = nn.CrossEntropyLoss(weight=wa, label_smoothing=LABEL_SMOOTH)
    cep = nn.CrossEntropyLoss(weight=wp, label_smoothing=LABEL_SMOOTH)
    bce = nn.BCEWithLogitsLoss()
    best = -1.0; bad = 0; best_state = None
    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(idx_tr))
        for st in range(0, len(idx_tr), BATCH):
            bi = idx_tr[order[st:st + BATCH]]
            f, nn_, L = trunc_augment(fseq_t[bi], nseq_t[bi], len_t[bi], rng, 0.5)
            c = ctx_t[bi].to(DEV); td = tr_t[bi].to(DEV)
            la, lp, ls = net(f, nn_, L, c, td)
            loss = (cea(la, ya_t[bi].to(DEV)) + cep(lp, yp_t[bi].to(DEV))
                    + W_SERVER * bce(ls, ys_t[bi].to(DEV)))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step()
        sched.step()
        if idx_va is None:
            continue
        net.eval()
        with torch.no_grad():
            pa, pp = [], []
            for st in range(0, len(idx_va), 1024):
                bi = idx_va[st:st + 1024]
                la, lp, _ = net(fseq_t[bi].to(DEV), nseq_t[bi].to(DEV), len_t[bi].to(DEV),
                                ctx_t[bi].to(DEV), tr_t[bi].to(DEV))
                pa.append(la.softmax(-1).cpu().numpy()); pp.append(lp.softmax(-1).cpu().numpy())
            pa = np.concatenate(pa); pp = np.concatenate(pp)
            score = f1a(data["y_action"][idx_va], pa.argmax(1)) + f1p(data["y_point"][idx_va], pp.argmax(1))
        if score > best + 1e-4:
            best = score; bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= pat:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return ep, best


def predict(net, idx, tensors):
    fseq_t, nseq_t, len_t, ctx_t, tr_t, *_ = tensors
    net.eval()
    A = np.zeros((len(idx), 15), np.float32); P = np.zeros((len(idx), 10), np.float32)
    S = np.zeros((len(idx),), np.float32)
    with torch.no_grad():
        for st in range(0, len(idx), 1024):
            bi = idx[st:st + 1024]
            la, lp, ls = net(fseq_t[bi].to(DEV), nseq_t[bi].to(DEV), len_t[bi].to(DEV),
                             ctx_t[bi].to(DEV), tr_t[bi].to(DEV))
            A[st:st + 1024] = la.softmax(-1).cpu().numpy()
            P[st:st + 1024] = lp.softmax(-1).cpu().numpy()
            S[st:st + 1024] = torch.sigmoid(ls).cpu().numpy()
    return A, P, S


def run_oof(data, tensors, seed=SEED):
    """Player-grouped StratifiedGroupKFold OOF. Inner early-stop = player-disjoint."""
    striker = data["striker"]; y_action = data["y_action"]
    n = len(striker)
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    folds = list(sgkf.split(np.zeros(n), y_action, groups=striker))
    for fi, (tr_i, va_i) in enumerate(folds):
        ov = set(striker[tr_i]) & set(striker[va_i])
        assert len(ov) == 0, f"fold {fi} NOT player-disjoint (overlap={len(ov)})"
    log("ASSERT OK: 5 folds player-disjoint (val strikers unseen)")

    oof_a = np.zeros((n, 15), np.float32)
    oof_p = np.zeros((n, 10), np.float32)
    oof_s = np.zeros((n,), np.float32)
    fold_fa, fold_fp, fold_auc = [], [], []
    t0 = time.time()
    for fi, (tr_i, va_i) in enumerate(folds):
        rng = np.random.default_rng(seed + fi)
        # inner early-stop split: hold out ~18% of TRAIN PLAYERS (player-disjoint)
        tr_players = np.unique(striker[tr_i])
        rng.shuffle(tr_players)
        n_inner = max(1, int(0.18 * len(tr_players)))
        inner_va_pl = set(tr_players[:n_inner].tolist())
        is_inner_va = np.array([p in inner_va_pl for p in striker[tr_i]])
        inner_va = tr_i[is_inner_va]; inner_tr = tr_i[~is_inner_va]

        net, _ = make_net()
        ep, best = train_one(net, inner_tr, inner_va, data, rng, tensors,
                             tag=f"f{fi}", seed=seed)
        A, P, S = predict(net, va_i, tensors)
        oof_a[va_i] = A; oof_p[va_i] = P; oof_s[va_i] = S
        fa = f1a(y_action[va_i], A.argmax(1)); fp = f1p(data["y_point"][va_i], P.argmax(1))
        try:
            auc = roc_auc_score(data["sgp"][va_i], S)
        except Exception:
            auc = float("nan")
        fold_fa.append(fa); fold_fp.append(fp); fold_auc.append(auc)
        log(f"[oof s{seed}] fold {fi} n_val={len(va_i)} (inner_tr={len(inner_tr)} "
            f"inner_va={len(inner_va)}) f1a={fa:.4f} f1p={fp:.4f} auc={auc:.4f} "
            f"ep={ep} best={best:.4f} t={time.time()-t0:.0f}s")
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return oof_a, oof_p, oof_s, (fold_fa, fold_fp, fold_auc)


def run_test_refit(data, tensors, test_pack, seeds=(42, 43, 44)):
    """Refit on ALL train (player-disjoint inner early-stop), predict 1845 test rallies,
    seed-averaged."""
    uids_s, t_fseq, t_nseq, t_len, t_ctx, transd_te = test_pack
    striker = data["striker"]; n = len(striker)
    te_tensors = (torch.from_numpy(t_fseq), torch.from_numpy(t_nseq), torch.from_numpy(t_len),
                  torch.from_numpy(t_ctx), torch.from_numpy(transd_te), None, None, None)
    acc_a = np.zeros((len(uids_s), 15), np.float64)
    acc_p = np.zeros((len(uids_s), 10), np.float64)
    acc_s = np.zeros((len(uids_s),), np.float64)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        all_idx = np.arange(n)
        tr_players = np.unique(striker); rng.shuffle(tr_players)
        n_inner = max(1, int(0.12 * len(tr_players)))
        inner_va_pl = set(tr_players[:n_inner].tolist())
        is_inner_va = np.array([p in inner_va_pl for p in striker])
        inner_va = all_idx[is_inner_va]; inner_tr = all_idx[~is_inner_va]
        net, _ = make_net()
        ep, best = train_one(net, inner_tr, inner_va, data, rng, tensors,
                             tag=f"refit_s{seed}", seed=seed)
        A, P, S = predict(net, np.arange(len(uids_s)), te_tensors)
        acc_a += A; acc_p += P; acc_s += S
        log(f"[refit s{seed}] ep={ep} best={best:.4f} -> test preds added")
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    ns = len(seeds)
    return uids_s, (acc_a / ns).astype(np.float32), (acc_p / ns).astype(np.float32), (acc_s / ns).astype(np.float32)


def summarize(oof_a, oof_p, oof_s, data, folds_metrics):
    y_a = data["y_action"]; y_p = data["y_point"]; sgp = data["sgp"]
    pred_a = oof_a.argmax(1)
    pred_p = oof_p.argmax(1)
    pred_p_c = pred_p.copy(); pred_p_c[pred_a == 0] = 0
    FA = f1a(y_a, pred_a); FP = f1p(y_p, pred_p); FP_c = f1p(y_p, pred_p_c)
    try:
        AUC = roc_auc_score(sgp, oof_s)
    except Exception:
        AUC = float("nan")
    rare_a = {int(c): round(float(f1_score(y_a == c, pred_a == c, zero_division=0)), 4) for c in RARE_A}
    rare_p = {int(c): round(float(f1_score(y_p == c, pred_p == c, zero_division=0)), 4) for c in RARE_P}
    fold_fa, fold_fp, fold_auc = folds_metrics
    return dict(
        oof_f1a=round(FA, 4), oof_f1p=round(FP, 4), oof_f1p_constrained=round(FP_c, 4),
        oof_server_auc=round(float(AUC), 4),
        fold_f1a=[round(x, 4) for x in fold_fa],
        fold_f1p=[round(x, 4) for x in fold_fp],
        fold_server_auc=[round(float(x), 4) for x in fold_auc],
        f1a_mean=round(float(np.mean(fold_fa)), 4), f1a_std=round(float(np.std(fold_fa)), 4),
        f1p_mean=round(float(np.mean(fold_fp)), 4), f1p_std=round(float(np.std(fold_fp)), 4),
        auc_mean=round(float(np.nanmean(fold_auc)), 4), auc_std=round(float(np.nanstd(fold_auc)), 4),
        rare_action=rare_a, rare_point=rare_p,
    )


def main(seeds_test=(42, 43, 44), oof_seed=SEED, do_test=True):
    t0 = time.time()
    log("=" * 80)
    log("v1080 transformer + transductive (player-agnostic, player-grouped CV)")
    log(f"DEVICE={DEV} cuda={torch.cuda.is_available()} "
        f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
    log("=" * 80)
    data = build_train_pack()
    tensors = (
        torch.from_numpy(data["fseq"]), torch.from_numpy(data["nseq"]),
        torch.from_numpy(data["lengths"]), torch.from_numpy(data["ctx"]),
        torch.from_numpy(data["transd"]),
        torch.from_numpy(data["y_action"]), torch.from_numpy(data["y_point"]),
        torch.from_numpy(data["sgp"]),
    )

    # ---- player-grouped OOF ----
    oof_a, oof_p, oof_s, fm = run_oof(data, tensors, seed=oof_seed)
    np.save(OUT / "oof_action.npy", oof_a)
    np.save(OUT / "oof_point.npy", oof_p)
    np.save(OUT / "oof_server.npy", oof_s)
    # save aligned order for downstream (same as v1075)
    pd.read_csv(V1075_OUT / "target_row_order.csv").to_csv(OUT / "target_row_order.csv", index=False)
    np.save(OUT / "y_action.npy", data["y_action"])
    np.save(OUT / "y_point.npy", data["y_point"])
    np.save(OUT / "groups_striker.npy", data["striker"])

    summ = summarize(oof_a, oof_p, oof_s, data, fm)
    GBDT = 0.3795
    summ["vs_gbdt_transductive_0.3795"] = round(summ["oof_f1a"] - GBDT, 4)
    summ["pred_LB_decomp"] = round(0.4 * (summ["oof_f1a"] + summ["oof_f1p"]) + 0.2 * 0.84, 4)
    if summ["oof_f1a"] >= GBDT + 0.005:
        verdict = "BEAT_GBDT"
    elif summ["oof_f1a"] >= GBDT - 0.005:
        verdict = "MATCH"
    else:
        verdict = "STALL"
    summ["verdict"] = verdict
    summ["gpu"] = (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    log(f"\n=== OOF (player-grouped) f1a={summ['oof_f1a']:.4f} f1p={summ['oof_f1p']:.4f} "
        f"(constr {summ['oof_f1p_constrained']:.4f}) server_auc={summ['oof_server_auc']:.4f} ===")
    log(f"    fold f1a {summ['f1a_mean']:.4f}±{summ['f1a_std']:.4f} | "
        f"f1p {summ['f1p_mean']:.4f}±{summ['f1p_std']:.4f} | auc {summ['auc_mean']:.4f}±{summ['auc_std']:.4f}")
    log(f"    rare_a={summ['rare_action']}  rare_p={summ['rare_point']}")
    log(f"    vs GBDT-transductive 0.3795: {summ['vs_gbdt_transductive_0.3795']:+.4f} | "
        f"pred_LB={summ['pred_LB_decomp']:.4f} | VERDICT={verdict}")

    # ---- test refit (skip if STALL badly to save time? -> still produce for chain) ----
    if do_test:
        log("\nRefitting on ALL train for 1845 test predictions (seed-averaged)...")
        uids_s, ta, tp, ts = run_test_refit(data, tensors, build_test_pack(), seeds=seeds_test)
        np.save(OUT / "test_action.npy", ta)
        np.save(OUT / "test_point.npy", tp)
        np.save(OUT / "test_server.npy", ts)
        np.save(OUT / "test_rally_uids.npy", uids_s)
        summ["test"] = dict(n_test=int(len(uids_s)),
                            action0_rate=round(float((ta.argmax(1) == 0).mean()), 4),
                            point0_rate=round(float((tp.argmax(1) == 0).mean()), 4),
                            server_mean=round(float(ts.mean()), 4),
                            seeds=list(seeds_test))
        log(f"    test: action0_rate={summ['test']['action0_rate']:.4f} "
            f"point0_rate={summ['test']['point0_rate']:.4f} server_mean={summ['test']['server_mean']:.4f}")

    summ["elapsed_min"] = round((time.time() - t0) / 60, 1)
    json.dump(summ, open(OUT / "summary.json", "w"), indent=2)
    log(f"\nDONE in {summ['elapsed_min']:.1f} min. summary.json written.")
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof_seed", type=int, default=SEED)
    ap.add_argument("--test_seeds", default="42,43,44")
    ap.add_argument("--no_test", action="store_true")
    a = ap.parse_args()
    main(seeds_test=tuple(int(x) for x in a.test_seeds.split(",")),
         oof_seed=a.oof_seed, do_test=not a.no_test)
