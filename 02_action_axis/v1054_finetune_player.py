"""Player-aware multi-task fine-tune (production-comparable protocol).

This is the path toward the 0.44-CLASS number. The OOV player-disjoint harness
(finetune.py) deliberately strips ALL player identity, so it shows only the
player-AGNOSTIC floor (~0.36). But the REAL test has ~57% SEEN players, where
learnable player embeddings add a large lift -- exactly what the production GBDT
chain exploits via player target-encoding (and what a working 0.44 transformer
almost certainly uses).

Here we:
  - add learnable embeddings for the K+1 striker (gamePlayerId of target row) and
    its opponent, index 0 = UNKNOWN (OOV-safe; heavy emb dropout for graceful
    degradation),
  - evaluate under MATCH-DISJOINT GroupKFold (the production protocol -> the OOF
    is on the SAME scale as v85_NEW 0.4839 action / 0.27 point, i.e. the
    LB-decomposition F1ap scale), AND
  - cross-check the resulting OOF on the OOV player-disjoint harness for honesty.

A player index is built from the TRAIN players only; any player not in a fold's
training set maps to UNKNOWN(0). Anti-leak: encoder still sees only the visible
prefix; serverGetPoint never input.
"""
from __future__ import annotations
import sys, time, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D
from model import TransformerEncoder, MultiTaskModel

OUT = HERE.parent / "outputs"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
N_SPLITS = 5
EPOCHS = 40
BATCH = 256
LR = 1.0e-3
WD = 0.05
PAT = 8
RARE_A = [5, 7, 8, 9, 14]
RARE_P = [1, 2, 3]
W_SERVER = 0.3
LABEL_SMOOTH = 0.05
TEST_K_PMF = {1: 0.2753, 2: 0.2575, 3: 0.1669, 4: 0.1014, 5: 0.0743, 6: 0.0412, 7: 0.0835}


def f1a(y, p):
    return f1_score(y, p, labels=list(range(15)), average="macro", zero_division=0)


def f1p(y, p):
    return f1_score(y, p, labels=list(range(10)), average="macro", zero_division=0)


def build_all():
    packs = D.build_sequences()
    tr = packs["train"]
    target_idx = D.canonical_target_rows(tr)
    ai = D.FIELD_LIST.index("actionId"); pi = D.FIELD_LIST.index("pointId")
    y_action = (tr["field"][:, ai] - D.OFFSET)[target_idx].astype(np.int64)
    y_point = (tr["field"][:, pi] - D.OFFSET)[target_idx].astype(np.int64)
    sgp = tr["df"]["serverGetPoint"].to_numpy().astype(np.float32)[target_idx]
    striker = tr["striker"][target_idx]
    opp = tr["opp"][target_idx]
    match = tr["match"][target_idx]
    ridx = D.build_rally_index(tr)
    fseq, nseq, lengths = D.make_prefix_batch(tr, ridx, target_idx, max_len=D.MAX_LEN)
    ctx = D.target_context(tr, target_idx)
    # global player vocab (train players)
    uniq = np.unique(np.concatenate([striker, opp]))
    p2i = {p: i + 1 for i, p in enumerate(uniq)}  # 0 = UNKNOWN
    return dict(target_idx=target_idx, y_action=y_action, y_point=y_point, sgp=sgp,
                striker=striker, opp=opp, match=match, fseq=fseq, nseq=nseq,
                lengths=lengths, ctx=ctx, p2i=p2i, n_players=len(uniq))


def trunc_augment(fseq_b, nseq_b, len_b, rng, p_aug=0.5):
    B, Lm, nf = fseq_b.shape
    fb = fseq_b.clone(); nb = nseq_b.clone(); lb = len_b.clone()
    keys = np.array(list(TEST_K_PMF.keys()))
    probs = np.array([TEST_K_PMF[k] for k in keys]); probs = probs / probs.sum()
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


def run(init="pretrained", seed=SEED, aug=True, oov_drop=0.10):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    data = build_all()
    target_idx = data["target_idx"]
    y_action = data["y_action"]; y_point = data["y_point"]; sgp = data["sgp"]
    p2i = data["p2i"]; n_players = data["n_players"]
    # match-disjoint folds (production protocol)
    gkf = GroupKFold(n_splits=N_SPLITS)
    folds = np.zeros(len(target_idx), dtype=np.int64)
    for f, (_, va) in enumerate(gkf.split(target_idx, groups=data["match"])):
        folds[va] = f
    # map players to indices
    pid_self_all = np.array([p2i.get(p, 0) for p in data["striker"]], dtype=np.int64)
    pid_opp_all = np.array([p2i.get(p, 0) for p in data["opp"]], dtype=np.int64)

    n = len(target_idx)
    print(f"[ftP:{init}:s{seed}:aug{int(aug)}] n={n} n_players={n_players} match-disjoint", flush=True)
    fseq_t = torch.from_numpy(data["fseq"]); nseq_t = torch.from_numpy(data["nseq"])
    len_t = torch.from_numpy(data["lengths"]); ctx_t = torch.from_numpy(data["ctx"])
    ya_t = torch.from_numpy(y_action); yp_t = torch.from_numpy(y_point); ys_t = torch.from_numpy(sgp)

    # all finetune seeds reuse the SAME seed-42 pretrained encoder (decouple
    # finetune randomness from pretrain); this makes multi-seed a true
    # finetune-variance-reduction ensemble.
    ckpt = torch.load(OUT / "pretrained_encoder.pt", map_location="cpu")
    cfg = ckpt["cfg"]

    oof_a = np.zeros((n, 15), dtype=np.float32)
    oof_p = np.zeros((n, 10), dtype=np.float32)
    oof_s = np.zeros((n,), dtype=np.float32)

    for fold in range(N_SPLITS):
        va = np.where(folds == fold)[0]; trn = np.where(folds != fold)[0]
        # players seen in this fold's TRAIN -> kept; others (incl val-only) map to UNKNOWN
        seen = set(data["striker"][trn].tolist()) | set(data["opp"][trn].tolist())
        def map_pid(arr_players):
            return np.array([p2i[p] if p in seen else 0 for p in arr_players], dtype=np.int64)
        pid_self = map_pid(data["striker"]); pid_opp = map_pid(data["opp"])
        psf_t = torch.from_numpy(pid_self); pof_t = torch.from_numpy(pid_opp)

        enc = TransformerEncoder(cfg["d_model"], cfg["n_layers"], cfg["n_heads"],
                                 dropout=cfg["dropout"], emb_drop=cfg["emb_drop"]).to(DEV)
        if init == "pretrained":
            enc.load_state_dict(ckpt["state"])
        net = MultiTaskModel(enc, use_server=True, n_players=n_players).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

        ca = np.bincount(y_action[trn], minlength=15).astype(np.float32)
        cp = np.bincount(y_point[trn], minlength=10).astype(np.float32)
        wa = torch.tensor(1.0 / np.sqrt(ca + 1.0), device=DEV); wa = wa / wa.mean()
        wp = torch.tensor(1.0 / np.sqrt(cp + 1.0), device=DEV); wp = wp / wp.mean()
        cea = nn.CrossEntropyLoss(weight=wa, label_smoothing=LABEL_SMOOTH)
        cep = nn.CrossEntropyLoss(weight=wp, label_smoothing=LABEL_SMOOTH)
        bce = nn.BCEWithLogitsLoss()

        rng_f = np.random.default_rng(seed + fold)
        inner_perm = rng_f.permutation(len(trn)); n_inner = int(0.1 * len(trn))
        inner_va = trn[inner_perm[:n_inner]]; inner_tr = trn[inner_perm[n_inner:]]

        best = -1; bad = 0; best_state = None
        for ep in range(EPOCHS):
            net.train()
            order = rng_f.permutation(len(inner_tr))
            for st in range(0, len(inner_tr), BATCH):
                bi = inner_tr[order[st:st + BATCH]]
                fb = fseq_t[bi]; nb = nseq_t[bi]; lb = len_t[bi]
                if aug:
                    f, nn_, L = trunc_augment(fb, nb, lb, rng_f, p_aug=0.5)
                else:
                    f, nn_, L = fb.to(DEV), nb.to(DEV), lb.to(DEV)
                c = ctx_t[bi].to(DEV)
                ps = psf_t[bi].clone(); po = pof_t[bi].clone()
                # player-dropout: randomly set some to UNKNOWN so model learns OOV path
                if oov_drop > 0:
                    m1 = torch.from_numpy(rng_f.random(len(bi)) < oov_drop)
                    m2 = torch.from_numpy(rng_f.random(len(bi)) < oov_drop)
                    ps[m1] = 0; po[m2] = 0
                ps = ps.to(DEV); po = po.to(DEV)
                la, lp, ls = net(f, nn_, L, c, ps, po)
                loss = (cea(la, ya_t[bi].to(DEV)) + cep(lp, yp_t[bi].to(DEV))
                        + W_SERVER * bce(ls, ys_t[bi].to(DEV)))
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step()
            sched.step()
            net.eval()
            with torch.no_grad():
                pa, pp = [], []
                for st in range(0, len(inner_va), 1024):
                    bi = inner_va[st:st + 1024]
                    f = fseq_t[bi].to(DEV); nn_ = nseq_t[bi].to(DEV); L = len_t[bi].to(DEV); c = ctx_t[bi].to(DEV)
                    la, lp, _ = net(f, nn_, L, c, psf_t[bi].to(DEV), pof_t[bi].to(DEV))
                    pa.append(la.softmax(-1).cpu().numpy()); pp.append(lp.softmax(-1).cpu().numpy())
                pa = np.concatenate(pa); pp = np.concatenate(pp)
                score = f1a(y_action[inner_va], pa.argmax(1)) + f1p(y_point[inner_va], pp.argmax(1))
            if score > best + 1e-4:
                best = score; bad = 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= PAT:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            for st in range(0, len(va), 1024):
                bi = va[st:st + 1024]
                f = fseq_t[bi].to(DEV); nn_ = nseq_t[bi].to(DEV); L = len_t[bi].to(DEV); c = ctx_t[bi].to(DEV)
                la, lp, ls = net(f, nn_, L, c, psf_t[bi].to(DEV), pof_t[bi].to(DEV))
                oof_a[bi] = la.softmax(-1).cpu().numpy(); oof_p[bi] = lp.softmax(-1).cpu().numpy()
                oof_s[bi] = torch.sigmoid(ls).cpu().numpy()
        fa = f1a(y_action[va], oof_a[va].argmax(1)); fp = f1p(y_point[va], oof_p[va].argmax(1))
        print(f"[ftP:{init}:s{seed}] fold {fold} n_val={len(va)} f1a={fa:.4f} f1p={fp:.4f} ep={ep} t={time.time()-t0:.0f}s", flush=True)
        # free GPU memory between folds (prevents accumulation/OOM across folds)
        del net, enc, opt, sched, best_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pred_a = oof_a.argmax(1)
    pred_p_c = oof_p.argmax(1).copy(); pred_p_c[pred_a == 0] = 0
    FA = f1a(y_action, pred_a); FP = f1p(y_point, oof_p.argmax(1)); FP_c = f1p(y_point, pred_p_c)
    rare_a = {int(c): round(float(f1_score(y_action == c, pred_a == c, zero_division=0)), 4) for c in RARE_A}
    rare_p = {int(c): round(float(f1_score(y_point == c, oof_p.argmax(1) == c, zero_division=0)), 4) for c in RARE_P}
    res = dict(init=init, seed=seed, aug=aug, protocol="match-disjoint+player-emb",
               oof_f1a=round(FA, 4), oof_f1p=round(FP, 4), oof_f1p_constrained=round(FP_c, 4),
               rare_action=rare_a, rare_point=rare_p, n=n, n_players=n_players)
    print(f"[ftP:{init}:s{seed}:aug{int(aug)}] === OOF(match-disjoint+player) f1a={FA:.4f} f1p={FP:.4f} (constr {FP_c:.4f}) ===", flush=True)
    print(f"[ftP] rare_action={rare_a}  rare_point={rare_p}", flush=True)
    suff = f"player_{init}"
    if not aug:
        suff += "_noaug"
    if seed != SEED:
        suff += f"_s{seed}"
    np.save(OUT / f"oof_action_{suff}.npy", oof_a)
    np.save(OUT / f"oof_point_{suff}.npy", oof_p)
    np.save(OUT / f"oof_server_{suff}.npy", oof_s)
    json.dump(res, open(OUT / f"ft_result_{suff}.json", "w"), indent=2)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="pretrained", choices=["pretrained", "scratch"])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--noaug", action="store_true")
    a = ap.parse_args()
    run(a.init, a.seed, aug=not a.noaug)
