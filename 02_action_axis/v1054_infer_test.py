"""Produce 1845-row TEST predictions from the v1054 transformer.

Two modes:
  --mode oov     : player-AGNOSTIC encoder (finetune.py weights re-fit on ALL
                   train, no player emb) -> OOV-robust test_action/test_point.
  --mode player  : player-AWARE (player emb), refit on ALL train -> production-
                   scale test_action/test_point. OOV test players map to UNKNOWN.

For simplicity and to avoid a second training pass, this script RE-FITS the model
on ALL canonical train rows (no held-out fold) for the chosen mode and seeds,
then predicts the 1845 test target strokes. Anti-leak: each test rally's prefix =
its VISIBLE strokes 1..K (the whole visible rally); target = stroke K+1 (never
seen). serverGetPoint never input. Outputs are saved sorted by rally_uid.
"""
from __future__ import annotations
import sys, time, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D
from model import TransformerEncoder, MultiTaskModel

OUT = HERE.parent / "outputs"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 35
BATCH = 256
LR = 1.0e-3
WD = 0.05
W_SERVER = 0.3
LABEL_SMOOTH = 0.05
TEST_K_PMF = {1: 0.2753, 2: 0.2575, 3: 0.1669, 4: 0.1014, 5: 0.0743, 6: 0.0412, 7: 0.0835}


def test_target_rows(te_pack):
    """For each test rally, the K+1 prediction uses the FULL visible prefix.

    We create one 'virtual target row' per rally by taking a row index that sits
    just AFTER the last visible stroke; since test has no K+1 stroke, we build the
    prefix directly from each rally's visible strokes and the target context from
    the K+1 striker = opponent of the last visible hitter (alternation).
    """
    rally = te_pack["rally"]; strike = te_pack["strike"]
    ridx = D.build_rally_index(te_pack)
    uids = []
    fseq_list = []; nseq_list = []; lengths = []; ctx_list = []
    pid_self = []; pid_opp = []
    field = te_pack["field"]; num = te_pack["num"]
    striker = te_pack["striker"]; opp = te_pack["opp"]; sex = te_pack["sex"]
    for ru, (s, e) in ridx.items():
        L = min(e - s, D.MAX_LEN)
        pidx = np.arange(e - L, e)  # last L visible strokes (right-aligned)
        fs = np.zeros((D.MAX_LEN, field.shape[1]), dtype=np.int64)
        ns = np.zeros((D.MAX_LEN, D.N_NUM), dtype=np.float32)
        fs[D.MAX_LEN - L:] = field[pidx]; ns[D.MAX_LEN - L:] = num[pidx]
        fseq_list.append(fs); nseq_list.append(ns); lengths.append(L)
        last_strike = strike[e - 1]
        # K+1 striker = opponent of last visible hitter
        tgt_par = float((last_strike + 1) % 2)
        ctx_list.append([tgt_par, float(sex[e - 1])])
        # player ids: K+1 hitter = opp of last stroke; its opponent = striker of last
        pid_self.append(int(opp[e - 1])); pid_opp.append(int(striker[e - 1]))
        uids.append(ru)
    return (np.array(uids), np.stack(fseq_list), np.stack(nseq_list).astype(np.float32),
            np.array(lengths, dtype=np.int64), np.array(ctx_list, dtype=np.float32),
            np.array(pid_self, dtype=np.int64), np.array(pid_opp, dtype=np.int64))


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


def run(mode="player", seeds=(42, 43, 44)):
    t0 = time.time()
    packs = D.build_sequences()
    tr = packs["train"]; te = packs["test"]
    target_idx = D.canonical_target_rows(tr)
    ai = D.FIELD_LIST.index("actionId"); pi = D.FIELD_LIST.index("pointId")
    y_action = (tr["field"][:, ai] - D.OFFSET)[target_idx].astype(np.int64)
    y_point = (tr["field"][:, pi] - D.OFFSET)[target_idx].astype(np.int64)
    sgp = tr["df"]["serverGetPoint"].to_numpy().astype(np.float32)[target_idx]
    striker = tr["striker"][target_idx]; opp = tr["opp"][target_idx]
    ridx = D.build_rally_index(tr)
    fseq, nseq, lengths = D.make_prefix_batch(tr, ridx, target_idx, max_len=D.MAX_LEN)
    ctx = D.target_context(tr, target_idx)

    use_player = (mode == "player")
    if use_player:
        uniq = np.unique(np.concatenate([striker, opp]))
        p2i = {p: i + 1 for i, p in enumerate(uniq)}
        n_players = len(uniq)
        pid_self_tr = np.array([p2i.get(p, 0) for p in striker], dtype=np.int64)
        pid_opp_tr = np.array([p2i.get(p, 0) for p in opp], dtype=np.int64)
    else:
        n_players = 0; p2i = {}

    uids, t_fseq, t_nseq, t_len, t_ctx, t_pid_self, t_pid_opp = test_target_rows(te)
    if use_player:
        t_pid_self = np.array([p2i.get(p, 0) for p in t_pid_self], dtype=np.int64)
        t_pid_opp = np.array([p2i.get(p, 0) for p in t_pid_opp], dtype=np.int64)
        oov_frac = float((t_pid_self == 0).mean())
        print(f"[infer:{mode}] test target OOV (UNKNOWN) fraction={oov_frac:.3f}", flush=True)

    fseq_t = torch.from_numpy(fseq); nseq_t = torch.from_numpy(nseq); len_t = torch.from_numpy(lengths)
    ctx_t = torch.from_numpy(ctx); ya_t = torch.from_numpy(y_action); yp_t = torch.from_numpy(y_point); ys_t = torch.from_numpy(sgp)
    if use_player:
        psf_t = torch.from_numpy(pid_self_tr); pof_t = torch.from_numpy(pid_opp_tr)
    tfs = torch.from_numpy(t_fseq); tns = torch.from_numpy(t_nseq); tl = torch.from_numpy(t_len); tc = torch.from_numpy(t_ctx)
    if use_player:
        tps = torch.from_numpy(t_pid_self); tpo = torch.from_numpy(t_pid_opp)

    test_a_acc = np.zeros((len(uids), 15), dtype=np.float64)
    test_p_acc = np.zeros((len(uids), 10), dtype=np.float64)
    test_s_acc = np.zeros((len(uids),), dtype=np.float64)

    for seed in seeds:
        tag_pt = "" if seed == 42 else f"_s{seed}"
        pt = OUT / f"pretrained_encoder{tag_pt}.pt"
        if not pt.exists():
            pt = OUT / "pretrained_encoder.pt"  # fallback to seed-42 encoder
        ckpt = torch.load(pt, map_location="cpu"); cfg = ckpt["cfg"]
        torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
        enc = TransformerEncoder(cfg["d_model"], cfg["n_layers"], cfg["n_heads"],
                                 dropout=cfg["dropout"], emb_drop=cfg["emb_drop"]).to(DEV)
        enc.load_state_dict(ckpt["state"])
        net = MultiTaskModel(enc, use_server=True, n_players=n_players).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        ca = np.bincount(y_action, minlength=15).astype(np.float32); cp = np.bincount(y_point, minlength=10).astype(np.float32)
        wa = torch.tensor(1.0 / np.sqrt(ca + 1.0), device=DEV); wa = wa / wa.mean()
        wp = torch.tensor(1.0 / np.sqrt(cp + 1.0), device=DEV); wp = wp / wp.mean()
        cea = nn.CrossEntropyLoss(weight=wa, label_smoothing=LABEL_SMOOTH)
        cep = nn.CrossEntropyLoss(weight=wp, label_smoothing=LABEL_SMOOTH)
        bce = nn.BCEWithLogitsLoss()
        N = len(target_idx)
        for ep in range(EPOCHS):
            net.train(); order = rng.permutation(N)
            for st in range(0, N, BATCH):
                bi = order[st:st + BATCH]
                f, nn_, L = trunc_augment(fseq_t[bi], nseq_t[bi], len_t[bi], rng, 0.5)
                c = ctx_t[bi].to(DEV)
                if use_player:
                    ps = psf_t[bi].clone(); po = pof_t[bi].clone()
                    m1 = torch.from_numpy(rng.random(len(bi)) < 0.10); m2 = torch.from_numpy(rng.random(len(bi)) < 0.10)
                    ps[m1] = 0; po[m2] = 0
                    la, lp, ls = net(f, nn_, L, c, ps.to(DEV), po.to(DEV))
                else:
                    la, lp, ls = net(f, nn_, L, c)
                loss = cea(la, ya_t[bi].to(DEV)) + cep(lp, yp_t[bi].to(DEV)) + W_SERVER * bce(ls, ys_t[bi].to(DEV))
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            for st in range(0, len(uids), 1024):
                sl = slice(st, st + 1024)
                f = tfs[sl].to(DEV); nn_ = tns[sl].to(DEV); L = tl[sl].to(DEV); c = tc[sl].to(DEV)
                if use_player:
                    la, lp, ls = net(f, nn_, L, c, tps[sl].to(DEV), tpo[sl].to(DEV))
                else:
                    la, lp, ls = net(f, nn_, L, c)
                test_a_acc[sl] += la.softmax(-1).cpu().numpy()
                test_p_acc[sl] += lp.softmax(-1).cpu().numpy()
                test_s_acc[sl] += torch.sigmoid(ls).cpu().numpy()
        print(f"[infer:{mode}] seed {seed} done t={time.time()-t0:.0f}s", flush=True)

    ns = len(seeds)
    test_a = (test_a_acc / ns).astype(np.float32); test_p = (test_p_acc / ns).astype(np.float32); test_s = (test_s_acc / ns).astype(np.float32)
    order = np.argsort(uids)
    uids_s = uids[order]; test_a = test_a[order]; test_p = test_p[order]; test_s = test_s[order]
    suff = mode
    np.save(OUT / f"test_action_{suff}.npy", test_a)
    np.save(OUT / f"test_point_{suff}.npy", test_p)
    np.save(OUT / f"test_server_{suff}.npy", test_s)
    np.save(OUT / f"test_rally_uids_{suff}.npy", uids_s)
    info = {"mode": mode, "seeds": list(seeds), "n_test": int(len(uids_s)),
            "action0_rate": float((test_a.argmax(1) == 0).mean()),
            "point0_rate": float((test_p.argmax(1) == 0).mean()),
            "server_mean": float(test_s.mean())}
    json.dump(info, open(OUT / f"test_infer_{suff}.json", "w"), indent=2)
    print(f"[infer:{mode}] DONE {info}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="player", choices=["player", "oov"])
    ap.add_argument("--seeds", default="42,43,44")
    a = ap.parse_args()
    seeds = tuple(int(x) for x in a.seeds.split(","))
    run(a.mode, seeds)
