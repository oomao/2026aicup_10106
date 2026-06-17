"""HEAVY transductive masked-stroke-modeling pretrain for v1054 transformer.

The #1 fix vs v1021 (which pretrained only 27s, val recon 2.68 STILL DROPPING,
early-stopped at patience 5): train the transformer to REAL convergence.
  - corpus = ALL 14995 train rallies (full) + ALL 1845 test rallies (VISIBLE
    strokes only, unlabeled) -> transductive, ~90k strokes.
  - mask 5 fields (actionId/pointId/spinId/strengthId/positionId), 15% BERT-style
    (80% MASK / 10% random / 10% keep) so the model can't trivially copy.
  - cosine LR with warmup, AdamW wd, gradient clip.
  - up to 200 epochs, patience 20 on val recon (NOT 5). Runs to convergence.
  - serverGetPoint NEVER used. No supervised target labels.

Saves outputs/pretrained_encoder.pt
"""
from __future__ import annotations
import sys, time, json, math, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D
from model import TransformerEncoder, MSMHead

OUT = HERE.parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- config ----
SEED = 42
D_MODEL = 160
N_LAYERS = 4
N_HEADS = 8
DROPOUT = 0.2
EMB_DROP = 0.1
MASK_FRAC = 0.15      # BERT-style fraction
EPOCHS = 200
BATCH = 256
LR = 1.5e-3
WARMUP = 8
PAT = 20             # generous patience -> real convergence
PRE_MAXLEN = 24
WD = 0.05

MASK_FIELDS = MSMHead.MASK_FIELDS
MF_IDX = {f: D.FIELD_LIST.index(f) for f in MASK_FIELDS}


def build_corpus():
    packs = D.build_sequences()
    sf, sn_, is_test = [], [], []
    for name in ["train", "test"]:
        pack = packs[name]
        ridx = D.build_rally_index(pack)
        field = pack["field"]; num = pack["num"]
        for ru, (s, e) in ridx.items():
            L = e - s
            if L < 2:
                continue
            L = min(L, PRE_MAXLEN)
            sf.append(field[s:s + L]); sn_.append(num[s:s + L]); is_test.append(name == "test")
    return sf, sn_, np.array(is_test)


def pad_batch(fields, nums, max_len):
    B = len(fields); nf = fields[0].shape[1]
    fb = np.zeros((B, max_len, nf), dtype=np.int64)
    nb = np.zeros((B, max_len, D.N_NUM), dtype=np.float32)
    lens = np.zeros(B, dtype=np.int64)
    for b, (f, n) in enumerate(zip(fields, nums)):
        L = min(len(f), max_len)
        fb[b, max_len - L:] = f[-L:]; nb[b, max_len - L:] = n[-L:]; lens[b] = L
    return fb, nb, lens


def make_masked(fb, lens, rng):
    """BERT-style masking of MASK_FIELDS at MASK_FRAC of valid positions.

    For each chosen (position, field): 80% -> MASK token, 10% -> random valid
    category, 10% -> keep original. Loss computed on all chosen positions.
    """
    B, Lm, nf = fb.shape
    inp = fb.copy()
    valid = np.zeros((B, Lm), dtype=bool)
    for b in range(B):
        valid[b, Lm - lens[b]:] = True
    targets = {}
    for f, j in MF_IDX.items():
        chosen = valid & (rng.random((B, Lm)) < MASK_FRAC)
        tgt = np.full((B, Lm), -100, dtype=np.int64)
        tgt[chosen] = fb[chosen, j]
        targets[f] = tgt
        # decide action per chosen
        r = rng.random((B, Lm))
        do_mask = chosen & (r < 0.8)
        do_rand = chosen & (r >= 0.8) & (r < 0.9)
        inp[do_mask, j] = D.MASK
        if do_rand.any():
            rand_vals = rng.integers(D.OFFSET, D.VOCAB[f], size=do_rand.sum())
            inp[do_rand, j] = rand_vals
        # remaining 10% keep original (no change)
    return inp, targets, valid


def lr_at(step_epoch):
    if step_epoch < WARMUP:
        return (step_epoch + 1) / WARMUP
    prog = (step_epoch - WARMUP) / max(1, EPOCHS - WARMUP)
    return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    seed = args.seed
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    print(f"[pretrain:s{seed}] building transductive corpus...", flush=True)
    sf, sn_, is_test = build_corpus()
    n = len(sf)
    n_strokes = sum(len(x) for x in sf)
    print(f"[pretrain] corpus rallies={n} strokes={n_strokes} test_share={is_test.mean():.3f}", flush=True)

    perm = rng.permutation(n)
    n_val = int(0.08 * n)
    val_set = set(perm[:n_val].tolist())
    tr_ix = [i for i in range(n) if i not in val_set]
    va_ix = [i for i in range(n) if i in val_set]
    print(f"[pretrain] train={len(tr_ix)} val={len(va_ix)}", flush=True)

    enc = TransformerEncoder(D_MODEL, N_LAYERS, N_HEADS, dropout=DROPOUT, emb_drop=EMB_DROP).to(DEV)
    head = MSMHead(enc.out_dim).to(DEV)
    nparams = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in head.parameters())
    print(f"[pretrain] params={nparams/1e6:.2f}M  d_model={D_MODEL} layers={N_LAYERS} heads={N_HEADS}", flush=True)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=LR, weight_decay=WD)
    ce = nn.CrossEntropyLoss(ignore_index=-100)

    def run_epoch(ixs, train, ep):
        enc.train(train); head.train(train)
        if train:
            for g in opt.param_groups:
                g["lr"] = LR * lr_at(ep)
        order = rng.permutation(len(ixs)) if train else np.arange(len(ixs))
        tot, cnt = 0.0, 0
        for st in range(0, len(ixs), BATCH):
            bi = [ixs[order[k]] for k in range(st, min(st + BATCH, len(ixs)))]
            fb, nb, lens = pad_batch([sf[i] for i in bi], [sn_[i] for i in bi], PRE_MAXLEN)
            inp, targets, valid = make_masked(fb, lens, rng)
            inp_t = torch.from_numpy(inp).to(DEV)
            nb_t = torch.from_numpy(nb).to(DEV)
            pad_mask = torch.from_numpy(~valid).to(DEV)
            with torch.set_grad_enabled(train):
                h = enc(inp_t, nb_t, key_padding_mask=pad_mask)
                logits = head(h)
                loss = 0.0
                for f in MASK_FIELDS:
                    loss = loss + ce(logits[f].reshape(-1, logits[f].size(-1)),
                                     torch.from_numpy(targets[f]).reshape(-1).to(DEV))
                if train:
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 2.0)
                    opt.step()
            tot += loss.item() * len(bi); cnt += len(bi)
        return tot / max(cnt, 1)

    best = 1e9; bad = 0; best_state = None; best_ep = 0
    curve = []
    for ep in range(EPOCHS):
        trl = run_epoch(tr_ix, True, ep)
        with torch.no_grad():
            val = run_epoch(va_ix, False, ep)
        curve.append({"epoch": ep, "train": round(trl, 4), "val": round(val, 4)})
        if ep % 5 == 0 or ep < 10:
            print(f"[pretrain] ep {ep:03d} train={trl:.4f} val={val:.4f} lr={LR*lr_at(ep):.2e} t={time.time()-t0:.0f}s", flush=True)
        if val < best - 1e-4:
            best = val; bad = 0; best_ep = ep
            best_state = {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}
        else:
            bad += 1
            if bad >= PAT:
                print(f"[pretrain] early stop at ep {ep} (best ep {best_ep} val {best:.4f})", flush=True)
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}
    tag = "" if seed == SEED else f"_s{seed}"
    torch.save({"state": best_state,
                "cfg": dict(d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                            dropout=DROPOUT, emb_drop=EMB_DROP),
                "best_val": best, "best_ep": best_ep},
               OUT / f"pretrained_encoder{tag}.pt")
    json.dump({"best_val_recon": best, "best_ep": best_ep, "epochs_ran": ep + 1,
               "corpus_n": n, "corpus_strokes": int(n_strokes),
               "test_share": float(is_test.mean()), "mask_frac": MASK_FRAC,
               "params_M": round(nparams / 1e6, 3), "curve": curve},
              open(OUT / f"pretrain_summary{tag}.json", "w"), indent=2)
    print(f"[pretrain:s{seed}] DONE best_val={best:.4f} @ep{best_ep} t={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
