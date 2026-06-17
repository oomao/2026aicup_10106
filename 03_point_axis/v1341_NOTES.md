# v1341 — CLEAN richer within-match transductive POINT GBDT

## Goal
Beat v701 (OOV/player-held-out point macro-F1 = **0.2728**) DEPLOYABLY (no terminal leak),
with RICHER transductive within-match features. v1340 tried a richer version but **LEAKED
terminal** (OOF class-0 f1 0.866, test p0 0.33) and was dead. Do it right: find & remove the
leak, keep the clean signal.

## Leak diagnosis (the v1340 death)
Two single-fold ablation diagnostics (player-held-out CV, GPU LGB 400 trees) over v1340's
full 173-feature matrix:

**Backward (remove one new block):**
| ablation | macroF1 | class0_f1 | verdict |
|---|---|---|---|
| FULL (v1340) | 0.7713 | 0.8686 | LEAK |
| minus wm_player_prevpoint | 0.7670 | 0.8627 | still leaks |
| minus wm_player_score | 0.7718 | 0.8567 | still leaks |
| minus wm_player_ssbucket | 0.7711 | 0.8592 | still leaks |
| minus wm_opponent | 0.7682 | 0.8583 | still leaks |
| **minus wm_opponent_serve** | **0.3267** | **0.4430** | **LEAK REMOVED (healthy)** |
| minus transd_mp_point | 0.7682 | 0.8634 | still leaks |
| v701-equiv (drop ALL new) | 0.2758 | 0.4246 | healthy (≈ v701 0.2728) |

**=> `wm_opponent_serve` is the sole dominant terminal leak.**
Mechanism: keyed `[match, opponent_id, serve_spin, serve_point]` with NO leave-self-out.
In a 2-player match these fine cells become near rally-unique and include the opponent's
in-rally stroke, so the conditional memorizes the rally outcome → encodes terminal (class 0).
v701 itself never had an opponent×serve block — this was a v1340 addition.

**Forward (BASE=v701-equiv + one block):**
| add | macroF1 | class0_f1 |
|---|---|---|
| BASE (v701-equiv) | 0.2756 | 0.4237 |
| BASE + wm_player_prevpoint | **0.2829 (+0.0073)** | 0.4169 (healthy) |

The zone-TRANSITION prior adds real, leak-free signal. (Forward sweep of the remaining
blocks was cut for GPU time; the build's full-5-fold gate is the authoritative leak check.)

## The clean build (v1341)
Features = v701's 6 within-match blocks + CLEAN richer blocks
(wm_player_prevpoint zone-transition, wm_player_score, wm_player_ssbucket, wm_opponent plain)
+ truncation-matched prior-rally striker point distribution (transd_mp_point) + raw context +
v85_NEW residual.  **wm_opponent_serve DROPPED.**  162 features (v1340 had 173).
- Player-grouped StratifiedGroupKFold(5) by striker (OOV-honest) + MATCH for context.
- BAL (class_weight=balanced) + NOBAL variants. Multi-seed (3 OOF, 5 refit) prob-avg.
- Anti-leak gates: OOF class-0 f1 < 0.5, |p0_drift| < 0.05, rule-96 (corr_test-corr_oof<0.03).
- NOTE: run on CPU LightGBM (env V1341_FORCE_CPU=1) because an unrelated GPU job (tennis
  sprint) made shared-GPU LGB 5x slower than CPU for this 53k×162 data (84s vs 17.6s/fit).
  Rule O4's intent is speed; GPU was the slower path here.

## RESULTS — standalone (PLAYER-held-out = OOV-honest), CLEAN (no terminal leak)
| variant | OOV f1p | vs v701 0.2728 | class0_f1 (leak<0.5) | oof_p0 | test_p0 | p0_drift | rule96 diff |
|---|---|---|---|---|---|---|---|
| **BAL** | **0.3147** | **+0.0419** | 0.4343 ✓ | 0.2114 | 0.1220 | -0.0895 | +0.132 ✗ |
| **NOBAL** | **0.2986** | **+0.0258** | 0.4414 ✓ | 0.2504 | 0.1913 | -0.0591 | +0.120 ✗ |

Both BEAT v701 OOV by far more than +0.003 and are CLEAN (class0_f1 ~0.43, NOT v1340's 0.866).
The richer transductive blocks add real, leak-free signal. BUT the STANDALONE full-axis fails
two deploy gates: p0_drift (terminal over-suppressed on test) and rule-96 (corr_test >> corr_oof
= the Day-19 OOF→LB-overfit red flag). So a full axis SWAP would be risky / p0-band-illegal.

## RESULTS — selective POINT OVERLAY on the record sub (rule 101) = the actual deliverable
Tuned the DRAG-zone flip rule on the v85_NEW OOF at the capped budget, applied gated to test.
**The standalone rule-96 leak-flag does NOT propagate to the capped DRAG overlay** — on the
flipped subset corr_test < corr_oof (SAFE), exactly like v701's profile.

| variant | OOF capped Δ macroF1p | test flips | test p0 | rule96 diff | a0 viol | deploy |
|---|---|---|---|---|---|---|
| **BAL (STAGED)** | **+0.0295** | 41 | 0.2499 | **-0.118 (SAFE)** | 0 | ✓ |
| NOBAL | +0.0241 | 35 | 0.2526 | -0.059 (SAFE) | 0 | ✓ |
| v701 (reference) | +0.0157 | 100 | 0.2542 | -0.121 (SAFE) | 0 | ✓ |

BAL overlay per-zone OOF lift: z1 +0.138, z3 +0.078, z4 +0.057, z5 +0.052; z0(terminal) -0.002
(preserved). Flips INTO {1:9, 4:10, 5:22}, FROM over-predicted {0:9 (false terminals removed),
9:10, 8:7, 7:4, 2:5, ...}. action + serverGetPoint BYTE-IDENTICAL to the record sub.

**The v1341 overlay's OOF capped delta (+0.0295) is ~1.9x v701's (+0.0157)** at a smaller test
flip count (41 vs 100) — the clean richer features make each flip higher-precision.

## VERDICT: **STAGE** (clean, no-leak, beats v701 deployably)
- Staged: `E:\AICUP_O\result\staging_day48\sub_day48_SPRINT_point.csv` (BAL overlay).
- Beats v701 OOV: standalone +0.0419 (BAL) / +0.0258 (NOBAL); overlay OOF capped Δ +0.0295 vs
  v701's +0.0157.  No terminal leak (class0_f1 0.43, p0 in band, a0-viol 0, rule-96 SAFE on the
  overlay subset).
- HONEST CAVEAT: the STANDALONE model fails rule-96 (corr_test>>corr_oof) — a transfer-risk flag.
  The selective overlay neutralizes it (rule-96 SAFE on the flipped DRAG subset), so the overlay
  is the correct integration (matches v701's mechanism). Predicted composite LB transfer of a
  point overlay is historically ~0.4-0.6 of OOF (rule 101); at +0.0295 OOF capped that is
  ~+0.005-0.007 composite over the 0.4132 record — but realized lift could land lower given the
  standalone rule-96 flag, so treat as a real-but-uncertain upgrade, not a guaranteed one.
- The v1340 leak (wm_opponent_serve) is DEAD and excluded; class0_f1 0.43 confirms no terminal leak.
