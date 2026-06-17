# 0.4225088 (Public) / 0.3643962 (Private, 40/423) — v1341 SPRINT drag-zone point overlay — Day 49, 2026-06-02

Δ **+0.0017535 vs prior record 0.4207553** (Public). File: `sub_day48_SPRINT_point.csv`. **Final clean submission.**
**Final official: Private LB 0.3643962, rank 40/423 (top 9.5%).**

> **Deployed variant = `BAL`** (class-balanced). overlay_BAL.json: alpha=0.3, margin=0.286, **41 flips**, p0=0.2499.
> The NOBAL variant gives only 35 flips and does NOT reproduce this submission.
> **Verified reproduction**: `py code/05_assembly_and_outputs/reproduce_final.py` → byte-identical, exit 0.

## What it is
Record (`sub_day48_v1080_OOVgated.csv`) + a SELECTIVE drag-zone POINT overlay (rule 101): **41 point flips** into zones {5×22, 4×10, 1×9} (from {0,2,7,8,9,...}). actionId + serverGetPoint byte-identical to the record. point0_rate 0.2547→0.2499 (in band). a0→p0 0 violations.

## Source: v1341 — CLEAN richer transductive point GBDT (the v1340 leak FIXED)
- v1340 leaked via `wm_opponent_serve` block (no leave-self-out → memorized rally outcome in 2-player matches). Removing it dropped OOF class-0 f1 0.857→0.443 (healthy).
- v1341 = v701's 6 within-match blocks + clean richer blocks (zone-transition `wm_player_prevpoint` +0.0073 standalone, score-conditioned, ss-bucket, opponent-plain). Player-grouped StratifiedGroupKFold, multi-seed.
- OOV f1p: BAL **0.3147 (+0.0419 over v701 0.2728)**, NOBAL 0.2986. class0_f1 0.43 (clean, no leak). Drag-zone OOF lift: z1 +0.138, z4 +0.057, z5 +0.052.

## THE LESSON: targeted drag-zone overlay TRANSFERS where push-1 didn't
- push-1 (v701+v1250 2-way soft-vote, margin 0.10, 95 flips, into class-6) REGRESSED −0.0018 (recall-inflated, didn't transfer).
- v1341 SPRINT (clean richer GBDT, 41 flips into the recoverable drag zones 1/4/5, p0 in band) TRANSFERRED **+0.00175**. Transfer ratio ~0.06 of the +0.0295 capped OOF (low, but POSITIVE).
- The deployable point lever = **few, high-margin, drag-zone flips from a clean (no-leak) richer-transductive model**, NOT a 2-way soft-vote into the modal/recall-classes.
- CAVEAT (honest): the v1341 STANDALONE full-axis fails rule-96 (corr_test 0.95 >> oof 0.81 = OOF-overfit); only the SELECTIVE drag overlay (subset rule-96 −0.118 SAFE) is deployable. The gamble (~40-50% est) paid off.

## Standing
- **New clean record: 0.4225088** (v1341 SPRINT drag-point overlay on the v1080-OOVgated record). Clean (transformer-transductive action + v701-chain point + drag overlay + D29 score-chain server). 628 teams; top is 0.58 (confirmed source-video reverse-match leak, organizer-banned → DQ-risk on the 6/3 review). Our score is method-based, clean.
- Day 49 = 2 slots used (floor 0.4207 + this 0.4225088). Deadline midnight 6/3.
