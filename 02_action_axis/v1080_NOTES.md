# v1080 — Transformer + truncation-matched transductive context (player-agnostic)

## Goal
Beat the v1075 GBDT-transductive **player-grouped (OOV-honest)** ceiling **f1a 0.3795** with the
pretrained v1054 transformer, by injecting the same de-confounded transductive signal that the
GBDT used — and judging on the honest player-grouped ruler (not the match-CV mirage where v1054
player-emb showed 0.407 but dies on the real unseen-player test).

## Architecture change vs v1054 finetune_player
- **Encoder**: REUSED verbatim from v1054 (`pretrained_encoder.pt`, d=160, 4 layers, 8 heads,
  MSM-pretrained). NOT re-pretrained. Sees only strokes 1..K (prefix); K+1 never enters.
- **Player-id embeddings: DISABLED** (`n_players=0`, no `player_emb` module at all). Player identity
  is the thing that collapses on the 8 unseen test players (rule: player-TE −0.10 f1a on a
  player-grouped ruler) — so it is removed entirely.
- **Transductive context (16-dim) ADDED** via a parallel projection
  `nn.Sequential(Linear(16,d), GELU, LayerNorm, Dropout(0.2))`, whose output is concatenated into
  the pooled representation: `pooled = [last || mean || attn || ctx_proj(ctx) || transd_proj(transd)]`
  → shared trunk → action(15)/point(10)/server(1) heads. (`model_v1080.py::TransductiveMultiTask`.)
- Multi-task loss: class-weighted CE (1/sqrt(count)) + label smoothing 0.05 for action & point,
  + 0.3·BCE for server. Truncation augmentation (test K-vis PMF) during training.

## The transductive feature (the signal that scored 0.3795)
Per target row, a 16-dim vector: the K+1 striker's **same-match PRIOR-rally action proportions**
`mp_action_0..14` + a `mp_has` flag. This is exactly v1075's de-confounded `D_trunc` block:
- **Train** (`transductive.build_train_transductive`): reuses v1075's
  `build_prior_style(truncate=True)` verbatim — cumulative-EXCLUSIVE over `rally_id` within
  `(match, gamePlayerId)`, each prior rally truncated to a visible prefix K_vis ~ test K-vis PMF,
  **5-seed averaged** (seeds 101/202/303/404/505). Merged on `(match, gamePlayerId, rally_id)` onto
  the canonical target-row order (bit-exact with v1054 `canonical_target_rows` and v1075
  `target_row_order.csv` — asserted in code, incl. `y_action` equality check vs v1075).
- **Test** (`transductive.build_test_transductive`): test matches are DISJOINT from train, so prior
  rallies come from `test.csv` itself. The K+1 striker per rally is derived by the parity rule
  (identical to v1075 `test_time_coverage`). The striker's action proportions are summed over the
  VISIBLE strokes of all PRIOR test rallies (same `(match, striker)`, `rally_id < current`). Test
  prior rallies are ALREADY truncated (they are real truncated test rallies), so the visible strokes
  ARE the truncation-matched sample — no extra truncation step needed. Returned sorted by `rally_uid`
  (matches the infer-test output order; asserted).

Coverage: train `mp_has` ≈ 0.925 (train rallies are long → most strikers have an in-match prior),
test `mp_has` ≈ 0.749 (≈ v1075's reported 0.762). Rows with no prior get a zero vector + `mp_has=0`
(graceful OOV — the projection sees an all-zero input and the model falls back to the encoder repr).

## Ruler (the antidote to OOF→LB death)
`StratifiedGroupKFold(5, shuffle=True, random_state=42)` grouped by **target striker gamePlayerId**,
stratified by `actionId`. All 5 folds asserted player-disjoint (val strikers unseen). Critically the
**inner early-stop / model-selection split is ALSO player-disjoint** (carved from ~18% of the train
fold's PLAYERS, not random rows) — so model selection never peeks at a player it will be scored on.

## Anti-leak checklist (all enforced in code)
- Encoder input = strokes 1..K only; target stroke K+1 and anything after never enter.
- Transductive features = striker's PRIOR in-match rallies only (`rally_id < current`),
  truncation-matched, normalized proportions (not counts → no K_full leak), multi-seed averaged.
  Never the target row's own attributes.
- serverGetPoint NEVER an input (not in tokens, not in transductive, not in ctx).
- Player-grouped folds, zero striker overlap (asserted); inner split player-disjoint too.
- Train transductive bit-exactly reproduces v1075's `D_trunc` block (`feat_cols == MP_COLS`,
  `y_action == v1075 y_action` asserted).

## GPU (rule O4)
PyTorch on RTX 4090 (24 GB free). Model 1.45M params, ~14 MB GPU. Device logged in summary.json.

## Files
- `src/transductive.py` — train + test truncation-matched 16-dim transductive builders.
- `src/model_v1080.py` — `TransductiveMultiTask` (encoder + transductive projection, no player-emb).
- `src/run.py` — player-grouped OOF + refit-on-all test inference + summary.
- `outputs/` — `oof_{action,point,server}.npy` (aligned to `target_row_order.csv`),
  `test_{action,point,server}.npy` + `test_rally_uids.npy` (1845, sorted), `summary.json`,
  `full_run.log`.

## RESULTS — (filled after run)
TBD.
