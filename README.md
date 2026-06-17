# 程式碼資料夾索引

本資料夾包含 AICUP 2026 桌球擊球預測競賽的**全部原始程式碼與產出物**，對應主辦要求項目 (2)(3)(4)(5)。

> **最終提交**：`05_assembly_and_outputs/FINAL_sub_day48_SPRINT_point_0.4225088.csv`（Public LB 0.4225088）
> **一鍵重現**：`py aicup_final_report/code/05_assembly_and_outputs/reproduce_final.py`

---

## 資料夾結構

```
code/
├── 01_data_preprocessing/      資料前處理（對應主辦項目 (3)）
├── 02_action_axis/             Action 軸訓練與部署（對應 (2)(4)）
├── 03_point_axis/              Point 軸訓練與部署（對應 (2)(4)）
├── 04_server_axis/             Server 軸推算（對應 (2)）
├── 05_assembly_and_outputs/    三階段組裝鏈 + 最終提交 CSV
└── requirements.txt            套件版本（Python 3.11.9，GPU CUDA 12.1）
```

---

## 01_data_preprocessing — 資料前處理

| 檔案 | 用途 |
|---|---|
| `v23_config.py` | 全域設定、GBDT 超參、集成權重（對應主辦項目 (5)） |
| `v23_features.py` | 生產線主特徵：lag、target encoding、發球、比分、累計統計 |
| `v23_matchup_features.py` | 對手 matchup 轉移張量，per-fold 計算防止資料洩漏 |
| `v23_postprocess.py` | 後處理：action=0 → point=0 強制約束 |
| `v701_within_match_features.py` | within-match leave-self-out 條件機率特徵 |
| `v1341_features_clean.py` | 乾淨版 within-match 特徵，移除 terminal stroke leak |
| `v1080_transductive_context.py` | 16 維截斷對齊透析向量（供 Transformer 使用） |
| `v1054_sequence_data.py` | Transformer 序列資料、prefix batch、context 組裝 |

---

## 02_action_axis — Action 軸

### GBDT 生產線（最終 seen 球員 842 rallies 使用）

| 檔案 | 用途 |
|---|---|
| `v23_train_gbdt_ensemble.py` | 生產線 GBDT（LGB + XGB + CatBoost）5-fold GroupKFold 訓練 |
| `v85_NEW_production_blend.py` | Nelder-Mead action-only 凸組合 + 4-gate 驗證 |

### Transformer（最終 OOV 球員 1003 rallies 使用）

| 檔案 | 用途 |
|---|---|
| `v1054_transformer_encoder.py` | Transformer encoder 架構定義 + MSM 自監督 head |
| `v1054_pretrain_selfsupervised.py` | 自監督預訓練（AICUP-only 語料，無外部資料） |
| `v1054_finetune_player.py` | player-grouped finetune |
| `v1054_infer_test.py` | 測試集推論工具 |
| `v1080_model.py` | transductive 多任務模型定義（player-agnostic） |
| `v1080_train.py` | player-grouped CV 訓練 + seed-averaged test refit |
| `v1080_build_deploy_OOVgated.py` | ★ OOV-gating 組裝（seen → GBDT / OOV → v1080） |
| `v1080_NOTES.md` | v1080 設計說明 |
| `pretrained_weights/pretrained_encoder.pt` | 自監督預訓練 encoder 權重 |

---

## 03_point_axis — Point 軸

| 檔案 | 用途 |
|---|---|
| `v701_build_within_match.py` | within-match 透析 LGB 訓練（match + player GroupKFold） |
| `v701_overlay.py` | ★ drag-zone overlay → Stage-1 提交（+0.0009 LB） |
| `v1341_build_lean.py` | 乾淨 richer within-match GBDT，400 棵樹，多 seed |
| `v1341_build_full.py` | 同上，700 棵樹完整版 |
| `v1341_overlay_FINAL.py` | ★ 最終 41-flip drag-zone overlay（BAL 變體）→ 最終提交 |
| `v1341_NOTES.md` | v1341 設計說明 |

---

## 04_server_axis — Server 軸

| 檔案 / 資料夾 | 用途 |
|---|---|
| `score_chain_FINAL.py` | ★ 確定性 score-progression 重建（部署所用演算法） |
| `score_chain_MAX_reference.py` | 最大化確定性 pin 參考版本 |
| `NOTES_server_methodology.md` | 誠實揭露 + 乾淨 ML 對照基準 + 調查結論 |
| `ml_server_investigation/` | rule-2 完整驗證：純 ML 能否取代算術 server |
| `ml_server_investigation/build.py` | GBDT 局部版 server（AUC ~0.71） |
| `ml_server_investigation/build_global.py` | GBDT 全域版 server（AUC ~0.78） |
| `ml_server_investigation/build_rich.py` | GBDT 加料版 |
| `ml_server_investigation/build_neural_solver.py` | 神經 solver 單一版（AUC ~0.82） |
| `ml_server_investigation/build_neural_ensemble.py` | 神經 solver ensemble 版 |
| `ml_server_investigation/build_ml_server_submission.py` | 組「全 ML 變體」對照 CSV |
| `ml_server_investigation/outputs/` | summary*.json + 神經 server 預測 .npy |

---

## 05_assembly_and_outputs — 三階段組裝鏈

### 核心腳本

| 檔案 | 用途 |
|---|---|
| `reproduce_final.py` | ★ 一鍵自包含重現（只用本資料夾內檔案，byte-identical，exit 0） |

### 三階段 CSV（依序產生）

| 檔案 | 階段 | Public LB |
|---|---|---:|
| `base0_agree1_D27action_G14v951server_0.4132214.csv` | Base（生產線錨點） | 0.4132214 |
| `stage1_v701_point_overlay_0.4141329.csv` | Stage 1：+point within-match overlay | 0.4141329 |
| `stage2_v1080_OOVgated_action_0.4207553.csv` | Stage 2：+OOV action swap | 0.4207553 |
| `FINAL_sub_day48_SPRINT_point_0.4225088.csv` | ★ Stage 3：+point drag-zone overlay（**最終提交**） | **0.4225088** |
| `ALT_allML_neuralserver_estPublic0.4135.csv` | 對照備案：神經 server（非提交版） | ~0.4135 |

### 重現用機率檔

```
reproduction_npy/
├── v85_NEW/    oof_point.npy、test_point.npy
├── v701/       oof_point.npy、test_point.npy、test_rally_uids.npy
├── v1341/      oof/test_point_{BAL,NOBAL}.npy、y_point.npy、test_rally_uids.npy
└── v1080/      test_action.npy、test_rally_uids.npy
```

---

## 重現流程

```
# 1. 安裝套件
pip install -r requirements.txt

# 2. 確認資料放置（路徑以 E:/AICUP_O 為根）
#    aicup_final_report/data/train.csv
#    aicup_final_report/data/test.csv

# 3. 一鍵重現（三階段驗證 + 最終 CSV byte-identical）
py aicup_final_report/code/05_assembly_and_outputs/reproduce_final.py

# 若要重新訓練各模型，依序執行：
py aicup_final_report/code/02_action_axis/v23_train_gbdt_ensemble.py
py aicup_final_report/code/02_action_axis/v1054_pretrain_selfsupervised.py
py aicup_final_report/code/02_action_axis/v1080_train.py
py aicup_final_report/code/03_point_axis/v701_build_within_match.py
py aicup_final_report/code/03_point_axis/v1341_build_lean.py
```

> 訓練硬體：NVIDIA GeForce RTX 4090 24GB，CUDA 12.1。詳見 `../06_environment/ENVIRONMENT.md`。
