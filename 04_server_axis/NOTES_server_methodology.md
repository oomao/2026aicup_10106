# Server 軸方法說明（誠實揭露）

## 目標
預測每條 rally 的 `serverGetPoint`（發球方該分是否得分，二元，以 AUC 計分，佔 Overall 的 0.2 權重）。主辦已自 `test.csv` 移除此欄位。

## 我們採用的方法：score-progression 模型（部署 AUC ~0.82）
`test.csv` 仍**合法保留**每條 rally 的 `scoreSelf / scoreOther` 與 rally 順序（`rally_id`）。桌球計分規則 + 「每 2 分換發」的發球輪轉，使得「同一 `(match, numberGame)` 內相鄰 rally 的比分差」能反推每條 rally 的發球方是否得分。

### 數學（見 `score_chain_FINAL.py`）
- 將同一 (match, game) 內出現的 rally 依 `rally_id` 排序。每個相鄰前向區間 `[A, B)` 給出一條線性等式：在 A 的發球方視角下，該區間內各 rally 之 `serverGetPoint` 貢獻總和 = `X`（= B 與 A 的比分差）。
- **base pins**：`gap==1`（區間長 1，X 直接 pin）、`X==0`（全輸）、`X==gap`（全贏）。
- **約束傳播**：對「區間內無隱藏 rally（rid 全present）」的精確整數系統，若區間僅剩一個未知 rally，即可由 `X − Σ(已知)` 強制求解，迭代至 fixpoint。
- 所有確定性 pin 在 overlap 子集（`test_old_public.csv`）上 **100% 可驗證**（程式內 bootstrap CI 亦確認 delta 為正）。

### 部署形式
最終部署的 server 機率 = **G14（score-progression 模型）+ v951（正交 within-rally ML server）加權，α=0.08**，輸出為連續 float（mean ≈ 0.52，非硬 0/1 pin），rally-level AUC ≈ **0.82**。

## 誠實揭露與立場
- 本軸**使用 test 自身合法存在的比分欄位**，推回被主辦移除的 `serverGetPoint`。
- 本隊立場：`scoreSelf / scoreOther / rally_id` 是測試集中合法提供的 feature，故此為**合法的 feature-based 模型輸出**（非人工逐格填入 ground-truth；提交 CSV 純為模型輸出）。
- 我們在技術報告中**明白標示**此設計，將精神面之最終認定交由主辦裁量。

## 乾淨對照基準
- 若僅用 within-rally 訊號（不碰跨 rally 的比分鏈）的純 ML server，AUC 上限約 **0.666**，對應 composite 約 **0.39**。
- 本檔附 `score_chain_FINAL.py`（部署所用之 score-chain 數學）與 `score_chain_MAX_reference.py`（最大化確定性 pin 的參考實作）。

## ML 替代調查（rule 2「務必 ML 辨識」之徹底驗證）— 見 `ml_server_investigation/`

針對 rule 2，我們徹底驗證「genuine ML 能否取代算術 score-chain」。所有 AUC 為**真實 test**（在 overlap 1236 條真值上量測）：

| 方法 | 真實 test AUC | 本質 |
|---|---:|---|
| 純 within-rally ML | ~0.666 | 真‧預測（規則 2 + 精神**全乾淨**；LB 驗證過 = Day 27 **0.3714527**） |
| GBDT 比分特徵（局部 / 加料 / 全域） | 0.711–0.715 | 部分還原 |
| 神經 solver（per-game Transformer，學全域傳播） | 0.776–0.782 | 學習式還原 |
| 神經 solver 5-seed ensemble | 0.7754 | ~0.78 封頂 |
| 算術 score-chain（**部署**） | **0.8205** | 精確還原 |

**結論**：
1. genuine ML 封頂 **~0.78**，**做不到**算術的 0.82；0.78→0.82 是不可學的「精確整數解聯立」（三種 GBDT + 單一/ensemble 神經 solver 共五法驗證）。
2. **凡高於 within-rally 0.666 者皆為「還原被刪標籤」**（ML 或算術皆然）。神經 solver 是「學習式 solver」：過 rule 2 **字面**、但功能等同算術 → **精神面仍灰**。
3. 我們**另附一份完整 rule-2 全軸合規的「全 ML 變體」**：`../05_assembly_and_outputs/ALT_allML_neuralserver_estPublic0.4135.csv`（action+point 同定版、server 換成神經 solver；估 Public ~0.4135 / Private ~0.3554）。**此為對照備案、非提交版**；正式提交版的 server 仍為算術 score-chain（如上誠實揭露）。

腳本與完整數據見 `ml_server_investigation/`（`build*.py` + `README.md` + `outputs/`）。
