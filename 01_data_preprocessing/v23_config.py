"""Config for v3 pipeline — v2 ensemble + serve-mask + two-stage point
+ class-inverse sample weights for macro-F1."""
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = EXP_ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Action serves 15-18 can never be the prediction target (target strokes
# are strikeNumber >= 2, so never a serve). v3 masks them in both
# training (remap labels / drop rows) and post-processing.
ACTION_KEEP = list(range(0, 15))          # 0..14
ACTION_MASK = [15, 16, 17, 18]

# Two-stage point:
#   Stage 1 (binary)   : class 0 (terminal: out/net) vs 1..9 (in-court)
#   Stage 2 (9-class)  : 1..9 landing zone, only on non-terminal rows
# Final pred: p0 = P(terminal), p_k = P(non-terminal) * P(zone=k | non-terminal)
POINT_TERMINAL_CLASS = 0
POINT_INCOURT_CLASSES = list(range(1, 10))

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

SUBMISSION_OUT = OUT_DIR / "submission.csv"
OOF_ACTION_NPY = OUT_DIR / "oof_action.npy"
OOF_POINT_NPY = OUT_DIR / "oof_point.npy"
OOF_SERVER_NPY = OUT_DIR / "oof_server.npy"
TEST_ACTION_NPY = OUT_DIR / "test_action.npy"
TEST_POINT_NPY = OUT_DIR / "test_point.npy"
TEST_SERVER_NPY = OUT_DIR / "test_server.npy"

N_SPLITS = 5
SEED = 42

N_ACTION = 19
N_POINT = 10
ACTION_CLASSES = list(range(N_ACTION))
POINT_CLASSES = list(range(N_POINT))

CAT_COLS = ['actionId', 'pointId', 'spinId', 'handId',
            'strengthId', 'strikeId', 'positionId']

# Random-truncation bag size: per training rally, sample K random target
# positions that mimic the test distribution (heavy on short contexts).
# Test visible length mean = 2.9; train rally mean = 5.65.  Using K=4 keeps
# training data volume ~= original while matching test's context distribution.
N_RANDOM_TARGETS_PER_RALLY = 4

# Weight on each (train_target) sample follows an importance-sampling
# correction to match the test context distribution (visible_len).
# Test visible_len ~= discrete ~Geometric-ish with mean 2.9.
USE_TEST_MATCHED_WEIGHTS = True

LGB_MULTI = {
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'learning_rate': 0.05,
    'num_leaves': 95,
    'max_depth': -1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.2,
    'verbose': -1,
    'seed': SEED,
    'num_threads': -1,
    'force_col_wise': True,
}

LGB_BIN = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 95,
    'max_depth': -1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.2,
    'verbose': -1,
    'seed': SEED,
    'num_threads': -1,
    'force_col_wise': True,
}

# XGBoost — CUDA GPU (RTX 4090).  `device='cuda'` + `tree_method='hist'` is
# the current idiomatic combo (replaces the deprecated gpu_hist).
XGB_MULTI = {
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'learning_rate': 0.06,
    'max_depth': 7,
    'min_child_weight': 4,
    'subsample': 0.85,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.5,
    'tree_method': 'hist',
    'device': 'cuda',
    'verbosity': 0,
    'seed': SEED,
}

XGB_BIN = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.06,
    'max_depth': 7,
    'min_child_weight': 4,
    'subsample': 0.85,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.5,
    'tree_method': 'hist',
    'device': 'cuda',
    'verbosity': 0,
    'seed': SEED,
}

# CatBoost — GPU mode.  `task_type='GPU'` + `devices='0'` makes it 10-30x
# faster than CPU on high-cardinality categorical features like our 45
# cat cols.  `verbose=200` prints a metric line every 200 iterations so
# the run is visibly progressing.
CB_MULTI = {
    'loss_function': 'MultiClass',
    'learning_rate': 0.06,
    'depth': 7,
    'l2_leaf_reg': 3.0,
    'random_seed': SEED,
    'task_type': 'GPU',
    'devices': '0',
    'allow_writing_files': False,
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.85,
    'verbose': 200,
}

CB_BIN = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'learning_rate': 0.06,
    'depth': 7,
    'l2_leaf_reg': 3.0,
    'random_seed': SEED,
    'task_type': 'GPU',
    'devices': '0',
    'allow_writing_files': False,
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.85,
    'verbose': 200,
}

# Iteration caps — aggressive ceilings with early stopping usually stops
# well before.  Kept tight to bound per-fit wall time on 55k × 135 feats.
LGB_NUM_ROUND = 2000
LGB_EARLY_STOP = 120
XGB_NUM_ROUND = 1500
XGB_EARLY_STOP = 100
CB_NUM_ROUND = 2000
CB_EARLY_STOP = 100

# LightGBM progress: print train/val metric every N rounds so the user can
# see the fit is actively making progress (rather than appearing to hang).
LGB_LOG_EVERY = 200
XGB_LOG_EVERY = 100

# Ensemble weights — LightGBM primary + XGBoost + CatBoost diversity
# (all GPU-accelerated except LGB which is fast on CPU).
ENSEMBLE_W = {'lgb': 0.45, 'xgb': 0.30, 'cb': 0.25}
