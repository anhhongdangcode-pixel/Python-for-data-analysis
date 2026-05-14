import os
import json
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import shap
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    log_loss, roc_auc_score, average_precision_score,
    precision_score, recall_score, fbeta_score
)
from sklearn.model_selection import TimeSeriesSplit
 
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
 
from constants import OUTPUT_DIR, MODEL_DIR, XGBOOST_MODEL_FILE
 
FEATURES_FILE    = f"{OUTPUT_DIR}/fomo_features.csv"
PREDICTIONS_FILE = f"{OUTPUT_DIR}/fomo_predictions.csv"
SHAP_FILE        = f"{OUTPUT_DIR}/shap_values.csv"
OPTUNA_PARAMS_FILE = f"{MODEL_DIR}/optuna_best_params.json"
RUN_METADATA_FILE  = f"{MODEL_DIR}/run_metadata.json"
 
# ── Config ────────────────────────────────────────────────────────────────
RANDOM_STATE     = 42
OPTUNA_TRIALS    = 50
OPTUNA_TIMEOUT   = 300
WEIGHT_EPSILON   = 0.01
CV_FOLDS         = 5
 
FOMO_HIGH_THRESH   = 0.65
FOMO_LOW_THRESH    = 0.35
FOMO_MEDIUM_THRESH = 0.40
 
# Danh sách loại bỏ để tránh Leakage (đã cập nhật theo danh sách của Luân)
NON_FEATURE_COLS = [
    "tx_id", "investor_id", "timestamp", "fomo_prob", "momentum_acceleration",
    "trade_gap_days", "total_value_pctrank_asset", "rolling_avg_position_size_last_10",
    "position_size_to_volatility_ratio", "position_size_ratio", "trades_per_investor_per_day",
    "same_day_multiple_flag", "return_1d", "market_fomo_pressure_score",
    "asset_popularity_zscore", "volatility_regime", "rolling_trade_freq_5",
    "price_distance_high", "macd", "macd_hist", "macd_signal",
    "investor_trade_index", "volatility_5d_rolling_std_10", "volatility_10d",
    "risk_level", "return_1d_rolling_std_10", "volatility_10d_pctrank_asset",
    "digital_trade_flag"
]
 
def assign_fomo_level(score):
    if score >= FOMO_HIGH_THRESH:     return "High"
    elif score >= FOMO_MEDIUM_THRESH: return "Medium"
    else:                             return "Low"
 
def prepare(subset, feature_cols):
    X = subset[feature_cols].fillna(subset[feature_cols].median())
    y = subset["fomo_prob"].values
    w = np.abs(y - 0.5) * 2 + WEIGHT_EPSILON
    return X, y, w
 
# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 & 2 — Load & Prepare Data
# ═══════════════════════════════════════════════════════════════════════════
print("🚀 Loading and preparing full dataset...")
df = pd.read_csv(FEATURES_FILE, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
X_full, y_full, w_full = prepare(df, feature_cols)
 
# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — Optuna Tuning (TS-CV)
# ═══════════════════════════════════════════════════════════════════════════
print(f"🎯 Optuna tuning ({OPTUNA_TRIALS} trials)...")
tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
 
def objective(trial):
    params = {
        "objective": "reg:logistic", "eval_metric": "logloss", "tree_method": "hist",
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": RANDOM_STATE
    }
    scores = []
    for tr_idx, val_idx in tscv.split(X_full):
        model = xgb.XGBRegressor(**params)
        model.fit(X_full.iloc[tr_idx], y_full[tr_idx], sample_weight=w_full[tr_idx])
        p = np.clip(model.predict(X_full.iloc[val_idx]), 1e-7, 1 - 1e-7)
        scores.append(log_loss((y_full[val_idx] > 0.5).astype(int), p, sample_weight=w_full[val_idx]))
    return np.mean(scores)
 
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=OPTUNA_TRIALS, timeout=OPTUNA_TIMEOUT)
best_params = study.best_params
# In kết quả tốt nhất của Optuna
print(f"\n🎯 Best CV LogLoss: {study.best_value:.6f}")
print("📌 Best Hyperparameters found:")
for key, value in best_params.items():
    print(f"    {key:<20}: {value}")

# ── Save Optuna best params ────────────────────────────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)
optuna_save = {
    "best_params":   best_params,
    "best_cv_logloss": study.best_value,
    "n_trials":      OPTUNA_TRIALS,
    "n_completed":   len(study.trials),
    "timeout_sec":   OPTUNA_TIMEOUT,
}
with open(OPTUNA_PARAMS_FILE, "w") as f:
    json.dump(optuna_save, f, indent=4)
print(f"✓ Optuna params saved  → {OPTUNA_PARAMS_FILE}")
 
# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — Evaluation (TS-CV)
# ═══════════════════════════════════════════════════════════════════════════
print("📊 Evaluation via TimeSeriesSplit CV...")
df_hc = df[(df["fomo_prob"] > FOMO_HIGH_THRESH) | (df["fomo_prob"] < FOMO_LOW_THRESH)].copy()
df_hc["fomo_label"] = (df_hc["fomo_prob"] > FOMO_HIGH_THRESH).astype(int)
X_hc, y_hc, w_hc = prepare(df_hc, feature_cols)
 
cv_res = []
oof_preds = np.full(len(df_hc), np.nan)
 
for tr_idx, te_idx in tscv.split(X_hc):
    cv_model = xgb.XGBRegressor(objective="reg:logistic", **best_params)
    cv_model.fit(X_hc.iloc[tr_idx], df_hc["fomo_label"].values[tr_idx], sample_weight=w_hc[tr_idx])
    p = np.clip(cv_model.predict(X_hc.iloc[te_idx]), 1e-7, 1 - 1e-7)
    oof_preds[te_idx] = p
    cv_res.append([roc_auc_score(df_hc["fomo_label"].values[te_idx], p), 
                   log_loss(df_hc["fomo_label"].values[te_idx], p),
                   np.sqrt(mean_squared_error(df_hc["fomo_label"].values[te_idx], p))])
 
# ═══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY & COMPARISON
# Baseline = Mean Value (per project evaluation guidelines)
# Metrics: MAE, RMSE, R² (Regression evaluation)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL PERFORMANCE SUMMARY — REGRESSION EVALUATION")
print("=" * 70)
 
valid_mask   = ~np.isnan(oof_preds)
y_valid      = df_hc["fomo_label"].values[valid_mask]
p_valid      = oof_preds[valid_mask]
 
# ── Mean Value Baseline ────────────────────────────────────────────────────
# Baseline: always predict the training mean — simplest possible benchmark
y_mean       = y_valid.mean()
baseline_preds = np.full(len(y_valid), y_mean)
 
# ── Regression Metrics: Baseline ──────────────────────────────────────────
base_mae  = mean_absolute_error(y_valid, baseline_preds)
base_rmse = np.sqrt(mean_squared_error(y_valid, baseline_preds))
base_r2   = r2_score(y_valid, baseline_preds)          # always 0 by definition
 
# ── Regression Metrics: XGBoost Model ─────────────────────────────────────
model_mae  = mean_absolute_error(y_valid, p_valid)
model_rmse = np.sqrt(mean_squared_error(y_valid, p_valid))
model_r2   = r2_score(y_valid, p_valid)
 
# ── Print Comparison Table ─────────────────────────────────────────────────
print(f"\nTable: Performance Comparison (XGBoost vs. Mean Baseline)")
print(f"{'-'*70}")
print(f"{'Strategy':<35} | {'MAE':<10} | {'RMSE':<10} | {'R²':<10}")
print(f"{'-'*35}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
print(f"{'Mean Baseline':<35} | {base_mae:<10.4f} | {base_rmse:<10.4f} | {base_r2:<10.4f}")
print(f"{'Proposed XGBoost Model':<35} | {model_mae:<10.4f} | {model_rmse:<10.4f} | {model_r2:<10.4f}")
print(f"{'-'*70}")
print(f"\nImprovement over Baseline:")
print(f"  MAE  reduction : {(base_mae  - model_mae)  / base_mae  * 100:+.2f}%")
print(f"  RMSE reduction : {(base_rmse - model_rmse) / base_rmse * 100:+.2f}%")
print(f"  R²   gain      : {model_r2 - base_r2:+.4f}")
 
mean_auc = np.mean([r[0] for r in cv_res])
std_auc  = np.std([r[0] for r in cv_res])
print(f"\nLabel Learnability Score (CV):")
print(f"  Mean AUC = {mean_auc:.4f} ± {std_auc:.4f}")
 
# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — Train Final Model & Save Artifacts
# ═══════════════════════════════════════════════════════════════════════════
final_model = xgb.XGBRegressor(objective="reg:logistic", **best_params)
final_model.fit(X_full, y_full, sample_weight=w_full)

os.makedirs(MODEL_DIR, exist_ok=True)
final_model.save_model(XGBOOST_MODEL_FILE)
print(f"\n✓ Model saved          → {XGBOOST_MODEL_FILE}")

# ── Save run metadata (params + all eval metrics) ─────────────────────────
import datetime
run_metadata = {
    "saved_at":         datetime.datetime.now().isoformat(timespec="seconds"),
    "model_file":       XGBOOST_MODEL_FILE,
    "optuna_params_file": OPTUNA_PARAMS_FILE,
    "config": {
        "random_state":   RANDOM_STATE,
        "optuna_trials":  OPTUNA_TRIALS,
        "optuna_timeout": OPTUNA_TIMEOUT,
        "cv_folds":       CV_FOLDS,
        "n_features":     len(feature_cols),
        "n_train_rows":   len(df),
    },
    "best_params": best_params,
    "optuna": {
        "best_cv_logloss": study.best_value,
        "n_completed_trials": len(study.trials),
    },
    "eval_metrics": {
        "baseline_mae":   round(base_mae,  6),
        "baseline_rmse":  round(base_rmse, 6),
        "baseline_r2":    round(base_r2,   6),
        "model_mae":      round(model_mae,  6),
        "model_rmse":     round(model_rmse, 6),
        "model_r2":       round(model_r2,   6),
        "mae_reduction_pct":  round((base_mae  - model_mae)  / base_mae  * 100, 4),
        "rmse_reduction_pct": round((base_rmse - model_rmse) / base_rmse * 100, 4),
        "r2_gain":            round(model_r2 - base_r2, 6),
        "cv_auc_mean":    round(mean_auc, 6),
        "cv_auc_std":     round(std_auc,  6),
        "cv_fold_results": [
            {"fold": i + 1, "auc": round(r[0], 6), "logloss": round(r[1], 6), "rmse": round(r[2], 6)}
            for i, r in enumerate(cv_res)
        ],
    },
}
with open(RUN_METADATA_FILE, "w") as f:
    json.dump(run_metadata, f, indent=4)
print(f"✓ Run metadata saved   → {RUN_METADATA_FILE}")
print(f"\n✓ Process Complete.")