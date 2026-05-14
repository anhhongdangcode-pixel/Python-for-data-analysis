"""validate_features.py — Kiểm tra feature set trước khi train XGBoost.

Pipeline chuẩn trước khi train:
    STEP 1 — Sanity check cơ bản
             Shape, dtypes, NaN rate, constant columns, duplicate rows
    STEP 2 — Leakage check
             Spearman correlation giữa mỗi feature với LF inputs raw
             Flag bất kỳ feature nào có |rho| > 0.5 với LF inputs
    STEP 3 — Feature quality check
             Distribution, outliers, near-zero variance
             Xem feature có đủ signal hay không trước khi đưa vào model
    STEP 4 — Label distribution check
             fomo_prob distribution sau khi drop all-abstain
             Class balance estimate nếu binarize

Chạy:
    python validate_features.py
"""

import pandas as pd
import numpy as np
from scipy import stats
from constants import OUTPUT_DIR

FEATURES_FILE = f"{OUTPUT_DIR}/fomo_features.csv"
LF_INPUT_FILE = f"{OUTPUT_DIR}/lf_input.csv"

# Threshold để flag leakage
LEAKAGE_THRESHOLD = 0.5

# LF input columns — những cột đã dùng để tạo label, KHÔNG được overlap
LF_INPUT_COLS = [
    "return_5d",
    "rsi_14",
    "price_above_bollinger",
    "price_below_bollinger",
    "asset_buy_count_same_day",
    "asset_buy_count_p95_60d",
    "totalValue",
    "p90_trade_value",
    "days_since_last_buy",
]


def sep(title="", width=60):
    if title:
        print(f"\n{'='*width}")
        print(f"  {title}")
        print(f"{'='*width}")
    else:
        print("=" * width)


# ── Load ──────────────────────────────────────────────────────────────────
sep("LOADING DATA")
features_df = pd.read_csv(FEATURES_FILE, parse_dates=["timestamp"])
lf_input_df = pd.read_csv(LF_INPUT_FILE, parse_dates=["timestamp"])

merged = features_df.merge(
    lf_input_df[["tx_id"] + LF_INPUT_COLS],
    on="tx_id", how="left"
)

META_COLS = ["tx_id", "investor_id", "timestamp", "fomo_prob"]
FEAT_COLS = [c for c in features_df.columns if c not in META_COLS]

print(f"  Features file: {features_df.shape}")
print(f"  LF input file: {lf_input_df.shape}")
print(f"  Merged shape:  {merged.shape}")
print(f"  Feature columns ({len(FEAT_COLS)}): {FEAT_COLS}")

# ── Tích lũy các cột bị loại qua từng bước ────────────────────────────────
dropped_cols = set()


def active_feats():
    """Trả về danh sách feature chưa bị loại."""
    return [c for c in FEAT_COLS if c not in dropped_cols]


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — SANITY CHECKS
# ════════════════════════════════════════════════════════════════════════════
sep("STEP 1: SANITY CHECKS")

# 1a. Shape & dtypes
print(f"\n[1a] Shape: {features_df.shape}")
print(f"     Rows: {len(features_df):,} | Features: {len(FEAT_COLS)}")

# 1b. Duplicate rows
n_dup = features_df.duplicated(subset=["tx_id"]).sum()
print(f"\n[1b] Duplicate tx_id: {n_dup}")
if n_dup > 0:
    print(f"     ⚠️  WARNING: {n_dup} duplicate tx_ids — kiểm tra join logic")
else:
    print(f"     ✓ Không có duplicate")

# 1c. NaN rate per feature
print(f"\n[1c] NaN rate per feature:")
nan_rates = features_df[active_feats()].isna().mean().sort_values(ascending=False)
high_nan  = nan_rates[nan_rates > 0.3]
med_nan   = nan_rates[(nan_rates > 0.05) & (nan_rates <= 0.3)]
low_nan   = nan_rates[(nan_rates > 0) & (nan_rates <= 0.05)]
zero_nan  = nan_rates[nan_rates == 0]

print(f"     NaN = 0%:        {len(zero_nan)} features ✓")
print(f"     NaN 0-5%:        {len(low_nan)} features (acceptable)")
print(f"     NaN 5-30%:       {len(med_nan)} features (check warmup)")
print(f"     NaN > 30%:       {len(high_nan)} features ⚠️")

if len(high_nan) > 0:
    print(f"\n     Features với NaN > 30% → bị loại khỏi các bước sau:")
    for col, rate in high_nan.items():
        print(f"       DROP {col:<45} {rate*100:.1f}%")
        dropped_cols.add(col)

if len(med_nan) > 0:
    print(f"\n     Features với NaN 5-30% (warmup expected):")
    for col, rate in med_nan.items():
        print(f"       {col:<45} {rate*100:.1f}%")

# 1d. Constant / near-constant columns
print(f"\n[1d] Constant / near-constant columns (nunique <= 2):")
for col in active_feats():
    n_unique = features_df[col].nunique(dropna=True)
    if n_unique <= 1:
        print(f"     ⚠️  CONSTANT: {col} (nunique={n_unique}) → DROP")
        dropped_cols.add(col)
    elif n_unique == 2:
        val_counts = features_df[col].value_counts(normalize=True)
        dominant   = val_counts.iloc[0]
        if dominant > 0.95:
            print(f"     ⚠️  NEAR-CONSTANT: {col} — dominant {dominant*100:.1f}% → DROP")
            dropped_cols.add(col)

# 1e. Data type check
print(f"\n[1e] Non-numeric features (cần encode trước khi train):")
non_numeric = [c for c in active_feats() if features_df[c].dtype == "object"]
if non_numeric:
    for col in non_numeric:
        print(f"     ⚠️  {col}: {features_df[col].dtype} → DROP")
        dropped_cols.add(col)
else:
    print(f"     ✓ Tất cả features đều numeric")

print(f"\n  → Sau Step 1: loại {len(dropped_cols)} features, còn lại {len(active_feats())}")


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — LEAKAGE CHECK  (chỉ test features chưa bị loại)
# ════════════════════════════════════════════════════════════════════════════
sep("STEP 2: LEAKAGE CHECK — Spearman vs LF Inputs")

step2_feats = active_feats()
print(f"\n  Features đưa vào check: {len(step2_feats)} (đã bỏ {len(dropped_cols)} từ Step 1)")
print(f"  Threshold: |rho| > {LEAKAGE_THRESHOLD} → flag leakage")
print(f"  LF inputs checked: {LF_INPUT_COLS}\n")

leakage_flags = []
leakage_table = []

for feat in step2_feats:
    row = {"feature": feat}
    max_rho = 0
    max_lf  = ""

    for lf_col in LF_INPUT_COLS:
        mask = merged[feat].notna() & merged[lf_col].notna()
        if mask.sum() < 100:
            row[lf_col] = np.nan
            continue
        rho, pval = stats.spearmanr(merged.loc[mask, feat], merged.loc[mask, lf_col])
        row[lf_col] = round(rho, 3)
        if abs(rho) > abs(max_rho):
            max_rho = rho
            max_lf  = lf_col

    row["max_|rho|"] = round(abs(max_rho), 3)
    row["max_lf"]    = max_lf
    leakage_table.append(row)

    if abs(max_rho) > LEAKAGE_THRESHOLD:
        leakage_flags.append((feat, max_lf, max_rho))

leakage_df = pd.DataFrame(leakage_table).set_index("feature")

print(f"  {'Feature':<45} {'Max |rho|':>10}  {'With LF input'}")
print(f"  {'-'*45} {'-'*10}  {'-'*25}")
for _, row in leakage_df.sort_values("max_|rho|", ascending=False).iterrows():
    flag = "⚠️  LEAKAGE" if row["max_|rho|"] > LEAKAGE_THRESHOLD else "✓"
    print(f"  {row.name:<45} {row['max_|rho|']:>10.3f}  {row['max_lf']:<25} {flag}")

if leakage_flags:
    print(f"\n  ⚠️  {len(leakage_flags)} features bị flag leakage → DROP:")
    for feat, lf_col, rho in sorted(leakage_flags, key=lambda x: abs(x[2]), reverse=True):
        print(f"     DROP: {feat:<40}  rho={rho:+.3f} với {lf_col}")
        dropped_cols.add(feat)
else:
    print(f"\n  ✓ Không có feature nào vượt threshold {LEAKAGE_THRESHOLD}")

leakage_df.to_csv(f"{OUTPUT_DIR}/leakage_check.csv")
print(f"\n  Saved: {OUTPUT_DIR}/leakage_check.csv")
print(f"\n  → Sau Step 2: loại tổng {len(dropped_cols)} features, còn lại {len(active_feats())}")


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — FEATURE QUALITY CHECK  (chỉ test features chưa bị loại)
# ════════════════════════════════════════════════════════════════════════════
sep("STEP 3: FEATURE QUALITY CHECK")

step3_feats = active_feats()
print(f"\n  Features đưa vào check: {len(step3_feats)} (đã bỏ {len(dropped_cols)} từ Step 1-2)")

# 3a. Basic stats
print(f"\n[3a] Descriptive statistics:")
desc = features_df[step3_feats].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])
print(desc.T[["mean", "std", "min", "1%", "50%", "99%", "max"]].to_string())

# 3b. Outlier check
print(f"\n[3b] Extreme outliers (ngoài mean ± 5*std):")
for col in step3_feats:
    series = features_df[col].dropna()
    if len(series) < 10:
        continue
    mean, std = series.mean(), series.std()
    if std == 0:
        continue
    pct_extreme = ((series - mean).abs() > 5 * std).mean() * 100
    if pct_extreme > 1.0:
        print(f"  ⚠️  {col:<45} {pct_extreme:.1f}% extreme values — consider clipping")

# 3c. Redundancy check
print(f"\n[3c] Highly correlated feature pairs (|rho| > 0.85 → redundant):")
numeric_feats = [c for c in step3_feats if features_df[c].dtype != "object"]
corr_matrix   = features_df[numeric_feats].corr(method="spearman")

redundant_pairs  = []
redundant_to_drop = set()
for i in range(len(numeric_feats)):
    for j in range(i+1, len(numeric_feats)):
        rho = corr_matrix.iloc[i, j]
        if abs(rho) > 0.85:
            f1, f2 = numeric_feats[i], numeric_feats[j]
            redundant_pairs.append((f1, f2, rho))
            # Giữ f1, loại f2 (cột đứng sau)
            redundant_to_drop.add(f2)

if redundant_pairs:
    for f1, f2, rho in sorted(redundant_pairs, key=lambda x: abs(x[2]), reverse=True):
        kept = "← keep" if f2 in redundant_to_drop else "← keep"
        print(f"  ⚠️  {f1:<40} × {f2:<40}  rho={rho:+.3f}  DROP: {f2}")
    dropped_cols.update(redundant_to_drop)
    print(f"\n  → Loại {len(redundant_to_drop)} redundant features")
else:
    print(f"  ✓ Không có cặp nào vượt threshold 0.85")

# 3d. Near-zero variance
print(f"\n[3d] Near-zero variance (CV < 0.01):")
nzv_found = False
for col in active_feats():   # dùng active_feats() vì redundant đã được thêm vào dropped_cols
    series = features_df[col].dropna()
    if len(series) < 10 or series.mean() == 0:
        continue
    cv = series.std() / abs(series.mean())
    if cv < 0.01:
        print(f"  ⚠️  {col:<45} CV={cv:.4f} → DROP")
        dropped_cols.add(col)
        nzv_found = True
if not nzv_found:
    print(f"  ✓ Tất cả features có variance đủ lớn")

print(f"\n  → Sau Step 3: loại tổng {len(dropped_cols)} features, còn lại {len(active_feats())}")


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — LABEL DISTRIBUTION CHECK  (chỉ dùng features còn sống sót)
# ════════════════════════════════════════════════════════════════════════════
sep("STEP 4: LABEL DISTRIBUTION CHECK")

step4_feats = active_feats()
print(f"\n  Features đưa vào check: {len(step4_feats)} (đã bỏ {len(dropped_cols)} từ Step 1-3)")

fomo = features_df["fomo_prob"]

print(f"\n[4a] fomo_prob distribution:")
print(f"     N       : {len(fomo):,}")
print(f"     Mean    : {fomo.mean():.4f}")
print(f"     Median  : {fomo.median():.4f}")
print(f"     Std     : {fomo.std():.4f}")
print(f"     Min/Max : {fomo.min():.4f} / {fomo.max():.4f}")

print(f"\n[4b] fomo_prob buckets:")
buckets = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
labels  = ["[0.0, 0.2)", "[0.2, 0.4)", "[0.4, 0.6)", "[0.6, 0.8)", "[0.8, 1.0]"]
for i, label in enumerate(labels):
    mask = (fomo >= buckets[i]) & (fomo < buckets[i+1])
    bar  = "█" * int(mask.mean() * 50)
    print(f"     {label}  {bar:<50} {mask.sum():>7,} ({mask.mean()*100:5.1f}%)")

print(f"\n[4c] Binarize estimate (threshold = 0.5):")
fomo_binary = (fomo > 0.5).astype(int)
n_fomo   = fomo_binary.sum()
n_normal = len(fomo_binary) - n_fomo
ratio    = n_fomo / len(fomo_binary)
print(f"     FOMO   (1): {n_fomo:>7,} ({ratio*100:.1f}%)")
print(f"     NORMAL (0): {n_normal:>7,} ({(1-ratio)*100:.1f}%)")

if ratio < 0.1 or ratio > 0.6:
    print(f"     ⚠️  Class imbalance — scale_pos_weight = {n_normal/n_fomo:.2f}")
else:
    print(f"     ✓ Class balance acceptable")

# 4d. Correlation với label — chỉ features còn sống sót
print(f"\n[4d] Correlation với fomo_prob — chỉ {len(step4_feats)} features còn lại:")
label_corr = []
for col in step4_feats:
    mask = features_df[col].notna() & fomo.notna()
    if mask.sum() < 100:
        continue
    rho, pval = stats.spearmanr(features_df.loc[mask, col], fomo[mask])
    label_corr.append((col, rho, pval))

label_corr_df = pd.DataFrame(label_corr, columns=["feature", "rho", "pval"])
label_corr_df = label_corr_df.reindex(
    label_corr_df["rho"].abs().sort_values(ascending=False).index
)

print(f"\n     {'Feature':<45} {'rho':>8}  {'p-value':>12}  Signal?")
print(f"     {'-'*45} {'-'*8}  {'-'*12}  -------")
for _, row in label_corr_df.iterrows():
    sig = "✓ strong" if abs(row["rho"]) > 0.15 else (
          "~ weak"   if abs(row["rho"]) > 0.05 else
          "✗ no signal")
    print(f"     {row['feature']:<45} {row['rho']:>+8.3f}  {row['pval']:>12.2e}  {sig}")

no_signal = label_corr_df[label_corr_df["rho"].abs() <= 0.05]
if len(no_signal) > 0:
    print(f"\n     ⚠️  {len(no_signal)} features gần như không correlate với label:")
    for _, row in no_signal.iterrows():
        print(f"       {row['feature']:<45} rho={row['rho']:+.4f}")
    dropped_cols.update(no_signal["feature"].tolist())  # ← thêm dòng này

print(f"\n  → Sau Step 4: loại tổng {len(dropped_cols)} features, còn lại {len(active_feats())}")


# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════
sep("SUMMARY — ACTION ITEMS")

final_feats = active_feats()
print(f"\n  Tổng features ban đầu : {len(FEAT_COLS)}")
print(f"  Tổng features bị loại : {len(dropped_cols)}")
print(f"  Features còn lại      : {len(final_feats)}")
print(f"\n  Features bị loại theo từng bước đã được ghi nhận inline ở trên.")

issues = []
if n_dup > 0:
    issues.append(f"[CRITICAL]  {n_dup} duplicate tx_ids — fix join logic")
if ratio < 0.1 or ratio > 0.6:
    issues.append(f"[IMBALANCE] scale_pos_weight = {n_normal/n_fomo:.2f} cho XGBoost")

if len(issues) == 0:
    print(f"\n  ✓ Không có issue bổ sung. Feature set sẵn sàng để train XGBoost.")
else:
    print(f"\n  {len(issues)} items còn cần xử lý:")
    for item in issues:
        print(f"     {item}")

print(f"\n✓ Done. Files saved:")
print(f"  {OUTPUT_DIR}/leakage_check.csv")
print(f"\nNext step: fix issues → train XGBoost với {len(final_feats)} features")