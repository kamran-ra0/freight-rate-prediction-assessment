from pathlib import Path

import pandas as pd

from preprocessing import (
    apply_preprocessing,
    fit_preprocessing,
    preprocessing_summary,
)


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
TABLE_DIR = ROOT / "reports" / "tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = RAW_DIR / "train-test.csv"
VALIDATION_FILE = RAW_DIR / "validation.csv"


# ---------------------------------------------------------
# Load raw data
# ---------------------------------------------------------
train_raw = pd.read_csv(TRAIN_FILE)
validation_raw = pd.read_csv(VALIDATION_FILE)


print("\nRAW DATA LOADED")
print("-" * 60)

print(f"Training rows:   {len(train_raw):,}")
print(f"Validation rows: {len(validation_raw):,}")


# ---------------------------------------------------------
# Fit preprocessing on TRAINING data only
# ---------------------------------------------------------
stats = fit_preprocessing(train_raw)


print("\nLEARNED PREPROCESSING STATISTICS")
print("-" * 60)

print("\nWeight median by equipment:")

for equipment, median in stats.weight_median_by_equipment.items():
    print(
        f"{equipment:<12} "
        f"{median:,.2f} lb"
    )

print(
    f"\nGlobal weight median: "
    f"{stats.global_weight_median:,.2f} lb"
)

print(
    f"Market index median: "
    f"{stats.market_index_median:.4f}"
)


# ---------------------------------------------------------
# Apply preprocessing
# ---------------------------------------------------------
train_clean = apply_preprocessing(
    train_raw,
    stats,
)

validation_clean = apply_preprocessing(
    validation_raw,
    stats,
)


# ---------------------------------------------------------
# Before / after summaries
# ---------------------------------------------------------
train_summary = preprocessing_summary(
    train_raw,
    train_clean,
)

validation_summary = preprocessing_summary(
    validation_raw,
    validation_clean,
)


train_summary.insert(
    0,
    "dataset",
    "train",
)

validation_summary.insert(
    0,
    "dataset",
    "validation",
)


combined_summary = pd.concat(
    [
        train_summary,
        validation_summary,
    ],
    ignore_index=True,
)


combined_summary.to_csv(
    TABLE_DIR / "preprocessing_check_summary.csv",
    index=False,
)


# ---------------------------------------------------------
# Console output
# ---------------------------------------------------------
print("\nPREPROCESSING CHECK")
print("-" * 60)

print(
    combined_summary.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# New feature checks
# ---------------------------------------------------------
print("\nNEW PREPROCESSING FEATURES")
print("-" * 60)

new_columns = [
    "weight_missing_flag",
    "weight_negative_flag",
    "weight_clean",
    "market_index_missing_flag",
    "market_index_clean",
]

for column in new_columns:
    print(
        f"{column:<30} "
        f"{str(train_clean[column].dtype):<10}"
    )


# ---------------------------------------------------------
# Flag counts
# ---------------------------------------------------------
print("\nTRAINING FLAG COUNTS")
print("-" * 60)

print(
    "weight_missing_flag:",
    int(train_clean["weight_missing_flag"].sum()),
)

print(
    "weight_negative_flag:",
    int(train_clean["weight_negative_flag"].sum()),
)

print(
    "market_index_missing_flag:",
    int(train_clean["market_index_missing_flag"].sum()),
)


print("\nVALIDATION FLAG COUNTS")
print("-" * 60)

print(
    "weight_missing_flag:",
    int(validation_clean["weight_missing_flag"].sum()),
)

print(
    "weight_negative_flag:",
    int(validation_clean["weight_negative_flag"].sum()),
)

print(
    "market_index_missing_flag:",
    int(validation_clean["market_index_missing_flag"].sum()),
)


# ---------------------------------------------------------
# Final checks
# ---------------------------------------------------------
assert train_clean["weight_clean"].notna().all()
assert validation_clean["weight_clean"].notna().all()

assert train_clean["market_index_clean"].notna().all()
assert validation_clean["market_index_clean"].notna().all()

assert (train_clean["weight_clean"] > 0).all()
assert (validation_clean["weight_clean"] > 0).all()


print("\nPREPROCESSING CHECK PASSED")

print(
    "No missing or negative values remain "
    "in the cleaned preprocessing fields."
)

print(
    "\nSummary saved to:",
    TABLE_DIR / "preprocessing_check_summary.csv",
)