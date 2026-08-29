from pathlib import Path

import numpy as np
import pandas as pd


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
# Load data
# ---------------------------------------------------------
train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VALIDATION_FILE)

train["date"] = pd.to_datetime(train["date"], errors="coerce")
validation["date"] = pd.to_datetime(validation["date"], errors="coerce")


# ---------------------------------------------------------
# 1. Negative-weight investigation
# ---------------------------------------------------------
negative_train = train.loc[
    train["weight"] < 0,
    [
        "load_id",
        "equipment",
        "weight",
        "distance",
        "posted_rate",
        "date",
    ],
].copy()

negative_train["absolute_weight"] = negative_train["weight"].abs()

negative_train.to_csv(
    TABLE_DIR / "negative_weight_rows_train.csv",
    index=False,
)


negative_validation = validation.loc[
    validation["weight"] < 0,
    [
        "load_id",
        "equipment",
        "weight",
        "distance",
        "date",
    ],
].copy()

negative_validation["absolute_weight"] = negative_validation["weight"].abs()

negative_validation.to_csv(
    TABLE_DIR / "negative_weight_rows_validation.csv",
    index=False,
)


print("\nNEGATIVE WEIGHT ANALYSIS")
print("-" * 60)

print(
    f"Training negative weights: "
    f"{len(negative_train):,}"
)

print(
    f"Validation negative weights: "
    f"{len(negative_validation):,}"
)

if not negative_train.empty:
    print("\nAbsolute magnitude of negative training weights:")

    print(
        negative_train["absolute_weight"]
        .describe(
            percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]
        )
        .round(2)
    )


# Compare negative-weight magnitude with normal positive weights.
positive_weight = train.loc[
    train["weight"] > 0,
    "weight",
]

print("\nPositive training weight distribution:")

print(
    positive_weight.describe(
        percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]
    ).round(2)
)


# ---------------------------------------------------------
# 2. Missing-weight investigation
# ---------------------------------------------------------
missing_weight_train = train[
    train["weight"].isna()
].copy()

missing_weight_validation = validation[
    validation["weight"].isna()
].copy()


missing_weight_by_equipment_train = (
    missing_weight_train["equipment"]
    .value_counts()
    .rename_axis("equipment")
    .reset_index(name="missing_weight_rows")
)

missing_weight_by_equipment_train.to_csv(
    TABLE_DIR / "missing_weight_by_equipment_train.csv",
    index=False,
)


print("\n\nMISSING WEIGHT BY EQUIPMENT - TRAIN")
print("-" * 60)

print(
    missing_weight_by_equipment_train.to_string(
        index=False
    )
)


# Equipment-level typical weights
equipment_weight_summary = (
    train.assign(
        weight_abs=train["weight"].abs()
    )
    .groupby("equipment")
    .agg(
        rows=("load_id", "size"),
        valid_weight=("weight_abs", "count"),
        median_weight=("weight_abs", "median"),
        mean_weight=("weight_abs", "mean"),
        std_weight=("weight_abs", "std"),
    )
    .reset_index()
)

equipment_weight_summary.to_csv(
    TABLE_DIR / "equipment_weight_summary.csv",
    index=False,
)


print("\nTYPICAL ABSOLUTE WEIGHT BY EQUIPMENT")
print("-" * 60)

print(
    equipment_weight_summary.round(2).to_string(
        index=False
    )
)


# ---------------------------------------------------------
# 3. Missing market-index investigation
# ---------------------------------------------------------
train["month"] = train["date"].dt.to_period("M").astype(str)
validation["month"] = validation["date"].dt.to_period("M").astype(str)


missing_market_train = (
    train.assign(
        market_missing=train["market_index"].isna()
    )
    .groupby("month")
    .agg(
        total_rows=("load_id", "size"),
        missing_market_index=("market_missing", "sum"),
    )
    .reset_index()
)

missing_market_train["missing_pct"] = (
    missing_market_train["missing_market_index"]
    / missing_market_train["total_rows"]
    * 100
)

missing_market_train.to_csv(
    TABLE_DIR / "missing_market_index_by_month_train.csv",
    index=False,
)


print("\n\nMISSING MARKET INDEX BY MONTH - TRAIN")
print("-" * 60)

print(
    missing_market_train.round(2).to_string(
        index=False
    )
)


# ---------------------------------------------------------
# 4. Rate-per-mile analysis
# ---------------------------------------------------------
train["rate_per_mile"] = (
    train["posted_rate"]
    / train["distance"]
)


rpm_summary = train["rate_per_mile"].describe(
    percentiles=[
        0.001,
        0.005,
        0.01,
        0.05,
        0.25,
        0.50,
        0.75,
        0.95,
        0.99,
        0.995,
        0.999,
    ]
)

print("\n\nRATE PER MILE DETAILED SUMMARY")
print("-" * 60)

print(
    rpm_summary.round(3)
)


# ---------------------------------------------------------
# 5. IQR anomaly flags
# ---------------------------------------------------------

# Posted rate
rate_q1 = train["posted_rate"].quantile(0.25)
rate_q3 = train["posted_rate"].quantile(0.75)

rate_iqr = rate_q3 - rate_q1

rate_lower = rate_q1 - 1.5 * rate_iqr
rate_upper = rate_q3 + 1.5 * rate_iqr


# Rate per mile
rpm_q1 = train["rate_per_mile"].quantile(0.25)
rpm_q3 = train["rate_per_mile"].quantile(0.75)

rpm_iqr = rpm_q3 - rpm_q1

rpm_lower = rpm_q1 - 1.5 * rpm_iqr
rpm_upper = rpm_q3 + 1.5 * rpm_iqr


train["posted_rate_iqr_flag"] = (
    (train["posted_rate"] < rate_lower)
    | (train["posted_rate"] > rate_upper)
)

train["rpm_iqr_flag"] = (
    (train["rate_per_mile"] < rpm_lower)
    | (train["rate_per_mile"] > rpm_upper)
)


# ---------------------------------------------------------
# 6. Extreme target rows
# ---------------------------------------------------------
extreme_rate_rows = (
    train.loc[
        train["posted_rate_iqr_flag"],
        [
            "load_id",
            "pickup",
            "delivery",
            "distance",
            "equipment",
            "weight",
            "date",
            "market_index",
            "quote_signal",
            "posted_rate",
            "rate_per_mile",
        ],
    ]
    .sort_values(
        "posted_rate",
        ascending=False,
    )
)

extreme_rate_rows.to_csv(
    TABLE_DIR / "extreme_posted_rate_rows.csv",
    index=False,
)


# ---------------------------------------------------------
# 7. Extreme rate-per-mile rows
# ---------------------------------------------------------
extreme_rpm_rows = (
    train.loc[
        train["rpm_iqr_flag"],
        [
            "load_id",
            "pickup",
            "delivery",
            "distance",
            "equipment",
            "weight",
            "date",
            "market_index",
            "quote_signal",
            "posted_rate",
            "rate_per_mile",
        ],
    ]
    .sort_values(
        "rate_per_mile",
        ascending=False,
    )
)

extreme_rpm_rows.to_csv(
    TABLE_DIR / "extreme_rate_per_mile_rows.csv",
    index=False,
)


print("\n\nTOP 15 HIGHEST POSTED RATES")
print("-" * 60)

print(
    train[
        [
            "load_id",
            "distance",
            "equipment",
            "posted_rate",
            "rate_per_mile",
        ]
    ]
    .nlargest(
        15,
        "posted_rate",
    )
    .round(2)
    .to_string(index=False)
)


print("\n\nTOP 15 HIGHEST RATE PER MILE")
print("-" * 60)

print(
    train[
        [
            "load_id",
            "distance",
            "equipment",
            "posted_rate",
            "rate_per_mile",
        ]
    ]
    .nlargest(
        15,
        "rate_per_mile",
    )
    .round(2)
    .to_string(index=False)
)


print("\n\nTOP 15 LOWEST RATE PER MILE")
print("-" * 60)

print(
    train[
        [
            "load_id",
            "distance",
            "equipment",
            "posted_rate",
            "rate_per_mile",
        ]
    ]
    .nsmallest(
        15,
        "rate_per_mile",
    )
    .round(2)
    .to_string(index=False)
)


# ---------------------------------------------------------
# 8. Multiple anomaly flags
# ---------------------------------------------------------
train["negative_weight_flag"] = (
    train["weight"] < 0
)

train["missing_weight_flag"] = (
    train["weight"].isna()
)

train["missing_market_index_flag"] = (
    train["market_index"].isna()
)


flag_columns = [
    "posted_rate_iqr_flag",
    "rpm_iqr_flag",
    "negative_weight_flag",
    "missing_weight_flag",
    "missing_market_index_flag",
]


train["anomaly_flag_count"] = (
    train[flag_columns]
    .astype(int)
    .sum(axis=1)
)


multi_anomaly_rows = train.loc[
    train["anomaly_flag_count"] >= 2,
    [
        "load_id",
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "date",
        "market_index",
        "quote_signal",
        "posted_rate",
        "rate_per_mile",
        "posted_rate_iqr_flag",
        "rpm_iqr_flag",
        "negative_weight_flag",
        "missing_weight_flag",
        "missing_market_index_flag",
        "anomaly_flag_count",
    ],
].copy()


multi_anomaly_rows.to_csv(
    TABLE_DIR / "multiple_anomaly_rows.csv",
    index=False,
)


# ---------------------------------------------------------
# 9. Summary table
# ---------------------------------------------------------
summary = pd.DataFrame(
    [
        {
            "anomaly": "negative_weight_train",
            "count": int((train["weight"] < 0).sum()),
        },
        {
            "anomaly": "missing_weight_train",
            "count": int(train["weight"].isna().sum()),
        },
        {
            "anomaly": "missing_market_index_train",
            "count": int(train["market_index"].isna().sum()),
        },
        {
            "anomaly": "posted_rate_iqr_flag",
            "count": int(train["posted_rate_iqr_flag"].sum()),
        },
        {
            "anomaly": "rate_per_mile_iqr_flag",
            "count": int(train["rpm_iqr_flag"].sum()),
        },
        {
            "anomaly": "two_or_more_flags",
            "count": int(
                (train["anomaly_flag_count"] >= 2).sum()
            ),
        },
        {
            "anomaly": "three_or_more_flags",
            "count": int(
                (train["anomaly_flag_count"] >= 3).sum()
            ),
        },
    ]
)


summary.to_csv(
    TABLE_DIR / "anomaly_analysis_summary.csv",
    index=False,
)


print("\n\nANOMALY SUMMARY")
print("-" * 60)

print(
    summary.to_string(index=False)
)


# ---------------------------------------------------------
# Completion
# ---------------------------------------------------------
print("\nANOMALY ANALYSIS COMPLETE")

print(
    f"Detailed tables saved to: {TABLE_DIR}"
)