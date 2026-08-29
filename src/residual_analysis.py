from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
TABLE_DIR = ROOT / "reports" / "tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = RAW_DIR / "train-test.csv"


# ---------------------------------------------------------
# Load labeled development data
# ---------------------------------------------------------
train = pd.read_csv(TRAIN_FILE)

train["date"] = pd.to_datetime(
    train["date"],
    errors="coerce",
)


# ---------------------------------------------------------
# Baseline features
# ---------------------------------------------------------
# Keep this model intentionally simple.
# It is being used for anomaly investigation, not as the final model.
feature_columns = [
    "distance",
    "equipment",
]

X = train[feature_columns]
y = train["posted_rate"]


# ---------------------------------------------------------
# Simple preprocessing
# ---------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        (
            "equipment",
            OneHotEncoder(
                handle_unknown="ignore",
                drop=None,
            ),
            ["equipment"],
        ),
        (
            "distance",
            "passthrough",
            ["distance"],
        ),
    ]
)


# ---------------------------------------------------------
# Baseline regression model
# ---------------------------------------------------------
baseline_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression()),
    ]
)

baseline_model.fit(X, y)


# ---------------------------------------------------------
# Predictions and residuals
# ---------------------------------------------------------
train["baseline_predicted_rate"] = (
    baseline_model.predict(X)
)

train["residual"] = (
    train["posted_rate"]
    - train["baseline_predicted_rate"]
)

train["absolute_residual"] = (
    train["residual"].abs()
)

train["residual_pct"] = (
    train["residual"]
    / train["baseline_predicted_rate"]
)


# ---------------------------------------------------------
# Residual distribution
# ---------------------------------------------------------
print("\nRESIDUAL SUMMARY")
print("-" * 60)

print(
    train["residual"]
    .describe(
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
    .round(2)
)


print("\nABSOLUTE RESIDUAL SUMMARY")
print("-" * 60)

print(
    train["absolute_residual"]
    .describe(
        percentiles=[
            0.90,
            0.95,
            0.975,
            0.99,
            0.995,
            0.999,
        ]
    )
    .round(2)
)


# ---------------------------------------------------------
# Robust residual anomaly rule using MAD
# ---------------------------------------------------------
residual_median = train["residual"].median()

mad = np.median(
    np.abs(
        train["residual"]
        - residual_median
    )
)

# Robust modified z-score
train["residual_modified_z"] = (
    0.6745
    * (
        train["residual"]
        - residual_median
    )
    / mad
)

# A common robust anomaly threshold
train["residual_anomaly_flag"] = (
    train["residual_modified_z"].abs()
    > 3.5
)


# ---------------------------------------------------------
# Percent-error diagnostic
# ---------------------------------------------------------
train["extreme_relative_error_flag"] = (
    train["residual_pct"].abs()
    > 0.50
)


# ---------------------------------------------------------
# Combined diagnostic
# ---------------------------------------------------------
train["strong_target_anomaly_candidate"] = (
    train["residual_anomaly_flag"]
    & train["extreme_relative_error_flag"]
)


# ---------------------------------------------------------
# Save full residual diagnostics
# ---------------------------------------------------------
columns_to_save = [
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
    "baseline_predicted_rate",
    "residual",
    "absolute_residual",
    "residual_pct",
    "residual_modified_z",
    "residual_anomaly_flag",
    "extreme_relative_error_flag",
    "strong_target_anomaly_candidate",
]

train[
    columns_to_save
].to_csv(
    TABLE_DIR / "residual_diagnostics.csv",
    index=False,
)


# ---------------------------------------------------------
# Strong anomaly candidates
# ---------------------------------------------------------
strong_candidates = (
    train.loc[
        train["strong_target_anomaly_candidate"],
        columns_to_save,
    ]
    .sort_values(
        "absolute_residual",
        ascending=False,
    )
)

strong_candidates.to_csv(
    TABLE_DIR / "strong_target_anomaly_candidates.csv",
    index=False,
)


# ---------------------------------------------------------
# Highest positive residuals
# ---------------------------------------------------------
highest_positive = (
    train[
        columns_to_save
    ]
    .nlargest(
        20,
        "residual",
    )
)

highest_positive.to_csv(
    TABLE_DIR / "highest_positive_residuals.csv",
    index=False,
)


# ---------------------------------------------------------
# Lowest negative residuals
# ---------------------------------------------------------
lowest_negative = (
    train[
        columns_to_save
    ]
    .nsmallest(
        20,
        "residual",
    )
)

lowest_negative.to_csv(
    TABLE_DIR / "lowest_negative_residuals.csv",
    index=False,
)


# ---------------------------------------------------------
# Console output
# ---------------------------------------------------------
print("\nROBUST RESIDUAL ANOMALIES")
print("-" * 60)

print(
    "Residual modified-z > 3.5:",
    f"{train['residual_anomaly_flag'].sum():,}",
)

print(
    "|Residual %| > 50%:",
    f"{train['extreme_relative_error_flag'].sum():,}",
)

print(
    "Strong combined candidates:",
    f"{train['strong_target_anomaly_candidate'].sum():,}",
)


print("\nTOP 20 POSITIVE RESIDUALS")
print("-" * 60)

print(
    highest_positive[
        [
            "load_id",
            "distance",
            "equipment",
            "posted_rate",
            "baseline_predicted_rate",
            "residual",
            "residual_pct",
            "residual_modified_z",
        ]
    ]
    .round(2)
    .to_string(index=False)
)


print("\nTOP 20 NEGATIVE RESIDUALS")
print("-" * 60)

print(
    lowest_negative[
        [
            "load_id",
            "distance",
            "equipment",
            "posted_rate",
            "baseline_predicted_rate",
            "residual",
            "residual_pct",
            "residual_modified_z",
        ]
    ]
    .round(2)
    .to_string(index=False)
)


print("\nRESIDUAL ANALYSIS COMPLETE")
print(
    f"Tables saved to: {TABLE_DIR}"
)