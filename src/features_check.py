from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import (
    apply_preprocessing,
    fit_preprocessing,
)
from features import create_features


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
TABLE_DIR = ROOT / "reports" / "tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = RAW_DIR / "train-test.csv"


# ---------------------------------------------------------
# Load raw labeled data
# ---------------------------------------------------------
train_raw = pd.read_csv(TRAIN_FILE)

print("\nRAW TRAINING DATA")
print("-" * 60)

print(f"Rows:    {len(train_raw):,}")
print(f"Columns: {len(train_raw.columns)}")


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------
stats = fit_preprocessing(
    train_raw
)

train_clean = apply_preprocessing(
    train_raw,
    stats,
)


# ---------------------------------------------------------
# Feature set A
# Coordinates + engineered geography
# Market signals included
# City names excluded
# ---------------------------------------------------------
X_base, categorical_base = create_features(
    train_clean,
    include_city_categories=False,
    include_market_signals=True,
)


print("\nFEATURE SET A")
print("-" * 60)

print(
    "City categories included: NO"
)

print(
    "Market signals included: YES"
)

print(
    f"Rows: {len(X_base):,}"
)

print(
    f"Features: {len(X_base.columns)}"
)

print(
    "Categorical features:",
    categorical_base,
)


# ---------------------------------------------------------
# Feature set B
# Market signals excluded
# ---------------------------------------------------------
X_no_market, categorical_no_market = create_features(
    train_clean,
    include_city_categories=False,
    include_market_signals=False,
)


print("\nFEATURE SET B")
print("-" * 60)

print(
    "City categories included: NO"
)

print(
    "Market signals included: NO"
)

print(
    f"Rows: {len(X_no_market):,}"
)

print(
    f"Features: {len(X_no_market.columns)}"
)

print(
    "Categorical features:",
    categorical_no_market,
)


# ---------------------------------------------------------
# Feature set C
# Raw city categories included
# ---------------------------------------------------------
X_with_city, categorical_with_city = create_features(
    train_clean,
    include_city_categories=True,
    include_market_signals=True,
)


print("\nFEATURE SET C")
print("-" * 60)

print(
    "City categories included: YES"
)

print(
    "Market signals included: YES"
)

print(
    f"Rows: {len(X_with_city):,}"
)

print(
    f"Features: {len(X_with_city.columns)}"
)

print(
    "Categorical features:",
    categorical_with_city,
)


# ---------------------------------------------------------
# Integrity checks
# ---------------------------------------------------------
def validate_feature_set(
    X: pd.DataFrame,
    categorical_features: list[str],
    name: str,
) -> None:
    """
    Confirm a feature matrix is suitable for modeling.
    """

    if len(X) != len(train_raw):
        raise ValueError(
            f"{name}: row count changed during feature engineering."
        )

    if X.columns.duplicated().any():
        raise ValueError(
            f"{name}: duplicate feature columns detected."
        )

    numeric_columns = [
        column
        for column in X.columns
        if column not in categorical_features
    ]

    if X[numeric_columns].isna().any().any():
        bad_columns = (
            X[numeric_columns]
            .columns[
                X[numeric_columns]
                .isna()
                .any()
            ]
            .tolist()
        )

        raise ValueError(
            f"{name}: missing numerical values found in "
            + ", ".join(bad_columns)
        )

    numeric_array = X[
        numeric_columns
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        numeric_array
    ).all():
        raise ValueError(
            f"{name}: non-finite numerical values detected."
        )

    for column in categorical_features:
        if X[column].isna().any():
            raise ValueError(
                f"{name}: missing categorical values in {column}."
            )

    print(
        f"{name}: integrity check PASSED"
    )


print("\nFEATURE INTEGRITY CHECKS")
print("-" * 60)

validate_feature_set(
    X_base,
    categorical_base,
    "Feature Set A",
)

validate_feature_set(
    X_no_market,
    categorical_no_market,
    "Feature Set B",
)

validate_feature_set(
    X_with_city,
    categorical_with_city,
    "Feature Set C",
)


# ---------------------------------------------------------
# Save feature inventory
# ---------------------------------------------------------
inventory_rows = []

for feature_set_name, X, categorical in [
    (
        "A_coordinates_market",
        X_base,
        categorical_base,
    ),
    (
        "B_coordinates_no_market",
        X_no_market,
        categorical_no_market,
    ),
    (
        "C_coordinates_city_market",
        X_with_city,
        categorical_with_city,
    ),
]:
    for column in X.columns:

        inventory_rows.append(
            {
                "feature_set": feature_set_name,
                "feature": column,
                "dtype": str(X[column].dtype),
                "categorical": (
                    column in categorical
                ),
            }
        )


inventory = pd.DataFrame(
    inventory_rows
)

inventory.to_csv(
    TABLE_DIR / "feature_inventory.csv",
    index=False,
)


# ---------------------------------------------------------
# Print Feature Set A names
# ---------------------------------------------------------
print("\nFEATURE SET A COLUMNS")
print("-" * 60)

for number, column in enumerate(
    X_base.columns,
    start=1,
):
    category_marker = (
        " [categorical]"
        if column in categorical_base
        else ""
    )

    print(
        f"{number:>2}. "
        f"{column}"
        f"{category_marker}"
    )


# ---------------------------------------------------------
# Basic feature statistics
# ---------------------------------------------------------
selected_numeric = [
    "distance",
    "weight_clean",
    "haversine_distance",
    "detour_ratio",
    "distance_x_weight",
]

feature_statistics = (
    X_base[selected_numeric]
    .describe()
    .T
)

feature_statistics.to_csv(
    TABLE_DIR / "engineered_feature_summary.csv"
)


# ---------------------------------------------------------
# Completion
# ---------------------------------------------------------
print("\nFEATURE ENGINEERING CHECK PASSED")

print(
    "\nFeature inventory saved to:",
    TABLE_DIR / "feature_inventory.csv",
)

print(
    "Feature statistics saved to:",
    TABLE_DIR / "engineered_feature_summary.csv",
)