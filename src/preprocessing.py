from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Preprocessing statistics
# ---------------------------------------------------------
@dataclass
class PreprocessingStats:
    """
    Statistics learned from training data only.

    These values must never be calculated using the final
    validation dataset or an internal holdout period.
    """

    weight_median_by_equipment: dict[str, float]
    global_weight_median: float
    market_index_median: float


# ---------------------------------------------------------
# Fit preprocessing statistics
# ---------------------------------------------------------
def fit_preprocessing(
    train_df: pd.DataFrame,
) -> PreprocessingStats:
    """
    Learn preprocessing statistics from a training dataset.

    Parameters
    ----------
    train_df:
        Training portion only.

    Returns
    -------
    PreprocessingStats
        Statistics required to preprocess training,
        holdout, and final validation data consistently.
    """

    df = train_df.copy()

    # -----------------------------------------------------
    # Weight
    # -----------------------------------------------------
    # Negative weights appear to be sign-entry errors.
    # Use their absolute magnitude when calculating typical
    # equipment-specific weights.
    weight_abs = df["weight"].abs()

    temp = pd.DataFrame(
        {
            "equipment": df["equipment"],
            "weight_abs": weight_abs,
        }
    )

    weight_median_by_equipment = (
        temp.groupby("equipment")["weight_abs"]
        .median()
        .dropna()
        .to_dict()
    )

    global_weight_median = float(
        weight_abs.median()
    )

    # -----------------------------------------------------
    # Market index
    # -----------------------------------------------------
    market_index_median = float(
        df["market_index"].median()
    )

    return PreprocessingStats(
        weight_median_by_equipment=weight_median_by_equipment,
        global_weight_median=global_weight_median,
        market_index_median=market_index_median,
    )


# ---------------------------------------------------------
# Apply preprocessing
# ---------------------------------------------------------
def apply_preprocessing(
    df: pd.DataFrame,
    stats: PreprocessingStats,
) -> pd.DataFrame:
    """
    Apply deterministic cleaning and training-derived
    imputations to a dataset.

    The function does not modify the input DataFrame.
    """

    result = df.copy()

    # -----------------------------------------------------
    # 1. Date parsing
    # -----------------------------------------------------
    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce",
    )

    if result["date"].isna().any():
        invalid_count = int(
            result["date"].isna().sum()
        )

        raise ValueError(
            f"Found {invalid_count} invalid or missing date values."
        )

    # -----------------------------------------------------
    # 2. Weight anomaly indicators
    # -----------------------------------------------------
    result["weight_missing_flag"] = (
        result["weight"].isna().astype(int)
    )

    result["weight_negative_flag"] = (
        (result["weight"] < 0)
        .fillna(False)
        .astype(int)
    )

    # -----------------------------------------------------
    # 3. Correct negative weights
    # -----------------------------------------------------
    # Negative values were shown during anomaly analysis
    # to have almost the same magnitude distribution as
    # valid positive weights.
    result["weight_clean"] = (
        result["weight"].abs()
    )

    # -----------------------------------------------------
    # 4. Impute missing weight by equipment
    # -----------------------------------------------------
    equipment_median = (
        result["equipment"]
        .map(
            stats.weight_median_by_equipment
        )
    )

    result["weight_clean"] = (
        result["weight_clean"]
        .fillna(equipment_median)
        .fillna(stats.global_weight_median)
    )

    # -----------------------------------------------------
    # 5. Market-index missing indicator
    # -----------------------------------------------------
    result["market_index_missing_flag"] = (
        result["market_index"]
        .isna()
        .astype(int)
    )

    # -----------------------------------------------------
    # 6. Market-index imputation
    # -----------------------------------------------------
    result["market_index_clean"] = (
        result["market_index"]
        .fillna(
            stats.market_index_median
        )
    )

    # -----------------------------------------------------
    # 7. Validation checks
    # -----------------------------------------------------
    if result["weight_clean"].isna().any():
        raise ValueError(
            "weight_clean still contains missing values."
        )

    if (result["weight_clean"] <= 0).any():
        raise ValueError(
            "weight_clean contains non-positive values."
        )

    if result["market_index_clean"].isna().any():
        raise ValueError(
            "market_index_clean still contains missing values."
        )

    if not np.isfinite(
        result["weight_clean"]
    ).all():
        raise ValueError(
            "weight_clean contains non-finite values."
        )

    if not np.isfinite(
        result["market_index_clean"]
    ).all():
        raise ValueError(
            "market_index_clean contains non-finite values."
        )

    return result


# ---------------------------------------------------------
# Preprocessing summary
# ---------------------------------------------------------
def preprocessing_summary(
    before: pd.DataFrame,
    after: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce a compact before/after preprocessing summary.
    """

    summary = pd.DataFrame(
        [
            {
                "check": "missing_weight",
                "before": int(
                    before["weight"].isna().sum()
                ),
                "after": int(
                    after["weight_clean"].isna().sum()
                ),
            },
            {
                "check": "negative_weight",
                "before": int(
                    (before["weight"] < 0).sum()
                ),
                "after": int(
                    (after["weight_clean"] < 0).sum()
                ),
            },
            {
                "check": "missing_market_index",
                "before": int(
                    before["market_index"]
                    .isna()
                    .sum()
                ),
                "after": int(
                    after["market_index_clean"]
                    .isna()
                    .sum()
                ),
            },
            {
                "check": "invalid_date",
                "before": int(
                    pd.to_datetime(
                        before["date"],
                        errors="coerce",
                    )
                    .isna()
                    .sum()
                ),
                "after": int(
                    after["date"]
                    .isna()
                    .sum()
                ),
            },
        ]
    )

    return summary