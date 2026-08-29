from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import OneHotEncoder

from features import create_features
from preprocessing import (
    apply_preprocessing,
    fit_preprocessing,
)
from split import (
    FORWARD_SPLITS,
    make_time_split,
)


# =========================================================
# Paths
# =========================================================
ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
TABLE_DIR = ROOT / "reports" / "tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = RAW_DIR / "train-test.csv"

TARGET = "posted_rate"
RANDOM_SEED = 42


# =========================================================
# Winning feature configuration
# =========================================================
FEATURE_OPTIONS = {
    "include_city_categories": False,
    "include_market_signals": False,
}


# =========================================================
# Tuned CatBoost configuration
# =========================================================
def build_model():

    return CatBoostRegressor(
        iterations=2500,
        depth=8,
        learning_rate=0.025,
        l2_leaf_reg=10.0,
        random_strength=1.0,

        loss_function="MAE",
        eval_metric="MAE",

        random_seed=RANDOM_SEED,
        allow_writing_files=False,
        use_best_model=True,
        verbose=False,
        thread_count=-1,
    )


# =========================================================
# Detect strong target anomalies
# =========================================================
def detect_strong_anomalies(train_df):

    """
    Detect suspicious target values using residuals from
    a simple distance + equipment linear baseline.

    IMPORTANT:
    This function is applied only to the training fold.
    """

    baseline_features = train_df[
        [
            "distance",
            "equipment",
        ]
    ].copy()

    target = train_df[
        TARGET
    ].astype(float)


    transformer = ColumnTransformer(
        transformers=[
            (
                "equipment",
                OneHotEncoder(
                    handle_unknown="ignore"
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


    X_baseline = transformer.fit_transform(
        baseline_features
    )


    baseline_model = LinearRegression()

    baseline_model.fit(
        X_baseline,
        target,
    )


    baseline_prediction = baseline_model.predict(
        X_baseline
    )


    residual = (
        target.to_numpy()
        - baseline_prediction
    )


    # -----------------------------------------------------
    # Robust modified Z-score
    # -----------------------------------------------------
    residual_median = np.median(
        residual
    )

    absolute_deviation = np.abs(
        residual - residual_median
    )

    mad = np.median(
        absolute_deviation
    )


    if mad == 0:

        residual_std = np.std(
            residual
        )

        if residual_std == 0:
            modified_z = np.zeros_like(
                residual
            )

        else:
            modified_z = (
                residual
                - residual_median
            ) / residual_std

    else:

        modified_z = (
            0.6745
            * (
                residual
                - residual_median
            )
            / mad
        )


    # -----------------------------------------------------
    # Relative residual
    # -----------------------------------------------------
    relative_residual = (
        np.abs(residual)
        / np.maximum(
            np.abs(
                target.to_numpy()
            ),
            1.0,
        )
    )


    # Strong anomaly requires BOTH:
    # 1. extreme robust residual
    # 2. prediction error > 50% of actual rate
    strong_anomaly = (
        (np.abs(modified_z) > 3.5)
        &
        (relative_residual > 0.50)
    )


    return strong_anomaly


# =========================================================
# Strategies
# =========================================================
STRATEGIES = [
    "keep_all",
    "remove_strong_anomalies",
    "cap_extreme_targets",
]


# =========================================================
# Load data
# =========================================================
data = pd.read_csv(
    TRAIN_FILE
)


print("\nCATBOOST TARGET-ANOMALY TEST")
print("=" * 78)

print(
    f"Development rows: {len(data):,}"
)

print(
    "Model: Tuned T4 CatBoost"
)

print(
    "Feature set: B_coordinates_no_market"
)


# =========================================================
# Storage
# =========================================================
result_rows = []


# =========================================================
# Strategy comparison
# =========================================================
for strategy in STRATEGIES:

    print("\n")
    print("=" * 78)
    print(
        f"STRATEGY: {strategy}"
    )
    print("=" * 78)


    for split in FORWARD_SPLITS:

        print(
            f"\n{split.name}"
        )
        print("-" * 60)


        # -------------------------------------------------
        # 1. Chronological split
        # -------------------------------------------------
        train_raw, holdout_raw = make_time_split(
            data,
            split,
        )


        original_train_rows = len(
            train_raw
        )


        # =================================================
        # Strategy 1: Keep everything
        # =================================================
        if strategy == "keep_all":

            model_train_raw = (
                train_raw.copy()
            )

            removed_rows = 0

            capped_rows = 0


        # =================================================
        # Strategy 2: Remove strong anomalies
        # =================================================
        elif strategy == "remove_strong_anomalies":

            anomaly_mask = detect_strong_anomalies(
                train_raw
            )

            removed_rows = int(
                anomaly_mask.sum()
            )

            capped_rows = 0

            model_train_raw = (
                train_raw
                .loc[
                    ~anomaly_mask
                ]
                .copy()
            )


        # =================================================
        # Strategy 3: Cap extreme targets
        # =================================================
        elif strategy == "cap_extreme_targets":

            model_train_raw = (
                train_raw.copy()
            )

            removed_rows = 0


            lower_limit = (
                model_train_raw[
                    TARGET
                ]
                .quantile(0.005)
            )

            upper_limit = (
                model_train_raw[
                    TARGET
                ]
                .quantile(0.995)
            )


            original_target = (
                model_train_raw[
                    TARGET
                ].copy()
            )


            model_train_raw[
                TARGET
            ] = (
                model_train_raw[
                    TARGET
                ]
                .clip(
                    lower=lower_limit,
                    upper=upper_limit,
                )
            )


            capped_rows = int(
                (
                    original_target
                    != model_train_raw[
                        TARGET
                    ]
                )
                .sum()
            )


        else:

            raise ValueError(
                f"Unknown strategy: {strategy}"
            )


        print(
            f"Original train rows: "
            f"{original_train_rows:,}"
        )

        print(
            f"Rows used:           "
            f"{len(model_train_raw):,}"
        )

        print(
            f"Rows removed:        "
            f"{removed_rows:,}"
        )

        print(
            f"Targets capped:      "
            f"{capped_rows:,}"
        )


        # -------------------------------------------------
        # 2. Fit preprocessing only on resulting
        #    training fold
        # -------------------------------------------------
        preprocessing_stats = (
            fit_preprocessing(
                model_train_raw
            )
        )


        train_clean = apply_preprocessing(
            model_train_raw,
            preprocessing_stats,
        )


        holdout_clean = apply_preprocessing(
            holdout_raw,
            preprocessing_stats,
        )


        # -------------------------------------------------
        # 3. Feature engineering
        # -------------------------------------------------
        X_train, categorical_features = (
            create_features(
                train_clean,
                **FEATURE_OPTIONS,
            )
        )


        X_holdout, holdout_categories = (
            create_features(
                holdout_clean,
                **FEATURE_OPTIONS,
            )
        )


        if (
            categorical_features
            != holdout_categories
        ):
            raise ValueError(
                "Categorical feature mismatch."
            )


        X_holdout = X_holdout.reindex(
            columns=X_train.columns
        )


        y_train = (
            train_clean[
                TARGET
            ]
            .astype(float)
        )


        # IMPORTANT:
        # Holdout target remains completely untouched.
        y_holdout = (
            holdout_clean[
                TARGET
            ]
            .astype(float)
        )


        # -------------------------------------------------
        # 4. Train tuned CatBoost
        # -------------------------------------------------
        model = build_model()


        model.fit(
            X_train,
            y_train,

            cat_features=(
                categorical_features
            ),

            eval_set=(
                X_holdout,
                y_holdout,
            ),

            early_stopping_rounds=150,

            verbose=False,
        )


        # -------------------------------------------------
        # 5. Predictions
        # -------------------------------------------------
        predictions = np.asarray(
            model.predict(
                X_holdout
            ),
            dtype=float,
        )


        # -------------------------------------------------
        # 6. Evaluation
        # -------------------------------------------------
        mae = mean_absolute_error(
            y_holdout,
            predictions,
        )


        rmse = np.sqrt(
            mean_squared_error(
                y_holdout,
                predictions,
            )
        )


        r2 = r2_score(
            y_holdout,
            predictions,
        )


        best_iteration = (
            model.get_best_iteration()
        )


        # -------------------------------------------------
        # Store
        # -------------------------------------------------
        result_rows.append(
            {
                "strategy": strategy,

                "fold": split.name,

                "original_train_rows":
                    original_train_rows,

                "model_train_rows":
                    len(model_train_raw),

                "removed_rows":
                    removed_rows,

                "capped_rows":
                    capped_rows,

                "best_iteration":
                    best_iteration,

                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            }
        )


        print(
            f"Best iteration: {best_iteration}"
        )

        print(
            f"MAE:            ${mae:,.2f}"
        )

        print(
            f"RMSE:           ${rmse:,.2f}"
        )

        print(
            f"R2:             {r2:.4f}"
        )


# =========================================================
# Save fold results
# =========================================================
results = pd.DataFrame(
    result_rows
)


results.to_csv(
    TABLE_DIR
    / "catboost_anomaly_fold_results.csv",
    index=False,
)


# =========================================================
# Aggregate comparison
# =========================================================
summary = (
    results
    .groupby(
        "strategy",
        as_index=False,
    )
    .agg(
        mean_MAE=(
            "MAE",
            "mean",
        ),

        std_MAE=(
            "MAE",
            "std",
        ),

        worst_fold_MAE=(
            "MAE",
            "max",
        ),

        mean_RMSE=(
            "RMSE",
            "mean",
        ),

        mean_R2=(
            "R2",
            "mean",
        ),

        mean_best_iteration=(
            "best_iteration",
            "mean",
        ),

        total_removed=(
            "removed_rows",
            "sum",
        ),

        total_capped=(
            "capped_rows",
            "sum",
        ),
    )
    .sort_values(
        [
            "mean_MAE",
            "worst_fold_MAE",
        ]
    )
    .reset_index(
        drop=True
    )
)


summary.to_csv(
    TABLE_DIR
    / "catboost_anomaly_strategy_comparison.csv",
    index=False,
)


# =========================================================
# Display summary
# =========================================================
print("\n")
print("=" * 78)
print("TARGET-ANOMALY STRATEGY COMPARISON")
print("=" * 78)


display_summary = (
    summary.copy()
)


for column in [
    "mean_MAE",
    "std_MAE",
    "worst_fold_MAE",
    "mean_RMSE",
]:

    display_summary[
        column
    ] = (
        display_summary[
            column
        ]
        .round(2)
    )


display_summary[
    "mean_R2"
] = (
    display_summary[
        "mean_R2"
    ]
    .round(4)
)


display_summary[
    "mean_best_iteration"
] = (
    display_summary[
        "mean_best_iteration"
    ]
    .round(1)
)


print(
    display_summary.to_string(
        index=False
    )
)


# =========================================================
# Best strategy
# =========================================================
best = summary.iloc[0]


print("\nBEST TARGET-HANDLING STRATEGY")
print("-" * 78)


print(
    f"Strategy:       "
    f"{best['strategy']}"
)

print(
    f"Mean MAE:       "
    f"${best['mean_MAE']:,.2f}"
)

print(
    f"Worst-fold MAE: "
    f"${best['worst_fold_MAE']:,.2f}"
)

print(
    f"Mean RMSE:      "
    f"${best['mean_RMSE']:,.2f}"
)

print(
    f"Mean R2:        "
    f"{best['mean_R2']:.4f}"
)


print("\nResults saved to:")

print(
    " -",
    TABLE_DIR
    / "catboost_anomaly_fold_results.csv"
)

print(
    " -",
    TABLE_DIR
    / "catboost_anomaly_strategy_comparison.csv"
)


print(
    "\nCATBOOST TARGET-ANOMALY TEST COMPLETE"
)