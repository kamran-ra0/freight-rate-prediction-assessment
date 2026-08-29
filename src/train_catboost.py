from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from features import create_features
from preprocessing import (
    apply_preprocessing,
    fit_preprocessing,
)
from split import (
    FORWARD_SPLITS,
    make_time_split,
)


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
TABLE_DIR = ROOT / "reports" / "tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = RAW_DIR / "train-test.csv"


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
TARGET = "posted_rate"

RANDOM_SEED = 42


FEATURE_CONFIGURATIONS = {
    "A_coordinates_market": {
        "include_city_categories": False,
        "include_market_signals": True,
    },
    "B_coordinates_no_market": {
        "include_city_categories": False,
        "include_market_signals": False,
    },
    "C_coordinates_city_market": {
        "include_city_categories": True,
        "include_market_signals": True,
    },
}


# ---------------------------------------------------------
# CatBoost factory
# ---------------------------------------------------------
def build_model() -> CatBoostRegressor:
    """
    Create the same initial CatBoost configuration for
    every fold and feature set.

    MAE is used as the training loss because the target
    analysis identified a small number of very large
    freight-rate anomalies. MAE is less sensitive to such
    extreme observations than squared-error loss.
    """

    return CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=8,
        loss_function="MAE",
        eval_metric="MAE",
        l2_leaf_reg=5.0,
        random_seed=RANDOM_SEED,
        allow_writing_files=False,
        verbose=False,
        thread_count=-1,
    )


# ---------------------------------------------------------
# Load labeled development data
# ---------------------------------------------------------
data = pd.read_csv(TRAIN_FILE)

print("\nCATBOOST FORWARD VALIDATION")
print("=" * 75)

print(f"Development rows: {len(data):,}")
print(f"Target: {TARGET}")


# ---------------------------------------------------------
# Storage
# ---------------------------------------------------------
result_rows = []
prediction_frames = []


# =========================================================
# Evaluate every feature configuration
# =========================================================
for feature_set_name, feature_options in FEATURE_CONFIGURATIONS.items():

    print("\n")
    print("=" * 75)
    print(f"FEATURE SET: {feature_set_name}")
    print("=" * 75)

    # -----------------------------------------------------
    # Evaluate all chronological folds
    # -----------------------------------------------------
    for split in FORWARD_SPLITS:

        print(f"\n{split.name}")
        print("-" * 60)

        # -------------------------------------------------
        # 1. Chronological split
        # -------------------------------------------------
        train_raw, holdout_raw = make_time_split(
            data,
            split,
        )

        print(
            f"Train rows:   {len(train_raw):,}"
        )

        print(
            f"Holdout rows: {len(holdout_raw):,}"
        )

        # -------------------------------------------------
        # 2. Fit preprocessing on TRAINING FOLD ONLY
        # -------------------------------------------------
        preprocessing_stats = fit_preprocessing(
            train_raw
        )

        train_clean = apply_preprocessing(
            train_raw,
            preprocessing_stats,
        )

        holdout_clean = apply_preprocessing(
            holdout_raw,
            preprocessing_stats,
        )

        # -------------------------------------------------
        # 3. Feature engineering
        # -------------------------------------------------
        X_train, categorical_features = create_features(
            train_clean,
            **feature_options,
        )

        X_holdout, holdout_categories = create_features(
            holdout_clean,
            **feature_options,
        )

        if categorical_features != holdout_categories:
            raise ValueError(
                "Categorical feature definitions differ "
                "between training and holdout sets."
            )

        # Enforce identical column order
        X_holdout = X_holdout.reindex(
            columns=X_train.columns
        )

        y_train = train_clean[
            TARGET
        ].astype(float)

        y_holdout = holdout_clean[
            TARGET
        ].astype(float)

        # -------------------------------------------------
        # 4. Train CatBoost
        # -------------------------------------------------
        model = build_model()

        model.fit(
            X_train,
            y_train,
            cat_features=categorical_features,
            eval_set=(
                X_holdout,
                y_holdout,
            ),
            early_stopping_rounds=100,
            verbose=False,
        )

        # -------------------------------------------------
        # 5. Predict holdout month
        # -------------------------------------------------
        predictions = model.predict(
            X_holdout
        )

        predictions = np.asarray(
            predictions,
            dtype=float,
        )

        # -------------------------------------------------
        # 6. Metrics
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
        # 7. Store fold metrics
        # -------------------------------------------------
        result_rows.append(
            {
                "feature_set": feature_set_name,
                "fold": split.name,
                "train_rows": len(train_raw),
                "holdout_rows": len(holdout_raw),
                "train_end": train_raw["date"].max(),
                "holdout_start": holdout_raw["date"].min(),
                "holdout_end": holdout_raw["date"].max(),
                "feature_count": len(X_train.columns),
                "categorical_feature_count": len(
                    categorical_features
                ),
                "best_iteration": best_iteration,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            }
        )

        # -------------------------------------------------
        # 8. Store predictions for later error analysis
        # -------------------------------------------------
        fold_predictions = pd.DataFrame(
            {
                "load_id": holdout_raw["load_id"].values,
                "date": holdout_raw["date"].values,
                "feature_set": feature_set_name,
                "fold": split.name,
                "actual_rate": y_holdout.values,
                "predicted_rate": predictions,
            }
        )

        fold_predictions["error"] = (
            fold_predictions["actual_rate"]
            - fold_predictions["predicted_rate"]
        )

        fold_predictions["absolute_error"] = (
            fold_predictions["error"].abs()
        )

        prediction_frames.append(
            fold_predictions
        )

        # -------------------------------------------------
        # Console output
        # -------------------------------------------------
        print(
            f"Features:       {len(X_train.columns)}"
        )

        print(
            f"Categoricals:   {categorical_features}"
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
# Save fold-level results
# =========================================================
results = pd.DataFrame(
    result_rows
)

results.to_csv(
    TABLE_DIR / "catboost_forward_validation.csv",
    index=False,
)


# ---------------------------------------------------------
# Aggregate model comparison
# ---------------------------------------------------------
summary = (
    results
    .groupby("feature_set")
    .agg(
        folds=("fold", "count"),
        mean_MAE=("MAE", "mean"),
        std_MAE=("MAE", "std"),
        mean_RMSE=("RMSE", "mean"),
        mean_R2=("R2", "mean"),
        mean_best_iteration=(
            "best_iteration",
            "mean",
        ),
    )
    .reset_index()
    .sort_values(
        "mean_MAE"
    )
)


summary.to_csv(
    TABLE_DIR / "catboost_feature_set_comparison.csv",
    index=False,
)


# ---------------------------------------------------------
# Save all forward-validation predictions
# ---------------------------------------------------------
all_predictions = pd.concat(
    prediction_frames,
    ignore_index=True,
)

all_predictions.to_csv(
    TABLE_DIR / "catboost_forward_predictions.csv",
    index=False,
)


# =========================================================
# Print final comparison
# =========================================================
print("\n")
print("=" * 75)
print("CATBOOST FEATURE-SET COMPARISON")
print("=" * 75)

display_summary = summary.copy()

for column in [
    "mean_MAE",
    "std_MAE",
    "mean_RMSE",
]:
    display_summary[column] = (
        display_summary[column]
        .round(2)
    )

display_summary["mean_R2"] = (
    display_summary["mean_R2"]
    .round(4)
)

display_summary["mean_best_iteration"] = (
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


# ---------------------------------------------------------
# Best feature set
# ---------------------------------------------------------
best_feature_set = summary.iloc[0]

print("\nBEST INITIAL FEATURE CONFIGURATION")
print("-" * 75)

print(
    f"Feature set: "
    f"{best_feature_set['feature_set']}"
)

print(
    f"Mean forward-validation MAE: "
    f"${best_feature_set['mean_MAE']:,.2f}"
)

print(
    f"Mean forward-validation RMSE: "
    f"${best_feature_set['mean_RMSE']:,.2f}"
)

print(
    f"Mean forward-validation R2: "
    f"{best_feature_set['mean_R2']:.4f}"
)


print("\nResults saved to:")

print(
    " -",
    TABLE_DIR / "catboost_forward_validation.csv",
)

print(
    " -",
    TABLE_DIR / "catboost_feature_set_comparison.csv",
)

print(
    " -",
    TABLE_DIR / "catboost_forward_predictions.csv",
)

print("\nCATBOOST FORWARD VALIDATION COMPLETE")