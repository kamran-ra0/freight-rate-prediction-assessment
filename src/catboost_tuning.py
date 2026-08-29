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
# Candidate CatBoost configurations
# =========================================================
PARAMETER_SETS = [
    {
        "name": "T1_baseline",
        "depth": 8,
        "learning_rate": 0.03,
        "l2_leaf_reg": 5.0,
        "random_strength": 1.0,
    },
    {
        "name": "T2_depth6",
        "depth": 6,
        "learning_rate": 0.04,
        "l2_leaf_reg": 5.0,
        "random_strength": 1.0,
    },
    {
        "name": "T3_depth7",
        "depth": 7,
        "learning_rate": 0.035,
        "l2_leaf_reg": 5.0,
        "random_strength": 1.0,
    },
    {
        "name": "T4_depth8_regularized",
        "depth": 8,
        "learning_rate": 0.025,
        "l2_leaf_reg": 10.0,
        "random_strength": 1.0,
    },
    {
        "name": "T5_depth9_regularized",
        "depth": 9,
        "learning_rate": 0.025,
        "l2_leaf_reg": 10.0,
        "random_strength": 1.0,
    },
    {
        "name": "T6_depth8_low_randomness",
        "depth": 8,
        "learning_rate": 0.03,
        "l2_leaf_reg": 7.0,
        "random_strength": 0.5,
    },
]


# =========================================================
# Model builder
# =========================================================
def build_model(parameters):

    return CatBoostRegressor(
        iterations=2500,
        learning_rate=parameters["learning_rate"],
        depth=parameters["depth"],
        l2_leaf_reg=parameters["l2_leaf_reg"],
        random_strength=parameters["random_strength"],

        loss_function="MAE",
        eval_metric="MAE",

        random_seed=RANDOM_SEED,

        allow_writing_files=False,
        verbose=False,

        # Explicitly retain the model state corresponding
        # to the best validation iteration.
        use_best_model=True,

        thread_count=-1,
    )


# =========================================================
# Load development data
# =========================================================
data = pd.read_csv(TRAIN_FILE)


print("\nCATBOOST HYPERPARAMETER TUNING")
print("=" * 78)

print(f"Rows: {len(data):,}")
print("Feature set: B_coordinates_no_market")
print(f"Configurations: {len(PARAMETER_SETS)}")
print(f"Chronological folds: {len(FORWARD_SPLITS)}")


# =========================================================
# Results
# =========================================================
result_rows = []


# =========================================================
# Parameter search
# =========================================================
for parameters in PARAMETER_SETS:

    print("\n")
    print("=" * 78)
    print(f"CONFIGURATION: {parameters['name']}")
    print("=" * 78)

    print(
        f"depth={parameters['depth']}, "
        f"learning_rate={parameters['learning_rate']}, "
        f"l2_leaf_reg={parameters['l2_leaf_reg']}, "
        f"random_strength={parameters['random_strength']}"
    )

    # -----------------------------------------------------
    # Chronological folds
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

        # -------------------------------------------------
        # 2. Leakage-safe preprocessing
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
        # 3. Winning feature configuration
        # -------------------------------------------------
        X_train, categorical_features = create_features(
            train_clean,
            **FEATURE_OPTIONS,
        )

        X_holdout, holdout_categories = create_features(
            holdout_clean,
            **FEATURE_OPTIONS,
        )

        if categorical_features != holdout_categories:
            raise ValueError(
                "Categorical feature definitions differ "
                "between train and holdout."
            )

        X_holdout = X_holdout.reindex(
            columns=X_train.columns
        )

        y_train = (
            train_clean[TARGET]
            .astype(float)
        )

        y_holdout = (
            holdout_clean[TARGET]
            .astype(float)
        )

        # -------------------------------------------------
        # 4. Build and train
        # -------------------------------------------------
        model = build_model(
            parameters
        )

        model.fit(
            X_train,
            y_train,

            cat_features=categorical_features,

            eval_set=(
                X_holdout,
                y_holdout,
            ),

            early_stopping_rounds=150,

            verbose=False,
        )

        # -------------------------------------------------
        # 5. Prediction
        # -------------------------------------------------
        predictions = np.asarray(
            model.predict(
                X_holdout
            ),
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
        # 7. Store
        # -------------------------------------------------
        result_rows.append(
            {
                "configuration": parameters["name"],
                "fold": split.name,

                "depth": parameters["depth"],
                "learning_rate": parameters[
                    "learning_rate"
                ],
                "l2_leaf_reg": parameters[
                    "l2_leaf_reg"
                ],
                "random_strength": parameters[
                    "random_strength"
                ],

                "train_rows": len(train_raw),
                "holdout_rows": len(holdout_raw),

                "best_iteration": best_iteration,

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
# Fold-level results
# =========================================================
results = pd.DataFrame(
    result_rows
)

results.to_csv(
    TABLE_DIR
    / "catboost_tuning_fold_results.csv",
    index=False,
)


# =========================================================
# Aggregate comparison
# =========================================================
summary = (
    results
    .groupby(
        [
            "configuration",
            "depth",
            "learning_rate",
            "l2_leaf_reg",
            "random_strength",
        ],
        as_index=False,
    )
    .agg(
        mean_MAE=("MAE", "mean"),
        std_MAE=("MAE", "std"),
        worst_fold_MAE=("MAE", "max"),

        mean_RMSE=("RMSE", "mean"),
        mean_R2=("R2", "mean"),

        mean_best_iteration=(
            "best_iteration",
            "mean",
        ),
    )
    .sort_values(
        [
            "mean_MAE",
            "worst_fold_MAE",
        ]
    )
    .reset_index(drop=True)
)


summary.to_csv(
    TABLE_DIR
    / "catboost_tuning_summary.csv",
    index=False,
)


# =========================================================
# Console summary
# =========================================================
print("\n")
print("=" * 78)
print("CATBOOST TUNING SUMMARY")
print("=" * 78)

display_summary = summary.copy()

for column in [
    "mean_MAE",
    "std_MAE",
    "worst_fold_MAE",
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


# =========================================================
# Best configuration
# =========================================================
best = summary.iloc[0]


print("\nBEST CATBOOST CONFIGURATION")
print("-" * 78)

print(
    f"Configuration:      "
    f"{best['configuration']}"
)

print(
    f"Depth:              "
    f"{int(best['depth'])}"
)

print(
    f"Learning rate:      "
    f"{best['learning_rate']}"
)

print(
    f"L2 leaf reg:        "
    f"{best['l2_leaf_reg']}"
)

print(
    f"Random strength:    "
    f"{best['random_strength']}"
)

print(
    f"Mean MAE:           "
    f"${best['mean_MAE']:,.2f}"
)

print(
    f"Worst-fold MAE:     "
    f"${best['worst_fold_MAE']:,.2f}"
)

print(
    f"Mean RMSE:          "
    f"${best['mean_RMSE']:,.2f}"
)

print(
    f"Mean R2:            "
    f"{best['mean_R2']:.4f}"
)

print(
    f"Mean best iteration:"
    f" {best['mean_best_iteration']:.1f}"
)


print("\nResults saved to:")

print(
    " -",
    TABLE_DIR
    / "catboost_tuning_fold_results.csv",
)

print(
    " -",
    TABLE_DIR
    / "catboost_tuning_summary.csv",
)


print("\nCATBOOST TUNING COMPLETE")