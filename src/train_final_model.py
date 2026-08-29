from pathlib import Path
from dataclasses import asdict
import json

import pandas as pd
from catboost import CatBoostRegressor

from features import create_features
from preprocessing import (
    apply_preprocessing,
    fit_preprocessing,
)


# =========================================================
# Paths
# =========================================================
ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
TABLE_DIR = ROOT / "reports" / "tables"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = RAW_DIR / "train-test.csv"

MODEL_FILE = (
    MODEL_DIR / "final_catboost_model.cbm"
)

PREPROCESSING_FILE = (
    MODEL_DIR / "preprocessing_stats.json"
)

FEATURE_METADATA_FILE = (
    MODEL_DIR / "feature_metadata.json"
)

MODEL_CONFIG_FILE = (
    MODEL_DIR / "final_model_config.json"
)

TRAINING_SUMMARY_FILE = (
    TABLE_DIR / "final_model_training_summary.csv"
)

TUNING_RESULTS_FILE = (
    TABLE_DIR / "catboost_tuning_fold_results.csv"
)


# =========================================================
# Fixed selected configuration
# =========================================================
TARGET = "posted_rate"
RANDOM_SEED = 42

BEST_CONFIGURATION = (
    "T4_depth8_regularized"
)

FEATURE_SET_NAME = (
    "B_coordinates_no_market"
)

FEATURE_OPTIONS = {
    "include_city_categories": False,
    "include_market_signals": False,
}


# =========================================================
# Determine final number of boosting iterations
# =========================================================
def determine_final_iterations():

    """
    Use the mean best iteration observed across the
    chronological validation folds.

    CatBoost get_best_iteration() is zero-based, so one
    tree is added when converting the index into the
    final number of boosting iterations.
    """

    if not TUNING_RESULTS_FILE.exists():

        print(
            "Tuning results file not found."
        )

        print(
            "Using fallback iterations = 1557"
        )

        return 1557


    tuning_results = pd.read_csv(
        TUNING_RESULTS_FILE
    )


    selected = tuning_results[
        tuning_results["configuration"]
        == BEST_CONFIGURATION
    ]


    if selected.empty:

        raise ValueError(
            f"{BEST_CONFIGURATION} was not found "
            "in the tuning results."
        )


    mean_best_iteration = (
        selected["best_iteration"].mean()
    )


    final_iterations = (
        int(round(mean_best_iteration))
        + 1
    )


    print(
        f"Mean validation best iteration: "
        f"{mean_best_iteration:.2f}"
    )

    print(
        f"Final boosting iterations: "
        f"{final_iterations}"
    )


    return final_iterations


# =========================================================
# Load all labeled development data
# =========================================================
print("\nFINAL CATBOOST TRAINING")
print("=" * 75)


data = pd.read_csv(
    TRAIN_FILE
)


print(
    f"Training rows: {len(data):,}"
)

print(
    f"Original columns: "
    f"{len(data.columns)}"
)


# =========================================================
# Validate target
# =========================================================
if TARGET not in data.columns:

    raise ValueError(
        f"Target column '{TARGET}' not found."
    )


if data[TARGET].isna().any():

    raise ValueError(
        "Missing target values detected."
    )


# =========================================================
# Fit preprocessing on ALL labeled training data
# =========================================================
print(
    "\nFitting preprocessing statistics "
    "on all labeled training data..."
)


preprocessing_stats = (
    fit_preprocessing(
        data
    )
)


train_clean = apply_preprocessing(
    data,
    preprocessing_stats,
)


# =========================================================
# Feature engineering
# =========================================================
print(
    "Creating selected Feature Set B..."
)


X_train, categorical_features = (
    create_features(
        train_clean,
        **FEATURE_OPTIONS,
    )
)


y_train = (
    train_clean[TARGET]
    .astype(float)
)


print(
    f"Final feature count: "
    f"{len(X_train.columns)}"
)

print(
    f"Categorical features: "
    f"{categorical_features}"
)


# =========================================================
# Final boosting iterations
# =========================================================
final_iterations = (
    determine_final_iterations()
)


# =========================================================
# Build final CatBoost model
# =========================================================
model = CatBoostRegressor(

    iterations=final_iterations,

    depth=8,

    learning_rate=0.025,

    l2_leaf_reg=10.0,

    random_strength=1.0,

    loss_function="MAE",

    random_seed=RANDOM_SEED,

    # No validation set exists here because all labeled
    # observations are now being used for final training.
    use_best_model=False,

    allow_writing_files=False,

    verbose=100,

    thread_count=-1,
)


# =========================================================
# Train final model
# =========================================================
print(
    "\nTraining final CatBoost model..."
)


model.fit(
    X_train,
    y_train,

    cat_features=(
        categorical_features
    ),
)


print(
    "\nFinal model training complete."
)


# =========================================================
# Save CatBoost model
# =========================================================
model.save_model(
    MODEL_FILE
)


# =========================================================
# Save preprocessing statistics
# =========================================================
stats_dict = asdict(
    preprocessing_stats
)


# Ensure JSON-compatible numeric values
stats_dict[
    "weight_median_by_equipment"
] = {
    str(key): float(value)
    for key, value in stats_dict[
        "weight_median_by_equipment"
    ].items()
}

stats_dict[
    "global_weight_median"
] = float(
    stats_dict[
        "global_weight_median"
    ]
)

stats_dict[
    "market_index_median"
] = float(
    stats_dict[
        "market_index_median"
    ]
)


with open(
    PREPROCESSING_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        stats_dict,
        file,
        indent=4,
    )


# =========================================================
# Save exact feature metadata
# =========================================================
feature_metadata = {

    "feature_set":
        FEATURE_SET_NAME,

    "feature_options":
        FEATURE_OPTIONS,

    "feature_columns":
        X_train.columns.tolist(),

    "categorical_features":
        categorical_features,
}


with open(
    FEATURE_METADATA_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        feature_metadata,
        file,
        indent=4,
    )


# =========================================================
# Save model configuration
# =========================================================
model_configuration = {

    "model_type":
        "CatBoostRegressor",

    "configuration":
        BEST_CONFIGURATION,

    "iterations":
        final_iterations,

    "depth":
        8,

    "learning_rate":
        0.025,

    "l2_leaf_reg":
        10.0,

    "random_strength":
        1.0,

    "loss_function":
        "MAE",

    "random_seed":
        RANDOM_SEED,

    "target_handling":
        "keep_all",

    "target":
        TARGET,

    "feature_set":
        FEATURE_SET_NAME,
}


with open(
    MODEL_CONFIG_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        model_configuration,
        file,
        indent=4,
    )


# =========================================================
# Save training summary
# =========================================================
training_dates = pd.to_datetime(
    data["date"]
)


training_summary = pd.DataFrame(
    [
        {
            "model":
                "CatBoostRegressor",

            "configuration":
                BEST_CONFIGURATION,

            "feature_set":
                FEATURE_SET_NAME,

            "training_rows":
                len(data),

            "feature_count":
                len(X_train.columns),

            "categorical_feature_count":
                len(categorical_features),

            "iterations":
                final_iterations,

            "depth":
                8,

            "learning_rate":
                0.025,

            "l2_leaf_reg":
                10.0,

            "random_strength":
                1.0,

            "target_handling":
                "keep_all",

            "training_start_date":
                training_dates.min(),

            "training_end_date":
                training_dates.max(),
        }
    ]
)


training_summary.to_csv(
    TRAINING_SUMMARY_FILE,
    index=False,
)


# =========================================================
# Final confirmation
# =========================================================
print("\n")
print("=" * 75)
print("FINAL MODEL SAVED")
print("=" * 75)


print(
    f"Model:\n  {MODEL_FILE}"
)

print(
    f"\nPreprocessing statistics:\n  "
    f"{PREPROCESSING_FILE}"
)

print(
    f"\nFeature metadata:\n  "
    f"{FEATURE_METADATA_FILE}"
)

print(
    f"\nModel configuration:\n  "
    f"{MODEL_CONFIG_FILE}"
)

print(
    f"\nTraining summary:\n  "
    f"{TRAINING_SUMMARY_FILE}"
)


print(
    "\nFINAL CATBOOST MODEL TRAINING COMPLETE"
)