from pathlib import Path
import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from features import create_features
from preprocessing import (
    PreprocessingStats,
    apply_preprocessing,
)


# =========================================================
# Paths
# =========================================================
ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
TABLE_DIR = ROOT / "reports" / "tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_FILE = RAW_DIR / "validation.csv"

TEMPLATE_FILE = (
    RAW_DIR
    / "validation-predictions-template.csv"
)

MODEL_FILE = (
    MODEL_DIR
    / "final_catboost_model.cbm"
)

PREPROCESSING_FILE = (
    MODEL_DIR
    / "preprocessing_stats.json"
)

FEATURE_METADATA_FILE = (
    MODEL_DIR
    / "feature_metadata.json"
)

OUTPUT_FILE = (
    ROOT
    / "validation_predictions.csv"
)


# =========================================================
# Load validation data
# =========================================================
print("\nVALIDATION PREDICTION")
print("=" * 75)

validation_raw = pd.read_csv(
    VALIDATION_FILE
)

template = pd.read_csv(
    TEMPLATE_FILE
)

print(
    f"Validation rows: {len(validation_raw):,}"
)

print(
    f"Template rows:   {len(template):,}"
)


# =========================================================
# Basic validation
# =========================================================
if len(validation_raw) != 12000:
    raise ValueError(
        "Expected exactly 12,000 validation rows."
    )


if len(template) != 12000:
    raise ValueError(
        "Expected exactly 12,000 template rows."
    )


if validation_raw["load_id"].duplicated().any():
    raise ValueError(
        "Duplicate load_id values found in validation data."
    )


if template["load_id"].duplicated().any():
    raise ValueError(
        "Duplicate load_id values found in prediction template."
    )


# Ensure both files contain exactly the same IDs
validation_ids = set(
    validation_raw["load_id"]
)

template_ids = set(
    template["load_id"]
)

if validation_ids != template_ids:
    raise ValueError(
        "Validation and template load IDs do not match."
    )


# =========================================================
# Load preprocessing statistics
# =========================================================
with open(
    PREPROCESSING_FILE,
    "r",
    encoding="utf-8",
) as file:

    stats_json = json.load(
        file
    )


preprocessing_stats = PreprocessingStats(
    weight_median_by_equipment={
        str(key): float(value)
        for key, value
        in stats_json[
            "weight_median_by_equipment"
        ].items()
    },

    global_weight_median=float(
        stats_json[
            "global_weight_median"
        ]
    ),

    market_index_median=float(
        stats_json[
            "market_index_median"
        ]
    ),
)


# =========================================================
# Load feature metadata
# =========================================================
with open(
    FEATURE_METADATA_FILE,
    "r",
    encoding="utf-8",
) as file:

    feature_metadata = json.load(
        file
    )


feature_options = (
    feature_metadata[
        "feature_options"
    ]
)

expected_feature_columns = (
    feature_metadata[
        "feature_columns"
    ]
)

expected_categorical_features = (
    feature_metadata[
        "categorical_features"
    ]
)


print(
    f"Feature set: "
    f"{feature_metadata['feature_set']}"
)


# =========================================================
# Apply saved preprocessing
# =========================================================
validation_clean = apply_preprocessing(
    validation_raw,
    preprocessing_stats,
)


# =========================================================
# Feature engineering
# =========================================================
X_validation, categorical_features = (
    create_features(
        validation_clean,
        **feature_options,
    )
)


# Ensure feature structure exactly matches training
missing_features = [
    column
    for column in expected_feature_columns
    if column not in X_validation.columns
]

if missing_features:
    raise ValueError(
        "Missing required validation features: "
        + ", ".join(missing_features)
    )


X_validation = X_validation.reindex(
    columns=expected_feature_columns
)


if (
    categorical_features
    != expected_categorical_features
):
    raise ValueError(
        "Categorical feature definition does not "
        "match final training metadata."
    )


print(
    f"Prediction features: "
    f"{len(X_validation.columns)}"
)


# =========================================================
# Load final CatBoost model
# =========================================================
model = CatBoostRegressor()

model.load_model(
    MODEL_FILE
)


print(
    "Final CatBoost model loaded."
)


# =========================================================
# Generate predictions
# =========================================================
predictions = np.asarray(
    model.predict(
        X_validation
    ),
    dtype=float,
)


# =========================================================
# Prediction integrity checks
# =========================================================
if len(predictions) != 12000:
    raise ValueError(
        "Prediction count is not 12,000."
    )


if not np.isfinite(
    predictions
).all():
    raise ValueError(
        "Non-finite predictions detected."
    )


if (
    predictions <= 0
).any():
    raise ValueError(
        "Non-positive predictions detected."
    )


# =========================================================
# Associate predictions with load IDs
# =========================================================
prediction_lookup = pd.DataFrame(
    {
        "load_id":
            validation_raw[
                "load_id"
            ].values,

        "predicted_rate":
            predictions,
    }
)


# =========================================================
# Fill original template order
# =========================================================
submission = (
    template[
        ["load_id"]
    ]
    .merge(
        prediction_lookup,
        on="load_id",
        how="left",
        validate="one_to_one",
    )
)


# =========================================================
# Final scorer-format checks
# =========================================================
if submission[
    "predicted_rate"
].isna().any():

    raise ValueError(
        "Missing predictions after template merge."
    )


expected_columns = [
    "load_id",
    "predicted_rate",
]

if submission.columns.tolist() != expected_columns:
    raise ValueError(
        "Submission columns are incorrect."
    )


# Expected Spotter scorer IDs
expected_ids = [
    f"TE-{number:06d}"
    for number in range(
        1,
        12001,
    )
]


if submission[
    "load_id"
].tolist() != expected_ids:

    raise ValueError(
        "Submission load IDs are not in the required "
        "TE-000001 through TE-012000 order."
    )


# =========================================================
# Save required submission file
# =========================================================
submission.to_csv(
    OUTPUT_FILE,
    index=False,
)


# =========================================================
# Save prediction summary
# =========================================================
prediction_summary = (
    submission[
        "predicted_rate"
    ]
    .describe()
    .to_frame(
        name="predicted_rate"
    )
)


prediction_summary.to_csv(
    TABLE_DIR
    / "validation_prediction_summary.csv"
)


# =========================================================
# Confirmation
# =========================================================
print("\n")
print("=" * 75)
print("VALIDATION PREDICTIONS COMPLETE")
print("=" * 75)

print(
    f"Rows: {len(submission):,}"
)

print(
    f"Minimum prediction: "
    f"${submission['predicted_rate'].min():,.2f}"
)

print(
    f"Mean prediction:    "
    f"${submission['predicted_rate'].mean():,.2f}"
)

print(
    f"Median prediction:  "
    f"${submission['predicted_rate'].median():,.2f}"
)

print(
    f"Maximum prediction: "
    f"${submission['predicted_rate'].max():,.2f}"
)

print(
    "\nSubmission file:"
)

print(
    f"  {OUTPUT_FILE}"
)

print(
    "\nRequired columns:"
)

print(
    submission.columns.tolist()
)

print(
    "\nFirst five predictions:"
)

print(
    submission.head().to_string(
        index=False
    )
)

print(
    "\nVALIDATION PREDICTION CHECK PASSED"
)