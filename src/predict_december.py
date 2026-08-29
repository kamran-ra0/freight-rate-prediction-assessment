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
DATA_DIR = ROOT / "data"
TABLE_DIR = ROOT / "reports" / "tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = (
    RAW_DIR / "train-test.csv"
)

DECEMBER_INPUT_FILE = (
    RAW_DIR / "december-chart-inputs.csv"
)

MODEL_FILE = (
    MODEL_DIR / "final_catboost_model.cbm"
)

PREPROCESSING_FILE = (
    MODEL_DIR / "preprocessing_stats.json"
)

FEATURE_METADATA_FILE = (
    MODEL_DIR / "feature_metadata.json"
)

# Scorer-ready output.
# The original supplied file remains unchanged.
OUTPUT_FILE = (
    DATA_DIR / "december_chart_inputs.csv"
)

SUMMARY_FILE = (
    TABLE_DIR / "december_prediction_summary.csv"
)


# =========================================================
# Load December input
# =========================================================
print("\nDECEMBER PREDICTION")
print("=" * 75)

december_raw = pd.read_csv(
    DECEMBER_INPUT_FILE
)

print(
    f"December rows: {len(december_raw):,}"
)


# =========================================================
# Basic structural check
# =========================================================
if len(december_raw) != 31:
    raise ValueError(
        "Expected exactly 31 December rows."
    )


required_columns = [
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "date",
]


missing_required_columns = [
    column
    for column in required_columns
    if column not in december_raw.columns
]


if missing_required_columns:
    raise ValueError(
        "December input is missing required columns: "
        + ", ".join(missing_required_columns)
    )


# =========================================================
# Verify fixed assessment scenario
# =========================================================
expected_dates = pd.date_range(
    start="2025-12-01",
    end="2025-12-31",
    freq="D",
)


actual_dates = (
    pd.to_datetime(
        december_raw["date"],
        errors="raise",
    )
    .reset_index(drop=True)
)


expected_date_series = pd.Series(
    expected_dates
)


if not actual_dates.equals(
    expected_date_series
):
    raise ValueError(
        "December dates are not exactly "
        "2025-12-01 through 2025-12-31."
    )


fixed_checks = {
    "pickup": "Lexington",
    "delivery": "Fort Wayne",
    "distance": 360,
    "equipment": "Dry Van",
    "weight": 32000,
}


for column, expected_value in fixed_checks.items():

    if not (
        december_raw[column]
        == expected_value
    ).all():

        raise ValueError(
            f"Unexpected value found in "
            f"December column '{column}'."
        )


print(
    "Fixed December scenario verified."
)


# =========================================================
# Load saved preprocessing statistics
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
# Load saved feature metadata
# =========================================================
with open(
    FEATURE_METADATA_FILE,
    "r",
    encoding="utf-8",
) as file:

    feature_metadata = json.load(
        file
    )


feature_set_name = (
    feature_metadata[
        "feature_set"
    ]
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
    f"Feature set: {feature_set_name}"
)


# =========================================================
# Prepare December model input
# =========================================================

# Do NOT modify the supplied December file.
# Create an in-memory copy for model preparation.
december_model_input = (
    december_raw.copy()
)


# =========================================================
# Obtain coordinates from original training data
# =========================================================
print(
    "Deriving route coordinates "
    "from training data..."
)


training_reference = pd.read_csv(
    TRAIN_FILE
)


location_columns_required = [
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
]


missing_location_columns = [
    column
    for column in location_columns_required
    if column not in training_reference.columns
]


if missing_location_columns:
    raise ValueError(
        "Training file is missing coordinate columns: "
        + ", ".join(missing_location_columns)
    )


# ---------------------------------------------------------
# Pickup city coordinates
# ---------------------------------------------------------
pickup_locations = (
    training_reference[
        [
            "pickup",
            "pickup_lat",
            "pickup_lon",
        ]
    ]
    .rename(
        columns={
            "pickup": "city",
            "pickup_lat": "lat",
            "pickup_lon": "lon",
        }
    )
)


# ---------------------------------------------------------
# Delivery city coordinates
# ---------------------------------------------------------
delivery_locations = (
    training_reference[
        [
            "delivery",
            "delivery_lat",
            "delivery_lon",
        ]
    ]
    .rename(
        columns={
            "delivery": "city",
            "delivery_lat": "lat",
            "delivery_lon": "lon",
        }
    )
)


# ---------------------------------------------------------
# Combine both city-coordinate sources
# ---------------------------------------------------------
city_locations = pd.concat(
    [
        pickup_locations,
        delivery_locations,
    ],
    ignore_index=True,
)


city_locations = (
    city_locations
    .dropna(
        subset=[
            "city",
            "lat",
            "lon",
        ]
    )
)


# Median is used defensively if the same city has more
# than one recorded coordinate pair.
city_lookup = (
    city_locations
    .groupby(
        "city",
        as_index=True,
    )
    .agg(
        lat=("lat", "median"),
        lon=("lon", "median"),
    )
)


# =========================================================
# Add pickup coordinates
# =========================================================
december_model_input[
    "pickup_lat"
] = (
    december_model_input[
        "pickup"
    ]
    .map(
        city_lookup["lat"]
    )
)


december_model_input[
    "pickup_lon"
] = (
    december_model_input[
        "pickup"
    ]
    .map(
        city_lookup["lon"]
    )
)


# =========================================================
# Add delivery coordinates
# =========================================================
december_model_input[
    "delivery_lat"
] = (
    december_model_input[
        "delivery"
    ]
    .map(
        city_lookup["lat"]
    )
)


december_model_input[
    "delivery_lon"
] = (
    december_model_input[
        "delivery"
    ]
    .map(
        city_lookup["lon"]
    )
)


coordinate_columns = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
]


if (
    december_model_input[
        coordinate_columns
    ]
    .isna()
    .any()
    .any()
):

    missing_cities = set()

    pickup_missing = (
        december_model_input[
            "pickup_lat"
        ]
        .isna()
    )

    delivery_missing = (
        december_model_input[
            "delivery_lat"
        ]
        .isna()
    )

    missing_cities.update(
        december_model_input.loc[
            pickup_missing,
            "pickup",
        ].tolist()
    )

    missing_cities.update(
        december_model_input.loc[
            delivery_missing,
            "delivery",
        ].tolist()
    )

    raise ValueError(
        "Could not derive coordinates for: "
        + ", ".join(
            sorted(missing_cities)
        )
    )


print(
    "Pickup coordinates:",
    float(
        december_model_input[
            "pickup_lat"
        ].iloc[0]
    ),
    float(
        december_model_input[
            "pickup_lon"
        ].iloc[0]
    ),
)


print(
    "Delivery coordinates:",
    float(
        december_model_input[
            "delivery_lat"
        ].iloc[0]
    ),
    float(
        december_model_input[
            "delivery_lon"
        ].iloc[0]
    ),
)


# =========================================================
# Add compatibility market_index column
# =========================================================

# Feature Set B does NOT use market_index.
#
# However, the common preprocessing function expects this
# source column because it was designed to support all
# feature configurations.
#
# Therefore, create the column only in memory. It will NOT
# appear in the final December output file.
if (
    "market_index"
    not in december_model_input.columns
):

    december_model_input[
        "market_index"
    ] = np.nan


# =========================================================
# Apply exact saved preprocessing
# =========================================================
december_clean = apply_preprocessing(
    december_model_input,
    preprocessing_stats,
)


# =========================================================
# Feature engineering
# =========================================================
X_december, categorical_features = (
    create_features(
        december_clean,
        **feature_options,
    )
)


# =========================================================
# Verify categorical configuration
# =========================================================
if (
    categorical_features
    != expected_categorical_features
):

    raise ValueError(
        "December categorical feature definition "
        "does not match final model metadata."
    )


# =========================================================
# Verify required feature columns
# =========================================================
missing_model_features = [
    column
    for column in expected_feature_columns
    if column not in X_december.columns
]


if missing_model_features:
    raise ValueError(
        "December feature engineering is missing: "
        + ", ".join(
            missing_model_features
        )
    )


# Use exactly the same columns and order used for training
X_december = X_december.reindex(
    columns=expected_feature_columns
)


print(
    f"Prediction features: "
    f"{len(X_december.columns)}"
)


# =========================================================
# Final feature integrity check
# =========================================================
numeric_columns = [
    column
    for column in X_december.columns
    if column not in categorical_features
]


if (
    X_december[
        numeric_columns
    ]
    .isna()
    .any()
    .any()
):
    raise ValueError(
        "Missing numeric values remain "
        "in December model features."
    )


numeric_array = (
    X_december[
        numeric_columns
    ]
    .to_numpy(
        dtype=float
    )
)


if not np.isfinite(
    numeric_array
).all():
    raise ValueError(
        "Non-finite values detected "
        "in December features."
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
        X_december
    ),
    dtype=float,
)


# =========================================================
# Prediction integrity checks
# =========================================================
if len(predictions) != 31:
    raise ValueError(
        "Expected exactly 31 predictions."
    )


if not np.isfinite(
    predictions
).all():
    raise ValueError(
        "Non-finite December predictions detected."
    )


if (
    predictions <= 0
).any():
    raise ValueError(
        "Non-positive December predictions detected."
    )


# =========================================================
# Create scorer-ready output
# =========================================================

# Start from the ORIGINAL supplied columns.
# This prevents temporary coordinate or market columns
# from leaking into the scorer file.
december_output = (
    december_raw.copy()
)


december_output[
    "predicted_rate"
] = predictions


# =========================================================
# Verify output
# =========================================================
if (
    december_output[
        "predicted_rate"
    ]
    .isna()
    .any()
):
    raise ValueError(
        "Missing December predictions detected."
    )


if len(december_output) != 31:
    raise ValueError(
        "December output does not contain 31 rows."
    )


# =========================================================
# Save scorer-ready file
# =========================================================
december_output.to_csv(
    OUTPUT_FILE,
    index=False,
)


# =========================================================
# Save prediction summary
# =========================================================
summary = pd.DataFrame(
    {
        "metric": [
            "count",
            "minimum",
            "mean",
            "median",
            "maximum",
            "standard_deviation",
        ],

        "value": [
            len(predictions),
            predictions.min(),
            predictions.mean(),
            np.median(predictions),
            predictions.max(),
            predictions.std(),
        ],
    }
)


summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


# =========================================================
# Final confirmation
# =========================================================
print("\n")
print("=" * 75)
print("DECEMBER PREDICTIONS COMPLETE")
print("=" * 75)


print(
    f"Rows:               "
    f"{len(december_output):,}"
)


print(
    f"Minimum prediction: "
    f"${predictions.min():,.2f}"
)


print(
    f"Mean prediction:    "
    f"${predictions.mean():,.2f}"
)


print(
    f"Median prediction:  "
    f"${np.median(predictions):,.2f}"
)


print(
    f"Maximum prediction: "
    f"${predictions.max():,.2f}"
)


print(
    "\nScorer-ready file:"
)


print(
    f"  {OUTPUT_FILE}"
)


print(
    "\nOutput columns:"
)


print(
    december_output.columns.tolist()
)


print(
    "\nFirst five predictions:"
)


print(
    december_output[
        [
            "date",
            "predicted_rate",
        ]
    ]
    .head()
    .to_string(
        index=False
    )
)


print(
    "\nLast five predictions:"
)


print(
    december_output[
        [
            "date",
            "predicted_rate",
        ]
    ]
    .tail()
    .to_string(
        index=False
    )
)


print(
    "\nDECEMBER PREDICTION CHECK PASSED"
)