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


# Keep the original assessment filenames exactly as provided.
TRAIN_FILE = RAW_DIR / "train-test.csv"
VALIDATION_FILE = RAW_DIR / "validation.csv"
TEMPLATE_FILE = RAW_DIR / "validation-predictions-template.csv"
DECEMBER_FILE = RAW_DIR / "december-chart-inputs.csv"


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def load_csv(path: Path, label: str) -> pd.DataFrame:
    """Load a CSV file and stop with a clear message if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")

    frame = pd.read_csv(path)

    print(f"\n{'=' * 70}")
    print(label)
    print(f"{'=' * 70}")
    print(f"Path: {path}")
    print(f"Rows: {len(frame):,}")
    print(f"Columns: {len(frame.columns)}")

    return frame


def basic_dataset_summary(
    frame: pd.DataFrame,
    dataset_name: str,
) -> dict:
    """Return basic structural information for one dataset."""
    summary = {
        "dataset": dataset_name,
        "rows": len(frame),
        "columns": len(frame.columns),
        "duplicate_rows": int(frame.duplicated().sum()),
        "total_missing_values": int(frame.isna().sum().sum()),
    }

    if "load_id" in frame.columns:
        summary["missing_load_id"] = int(frame["load_id"].isna().sum())
        summary["duplicate_load_id"] = int(frame["load_id"].duplicated().sum())
        summary["unique_load_id"] = int(frame["load_id"].nunique())
    else:
        summary["missing_load_id"] = np.nan
        summary["duplicate_load_id"] = np.nan
        summary["unique_load_id"] = np.nan

    return summary


def missing_value_summary(
    frame: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """Create a column-level missing-value summary."""
    result = pd.DataFrame(
        {
            "dataset": dataset_name,
            "column": frame.columns,
            "missing_count": frame.isna().sum().values,
            "missing_pct": frame.isna().mean().values * 100,
        }
    )

    return result


def numeric_summary(
    frame: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """Summarize all numeric columns."""
    numeric_columns = frame.select_dtypes(include=np.number).columns
    rows = []

    for column in numeric_columns:
        values = frame[column]

        # Some supplied output/template columns are intentionally all missing.
        # Avoid calculating quantiles on an entirely empty numeric series.
        if values.notna().sum() == 0:
            rows.append(
                {
                    "dataset": dataset_name,
                    "column": column,
                    "count": 0,
                    "missing": int(values.isna().sum()),
                    "min": np.nan,
                    "p01": np.nan,
                    "p05": np.nan,
                    "median": np.nan,
                    "mean": np.nan,
                    "p95": np.nan,
                    "p99": np.nan,
                    "max": np.nan,
                    "std": np.nan,
                    "negative_count": 0,
                    "zero_count": 0,
                }
            )
            continue

        rows.append(
            {
                "dataset": dataset_name,
                "column": column,
                "count": int(values.count()),
                "missing": int(values.isna().sum()),
                "min": values.min(),
                "p01": values.quantile(0.01),
                "p05": values.quantile(0.05),
                "median": values.median(),
                "mean": values.mean(),
                "p95": values.quantile(0.95),
                "p99": values.quantile(0.99),
                "max": values.max(),
                "std": values.std(),
                "negative_count": int((values < 0).sum()),
                "zero_count": int((values == 0).sum()),
            }
        )

    return pd.DataFrame(rows)


def add_anomaly(
    anomalies: list,
    dataset: str,
    column: str,
    issue: str,
    count: int,
    severity: str,
    note: str,
) -> None:
    """Add an anomaly to the audit report."""
    anomalies.append(
        {
            "dataset": dataset,
            "column": column,
            "issue": issue,
            "count": int(count),
            "severity": severity,
            "note": note,
        }
    )


# ---------------------------------------------------------
# Load all assessment files
# ---------------------------------------------------------
train = load_csv(TRAIN_FILE, "TRAINING DATA")
validation = load_csv(VALIDATION_FILE, "FINAL VALIDATION DATA")
template = load_csv(TEMPLATE_FILE, "PREDICTION TEMPLATE")
december = load_csv(DECEMBER_FILE, "DECEMBER CHART INPUTS")


datasets = {
    "train": train,
    "validation": validation,
    "template": template,
    "december": december,
}


# ---------------------------------------------------------
# 1. Dataset structure
# ---------------------------------------------------------
print("\n\nCOLUMN STRUCTURE")

for name, frame in datasets.items():
    print(f"\n{name.upper()}")
    print("-" * 50)

    for column, dtype in frame.dtypes.items():
        print(f"{column:<25} {dtype}")


overview_rows = [
    basic_dataset_summary(frame, name)
    for name, frame in datasets.items()
]

overview = pd.DataFrame(overview_rows)

overview.to_csv(
    TABLE_DIR / "dataset_overview.csv",
    index=False,
)


# ---------------------------------------------------------
# 2. Missing values
# ---------------------------------------------------------
missing_tables = []

for name, frame in datasets.items():
    missing_tables.append(
        missing_value_summary(frame, name)
    )

missing_report = pd.concat(
    missing_tables,
    ignore_index=True,
)

missing_report.to_csv(
    TABLE_DIR / "missing_values_summary.csv",
    index=False,
)


print("\n\nMISSING VALUES")

for name, frame in datasets.items():
    print(f"\n{name.upper()}")
    missing = frame.isna().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values.")
    else:
        for column, count in missing.items():
            pct = count / len(frame) * 100

            print(
                f"{column:<25} "
                f"{count:>6,} "
                f"({pct:6.2f}%)"
            )


# ---------------------------------------------------------
# 3. Duplicate checks
# ---------------------------------------------------------
print("\n\nDUPLICATE CHECKS")

for name, frame in datasets.items():
    print(f"\n{name.upper()}")

    print(
        "Duplicate full rows:",
        f"{frame.duplicated().sum():,}",
    )

    if "load_id" in frame.columns:
        print(
            "Duplicate load_id:",
            f"{frame['load_id'].duplicated().sum():,}",
        )


# ---------------------------------------------------------
# 4. Numeric column summaries
# ---------------------------------------------------------
numeric_tables = []

for name, frame in datasets.items():
    numeric_tables.append(
        numeric_summary(frame, name)
    )

numeric_report = pd.concat(
    numeric_tables,
    ignore_index=True,
)

numeric_report.to_csv(
    TABLE_DIR / "numeric_summary.csv",
    index=False,
)


# ---------------------------------------------------------
# 5. Date audit
# ---------------------------------------------------------
date_rows = []

for name, frame in datasets.items():
    if "date" not in frame.columns:
        continue

    converted = pd.to_datetime(
        frame["date"],
        errors="coerce",
    )

    original_missing = frame["date"].isna().sum()

    invalid_dates = int(
        converted.isna().sum()
        - original_missing
    )

    date_rows.append(
        {
            "dataset": name,
            "min_date": converted.min(),
            "max_date": converted.max(),
            "missing_date": int(original_missing),
            "invalid_date": invalid_dates,
            "unique_dates": int(converted.nunique()),
        }
    )


date_report = pd.DataFrame(date_rows)

date_report.to_csv(
    TABLE_DIR / "date_summary.csv",
    index=False,
)


print("\n\nDATE RANGES")

for _, row in date_report.iterrows():
    print(
        f"{row['dataset']:<12} "
        f"{row['min_date']}  ->  {row['max_date']}"
    )


# ---------------------------------------------------------
# 6. Domain-based anomaly checks
# ---------------------------------------------------------
anomalies = []


for dataset_name in ["train", "validation"]:
    frame = datasets[dataset_name]

    # Load ID
    if "load_id" in frame.columns:
        add_anomaly(
            anomalies,
            dataset_name,
            "load_id",
            "missing load_id",
            frame["load_id"].isna().sum(),
            "critical",
            "Every load must have a unique identifier.",
        )

        add_anomaly(
            anomalies,
            dataset_name,
            "load_id",
            "duplicate load_id",
            frame["load_id"].duplicated().sum(),
            "critical",
            "load_id should uniquely identify each load.",
        )

    # Distance
    if "distance" in frame.columns:
        add_anomaly(
            anomalies,
            dataset_name,
            "distance",
            "missing distance",
            frame["distance"].isna().sum(),
            "high",
            "Distance is a core freight-rate predictor.",
        )

        add_anomaly(
            anomalies,
            dataset_name,
            "distance",
            "non-positive distance",
            (frame["distance"] <= 0).sum(),
            "critical",
            "Freight distance must be positive.",
        )

    # Weight
    if "weight" in frame.columns:
        add_anomaly(
            anomalies,
            dataset_name,
            "weight",
            "missing weight",
            frame["weight"].isna().sum(),
            "medium",
            "Missing values require a training-derived imputation rule.",
        )

        add_anomaly(
            anomalies,
            dataset_name,
            "weight",
            "negative weight",
            (frame["weight"] < 0).sum(),
            "high",
            "Negative physical freight weight is invalid and may represent a sign-entry error.",
        )

        add_anomaly(
            anomalies,
            dataset_name,
            "weight",
            "zero weight",
            (frame["weight"] == 0).sum(),
            "high",
            "Zero freight weight should be investigated.",
        )

    # Coordinates
    coordinate_rules = {
        "pickup_lat": (-90, 90),
        "delivery_lat": (-90, 90),
        "pickup_lon": (-180, 180),
        "delivery_lon": (-180, 180),
    }

    for column, (minimum, maximum) in coordinate_rules.items():
        if column in frame.columns:
            invalid = (
                frame[column].notna()
                & ~frame[column].between(minimum, maximum)
            ).sum()

            add_anomaly(
                anomalies,
                dataset_name,
                column,
                "coordinate outside valid range",
                invalid,
                "critical",
                f"Expected range is {minimum} to {maximum}.",
            )

    # Equipment
    if "equipment" in frame.columns:
        add_anomaly(
            anomalies,
            dataset_name,
            "equipment",
            "missing equipment",
            frame["equipment"].isna().sum(),
            "high",
            "Equipment type is a categorical freight-rate predictor.",
        )

    # Market index
    if "market_index" in frame.columns:
        add_anomaly(
            anomalies,
            dataset_name,
            "market_index",
            "missing market_index",
            frame["market_index"].isna().sum(),
            "medium",
            "Missing values must be handled without using final validation statistics.",
        )

        add_anomaly(
            anomalies,
            dataset_name,
            "market_index",
            "non-positive market_index",
            (frame["market_index"] <= 0).sum(),
            "medium",
            "Non-positive values should be inspected for validity.",
        )

    # Quote signal
    if "quote_signal" in frame.columns:
        add_anomaly(
            anomalies,
            dataset_name,
            "quote_signal",
            "missing quote_signal",
            frame["quote_signal"].isna().sum(),
            "medium",
            "Missing values should be audited.",
        )

        add_anomaly(
            anomalies,
            dataset_name,
            "quote_signal",
            "non-positive quote_signal",
            (frame["quote_signal"] <= 0).sum(),
            "medium",
            "Non-positive values should be inspected.",
        )


# Target-specific checks only apply to labeled training data.
if "posted_rate" in train.columns:
    add_anomaly(
        anomalies,
        "train",
        "posted_rate",
        "missing posted_rate",
        train["posted_rate"].isna().sum(),
        "critical",
        "The supervised-learning target cannot be missing.",
    )

    add_anomaly(
        anomalies,
        "train",
        "posted_rate",
        "non-positive posted_rate",
        (train["posted_rate"] <= 0).sum(),
        "critical",
        "Freight rate should be positive.",
    )


anomaly_report = pd.DataFrame(anomalies)

anomaly_report.to_csv(
    TABLE_DIR / "anomaly_summary.csv",
    index=False,
)


print("\n\nDOMAIN-BASED ANOMALIES")

visible_anomalies = anomaly_report[
    anomaly_report["count"] > 0
]

if visible_anomalies.empty:
    print("No domain-based anomalies found.")
else:
    print(
        visible_anomalies.to_string(index=False)
    )


# ---------------------------------------------------------
# 7. Candidate statistical anomalies in the target
# ---------------------------------------------------------
if {"posted_rate", "distance"}.issubset(train.columns):

    train_rate_analysis = train[
        ["load_id", "distance", "posted_rate"]
    ].copy()

    train_rate_analysis["rate_per_mile"] = (
        train_rate_analysis["posted_rate"]
        / train_rate_analysis["distance"]
    )

    # Posted-rate IQR
    rate_q1 = train["posted_rate"].quantile(0.25)
    rate_q3 = train["posted_rate"].quantile(0.75)
    rate_iqr = rate_q3 - rate_q1

    rate_lower = rate_q1 - 1.5 * rate_iqr
    rate_upper = rate_q3 + 1.5 * rate_iqr

    train_rate_analysis["posted_rate_iqr_flag"] = (
        (train_rate_analysis["posted_rate"] < rate_lower)
        | (train_rate_analysis["posted_rate"] > rate_upper)
    )

    # Rate-per-mile IQR
    rpm = train_rate_analysis["rate_per_mile"]

    rpm_q1 = rpm.quantile(0.25)
    rpm_q3 = rpm.quantile(0.75)
    rpm_iqr = rpm_q3 - rpm_q1

    rpm_lower = rpm_q1 - 1.5 * rpm_iqr
    rpm_upper = rpm_q3 + 1.5 * rpm_iqr

    train_rate_analysis["rate_per_mile_iqr_flag"] = (
        (rpm < rpm_lower)
        | (rpm > rpm_upper)
    )

    train_rate_analysis.to_csv(
        TABLE_DIR / "target_anomaly_candidates.csv",
        index=False,
    )

    print("\n\nTARGET ANOMALY CANDIDATES")

    print(
        f"Posted-rate IQR range: "
        f"{rate_lower:,.2f} to {rate_upper:,.2f}"
    )

    print(
        "Posted-rate candidate outliers:",
        f"{train_rate_analysis['posted_rate_iqr_flag'].sum():,}",
    )

    print(
        f"Rate-per-mile IQR range: "
        f"{rpm_lower:,.3f} to {rpm_upper:,.3f}"
    )

    print(
        "Rate-per-mile candidate outliers:",
        f"{train_rate_analysis['rate_per_mile_iqr_flag'].sum():,}",
    )

    print(
        "\nImportant: these are statistical flags only. "
        "They are NOT automatically removed."
    )


# ---------------------------------------------------------
# 8. Train vs validation categorical coverage
# ---------------------------------------------------------
coverage_rows = []

for column in ["pickup", "delivery", "equipment"]:

    if (
        column in train.columns
        and column in validation.columns
    ):
        train_values = set(
            train[column].dropna().astype(str)
        )

        validation_values = set(
            validation[column].dropna().astype(str)
        )

        unseen = sorted(
            validation_values - train_values
        )

        affected_rows = validation[
            column
        ].astype(str).isin(unseen).sum()

        coverage_rows.append(
            {
                "column": column,
                "train_unique": len(train_values),
                "validation_unique": len(validation_values),
                "unseen_validation_values": len(unseen),
                "validation_rows_affected": int(affected_rows),
                "unseen_values": " | ".join(unseen),
            }
        )


coverage_report = pd.DataFrame(coverage_rows)

coverage_report.to_csv(
    TABLE_DIR / "categorical_coverage.csv",
    index=False,
)


print("\n\nTRAIN VS VALIDATION CATEGORY COVERAGE")

print(
    coverage_report.to_string(index=False)
)


# ---------------------------------------------------------
# 9. Final audit message
# ---------------------------------------------------------
print("\n\nAUDIT COMPLETE")

print(
    "Reports saved to:",
    TABLE_DIR,
)

print("\nGenerated files:")

for path in sorted(TABLE_DIR.glob("*.csv")):
    print(" -", path.name)