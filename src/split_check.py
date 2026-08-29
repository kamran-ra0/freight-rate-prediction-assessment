from pathlib import Path

import pandas as pd

from split import (
    FORWARD_SPLITS,
    describe_split,
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
# Load labeled development data
# ---------------------------------------------------------
data = pd.read_csv(TRAIN_FILE)


# ---------------------------------------------------------
# Create and verify all chronological folds
# ---------------------------------------------------------
summaries = []

print("\nCHRONOLOGICAL VALIDATION SPLITS")
print("-" * 75)

for split in FORWARD_SPLITS:

    train_df, validation_df = make_time_split(
        data,
        split,
    )

    summary = describe_split(
        train_df,
        validation_df,
        split,
    )

    summaries.append(summary)

    print(f"\n{split.name}")
    print("-" * 50)

    print(
        f"Training:   "
        f"{summary['train_start'].date()} "
        f"to {summary['train_end'].date()}"
    )

    print(
        f"Train rows: "
        f"{summary['train_rows']:,}"
    )

    print(
        f"Validation: "
        f"{summary['validation_start'].date()} "
        f"to {summary['validation_end'].date()}"
    )

    print(
        f"Validation rows: "
        f"{summary['validation_rows']:,}"
    )

    # -----------------------------------------------------
    # Leakage checks
    # -----------------------------------------------------
    assert (
        train_df["date"].max()
        < validation_df["date"].min()
    )

    train_ids = set(train_df["load_id"])
    validation_ids = set(validation_df["load_id"])

    assert train_ids.isdisjoint(
        validation_ids
    )

    print("Chronological leakage check: PASSED")
    print("Load ID overlap check:       PASSED")


# ---------------------------------------------------------
# Save split summary
# ---------------------------------------------------------
summary_table = pd.DataFrame(
    summaries
)

summary_table.to_csv(
    TABLE_DIR / "chronological_split_summary.csv",
    index=False,
)


# ---------------------------------------------------------
# Final output
# ---------------------------------------------------------
print("\n" + "=" * 75)

print("ALL CHRONOLOGICAL SPLITS PASSED")

print(
    "\nSummary saved to:",
    TABLE_DIR / "chronological_split_summary.csv",
)