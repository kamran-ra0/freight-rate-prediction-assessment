from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class TimeSplit:
    name: str
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str


FORWARD_SPLITS = [
    TimeSplit(
        name="fold_1_august",
        train_start="2025-01-01",
        train_end="2025-07-31",
        validation_start="2025-08-01",
        validation_end="2025-08-31",
    ),
    TimeSplit(
        name="fold_2_september",
        train_start="2025-01-01",
        train_end="2025-08-31",
        validation_start="2025-09-01",
        validation_end="2025-09-30",
    ),
    TimeSplit(
        name="fold_3_october",
        train_start="2025-01-01",
        train_end="2025-09-30",
        validation_start="2025-10-01",
        validation_end="2025-10-31",
    ),
]


def prepare_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy with a validated datetime column.
    """
    result = df.copy()

    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce",
    )

    if result["date"].isna().any():
        raise ValueError(
            "Invalid or missing date values found."
        )

    return result


def make_time_split(
    df: pd.DataFrame,
    split: TimeSplit,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create one chronological train/validation split.
    """
    data = prepare_dates(df)

    train_mask = (
        (data["date"] >= split.train_start)
        & (data["date"] <= split.train_end)
    )

    validation_mask = (
        (data["date"] >= split.validation_start)
        & (data["date"] <= split.validation_end)
    )

    train_df = data.loc[train_mask].copy()
    validation_df = data.loc[validation_mask].copy()

    if train_df.empty:
        raise ValueError(
            f"{split.name}: training split is empty."
        )

    if validation_df.empty:
        raise ValueError(
            f"{split.name}: validation split is empty."
        )

    if train_df["date"].max() >= validation_df["date"].min():
        raise ValueError(
            f"{split.name}: chronological leakage detected."
        )

    return train_df, validation_df


def describe_split(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    split: TimeSplit,
) -> dict:
    """
    Return summary information for one split.
    """
    return {
        "fold": split.name,
        "train_rows": len(train_df),
        "train_start": train_df["date"].min(),
        "train_end": train_df["date"].max(),
        "validation_rows": len(validation_df),
        "validation_start": validation_df["date"].min(),
        "validation_end": validation_df["date"].max(),
    }