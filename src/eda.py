from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
FIGURE_DIR = ROOT / "reports" / "figures"
TABLE_DIR = ROOT / "reports" / "tables"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


TRAIN_FILE = RAW_DIR / "train-test.csv"
VALIDATION_FILE = RAW_DIR / "validation.csv"


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VALIDATION_FILE)

train["date"] = pd.to_datetime(
    train["date"],
    errors="coerce",
)

validation["date"] = pd.to_datetime(
    validation["date"],
    errors="coerce",
)


# ---------------------------------------------------------
# Derived variables for analysis only
# ---------------------------------------------------------
train["rate_per_mile"] = (
    train["posted_rate"]
    / train["distance"]
)

train["month"] = train["date"].dt.to_period("M").astype(str)
train["day_of_week"] = train["date"].dt.day_name()

# Keep raw weight unchanged.
# abs_weight is used only to understand whether negative values
# resemble realistic sign-entry errors.
train["abs_weight"] = train["weight"].abs()
validation["abs_weight"] = validation["weight"].abs()


# ---------------------------------------------------------
# Plot helper
# ---------------------------------------------------------
def save_current_figure(filename: str) -> None:
    """Save the current Matplotlib figure and close it."""
    output = FIGURE_DIR / filename

    plt.tight_layout()
    plt.savefig(
        output,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved: {output.name}")


# =========================================================
# 1. Target-variable analysis
# =========================================================

# Posted-rate histogram
plt.figure(figsize=(9, 5))

plt.hist(
    train["posted_rate"],
    bins=80,
)

plt.title("Distribution of Posted Rate")
plt.xlabel("Posted rate ($)")
plt.ylabel("Number of loads")

save_current_figure(
    "01_posted_rate_distribution.png"
)


# Posted-rate boxplot
plt.figure(figsize=(9, 4))

plt.boxplot(
    train["posted_rate"].dropna(),
    vert=False,
)

plt.title("Posted Rate Boxplot")
plt.xlabel("Posted rate ($)")

save_current_figure(
    "02_posted_rate_boxplot.png"
)


# Log target distribution
plt.figure(figsize=(9, 5))

plt.hist(
    np.log1p(train["posted_rate"]),
    bins=70,
)

plt.title("Log-Transformed Posted Rate Distribution")
plt.xlabel("log(1 + posted rate)")
plt.ylabel("Number of loads")

save_current_figure(
    "03_log_posted_rate_distribution.png"
)


# =========================================================
# 2. Distance analysis
# =========================================================

plt.figure(figsize=(9, 5))

plt.hist(
    train["distance"],
    bins=70,
)

plt.title("Training Distance Distribution")
plt.xlabel("Distance (miles)")
plt.ylabel("Number of loads")

save_current_figure(
    "04_distance_distribution.png"
)


# Rate vs distance
sample_size = min(
    12000,
    len(train),
)

sample = train.sample(
    sample_size,
    random_state=42,
)

plt.figure(figsize=(9, 5))

plt.scatter(
    sample["distance"],
    sample["posted_rate"],
    s=8,
    alpha=0.30,
)

plt.title("Posted Rate vs Distance")
plt.xlabel("Distance (miles)")
plt.ylabel("Posted rate ($)")

save_current_figure(
    "05_posted_rate_vs_distance.png"
)


# =========================================================
# 3. Rate-per-mile analysis
# =========================================================

plt.figure(figsize=(9, 5))

plt.hist(
    train["rate_per_mile"],
    bins=80,
)

plt.title("Rate per Mile Distribution")
plt.xlabel("Rate per mile ($/mile)")
plt.ylabel("Number of loads")

save_current_figure(
    "06_rate_per_mile_distribution.png"
)


plt.figure(figsize=(9, 4))

plt.boxplot(
    train["rate_per_mile"].dropna(),
    vert=False,
)

plt.title("Rate per Mile Boxplot")
plt.xlabel("Rate per mile ($/mile)")

save_current_figure(
    "07_rate_per_mile_boxplot.png"
)


plt.figure(figsize=(9, 5))

plt.scatter(
    sample["distance"],
    sample["rate_per_mile"],
    s=8,
    alpha=0.30,
)

plt.title("Rate per Mile vs Distance")
plt.xlabel("Distance (miles)")
plt.ylabel("Rate per mile ($/mile)")

save_current_figure(
    "08_rate_per_mile_vs_distance.png"
)


# =========================================================
# 4. Weight analysis
# =========================================================

plt.figure(figsize=(9, 5))

plt.hist(
    train["weight"].dropna(),
    bins=80,
)

plt.title("Raw Weight Distribution")
plt.xlabel("Weight (lb)")
plt.ylabel("Number of loads")

save_current_figure(
    "09_raw_weight_distribution.png"
)


# Absolute weight helps inspect whether negative weights
# resemble realistic positive freight weights.
plt.figure(figsize=(9, 5))

plt.hist(
    train["abs_weight"].dropna(),
    bins=70,
)

plt.title("Absolute Weight Distribution")
plt.xlabel("Absolute weight (lb)")
plt.ylabel("Number of loads")

save_current_figure(
    "10_absolute_weight_distribution.png"
)


weight_sample = train.dropna(
    subset=["weight", "posted_rate"]
).sample(
    min(
        12000,
        train["weight"].notna().sum(),
    ),
    random_state=42,
)

plt.figure(figsize=(9, 5))

plt.scatter(
    weight_sample["weight"],
    weight_sample["posted_rate"],
    s=8,
    alpha=0.30,
)

plt.title("Posted Rate vs Raw Weight")
plt.xlabel("Weight (lb)")
plt.ylabel("Posted rate ($)")

save_current_figure(
    "11_posted_rate_vs_weight.png"
)


# =========================================================
# 5. Equipment analysis
# =========================================================

equipment_order = (
    train["equipment"]
    .value_counts()
    .index
    .tolist()
)

equipment_rates = [
    train.loc[
        train["equipment"] == equipment,
        "posted_rate",
    ].dropna()
    for equipment in equipment_order
]

plt.figure(figsize=(8, 5))

plt.boxplot(
    equipment_rates,
    tick_labels=equipment_order,
    showfliers=False,
)

plt.title("Posted Rate by Equipment Type")
plt.xlabel("Equipment")
plt.ylabel("Posted rate ($)")

save_current_figure(
    "12_posted_rate_by_equipment.png"
)


equipment_counts = (
    train["equipment"]
    .value_counts()
    .reindex(equipment_order)
)

plt.figure(figsize=(8, 5))

plt.bar(
    equipment_counts.index,
    equipment_counts.values,
)

plt.title("Training Loads by Equipment Type")
plt.xlabel("Equipment")
plt.ylabel("Number of loads")

save_current_figure(
    "13_equipment_counts.png"
)


# =========================================================
# 6. Temporal analysis
# =========================================================

monthly = (
    train.groupby("month")
    .agg(
        loads=("posted_rate", "size"),
        mean_rate=("posted_rate", "mean"),
        median_rate=("posted_rate", "median"),
        mean_distance=("distance", "mean"),
        median_rate_per_mile=("rate_per_mile", "median"),
    )
    .reset_index()
)

monthly.to_csv(
    TABLE_DIR / "monthly_eda_summary.csv",
    index=False,
)


plt.figure(figsize=(10, 5))

plt.plot(
    monthly["month"],
    monthly["median_rate"],
    marker="o",
)

plt.title("Monthly Median Posted Rate")
plt.xlabel("Month")
plt.ylabel("Median posted rate ($)")
plt.xticks(rotation=45)

save_current_figure(
    "14_monthly_median_posted_rate.png"
)


plt.figure(figsize=(10, 5))

plt.bar(
    monthly["month"],
    monthly["loads"],
)

plt.title("Monthly Load Counts")
plt.xlabel("Month")
plt.ylabel("Number of loads")
plt.xticks(rotation=45)

save_current_figure(
    "15_monthly_load_counts.png"
)


day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

dow_summary = (
    train.groupby("day_of_week")
    .agg(
        loads=("posted_rate", "size"),
        mean_rate=("posted_rate", "mean"),
        median_rate=("posted_rate", "median"),
        median_rate_per_mile=("rate_per_mile", "median"),
    )
    .reindex(day_order)
    .reset_index()
)

dow_summary.to_csv(
    TABLE_DIR / "day_of_week_summary.csv",
    index=False,
)


plt.figure(figsize=(9, 5))

plt.plot(
    dow_summary["day_of_week"],
    dow_summary["median_rate"],
    marker="o",
)

plt.title("Median Posted Rate by Day of Week")
plt.xlabel("Day of week")
plt.ylabel("Median posted rate ($)")
plt.xticks(rotation=35)

save_current_figure(
    "16_day_of_week_median_rate.png"
)


# =========================================================
# 7. Market-index analysis
# =========================================================

market_sample = train.dropna(
    subset=["market_index", "posted_rate"]
).sample(
    min(
        12000,
        train["market_index"].notna().sum(),
    ),
    random_state=42,
)

plt.figure(figsize=(9, 5))

plt.scatter(
    market_sample["market_index"],
    market_sample["posted_rate"],
    s=8,
    alpha=0.30,
)

plt.title("Posted Rate vs Market Index")
plt.xlabel("Market index")
plt.ylabel("Posted rate ($)")

save_current_figure(
    "17_posted_rate_vs_market_index.png"
)


# =========================================================
# 8. Quote-signal analysis
# =========================================================

quote_sample = train.dropna(
    subset=["quote_signal", "posted_rate"]
).sample(
    min(
        12000,
        train["quote_signal"].notna().sum(),
    ),
    random_state=42,
)

plt.figure(figsize=(9, 5))

plt.scatter(
    quote_sample["quote_signal"],
    quote_sample["posted_rate"],
    s=8,
    alpha=0.30,
)

plt.title("Posted Rate vs Quote Signal")
plt.xlabel("Quote signal")
plt.ylabel("Posted rate ($)")

save_current_figure(
    "18_posted_rate_vs_quote_signal.png"
)


# =========================================================
# 9. Missing-value visualization
# =========================================================

missing_counts = train.isna().sum()
missing_counts = missing_counts[
    missing_counts > 0
].sort_values(
    ascending=False
)

plt.figure(figsize=(8, 5))

plt.bar(
    missing_counts.index,
    missing_counts.values,
)

plt.title("Missing Values in Training Data")
plt.xlabel("Feature")
plt.ylabel("Missing rows")
plt.xticks(rotation=35)

save_current_figure(
    "19_training_missing_values.png"
)


# =========================================================
# 10. Training vs final-validation distribution shift
# =========================================================

# Distance
plt.figure(figsize=(9, 5))

plt.hist(
    train["distance"],
    bins=60,
    density=True,
    alpha=0.50,
    label="Training",
)

plt.hist(
    validation["distance"],
    bins=60,
    density=True,
    alpha=0.50,
    label="Validation",
)

plt.title("Distance Distribution: Training vs Validation")
plt.xlabel("Distance (miles)")
plt.ylabel("Density")
plt.legend()

save_current_figure(
    "20_distance_train_vs_validation.png"
)


# Absolute weight
plt.figure(figsize=(9, 5))

plt.hist(
    train["abs_weight"].dropna(),
    bins=60,
    density=True,
    alpha=0.50,
    label="Training",
)

plt.hist(
    validation["abs_weight"].dropna(),
    bins=60,
    density=True,
    alpha=0.50,
    label="Validation",
)

plt.title("Absolute Weight: Training vs Validation")
plt.xlabel("Absolute weight (lb)")
plt.ylabel("Density")
plt.legend()

save_current_figure(
    "21_weight_train_vs_validation.png"
)


# Market index
plt.figure(figsize=(9, 5))

plt.hist(
    train["market_index"].dropna(),
    bins=60,
    density=True,
    alpha=0.50,
    label="Training",
)

plt.hist(
    validation["market_index"].dropna(),
    bins=60,
    density=True,
    alpha=0.50,
    label="Validation",
)

plt.title("Market Index: Training vs Validation")
plt.xlabel("Market index")
plt.ylabel("Density")
plt.legend()

save_current_figure(
    "22_market_index_train_vs_validation.png"
)


# Quote signal
plt.figure(figsize=(9, 5))

plt.hist(
    train["quote_signal"].dropna(),
    bins=60,
    density=True,
    alpha=0.50,
    label="Training",
)

plt.hist(
    validation["quote_signal"].dropna(),
    bins=60,
    density=True,
    alpha=0.50,
    label="Validation",
)

plt.title("Quote Signal: Training vs Validation")
plt.xlabel("Quote signal")
plt.ylabel("Density")
plt.legend()

save_current_figure(
    "23_quote_signal_train_vs_validation.png"
)


# =========================================================
# 11. Correlation table
# =========================================================

correlation_columns = [
    "posted_rate",
    "distance",
    "weight",
    "abs_weight",
    "market_index",
    "quote_signal",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "rate_per_mile",
]

correlation = train[
    correlation_columns
].corr(
    method="pearson"
)

correlation.to_csv(
    TABLE_DIR / "correlation_matrix.csv"
)


# ---------------------------------------------------------
# Completion message
# ---------------------------------------------------------
print("\nEDA COMPLETE")

print(
    f"Figures saved to: {FIGURE_DIR}"
)

print(
    f"Tables saved to: {TABLE_DIR}"
)

print(
    f"Total PNG figures: "
    f"{len(list(FIGURE_DIR.glob('*.png')))}"
)