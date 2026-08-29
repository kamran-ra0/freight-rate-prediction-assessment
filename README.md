````markdown
# Freight Rate Prediction — Machine Learning Engineer Recruitment Assessment

## Overview

This repository contains an end-to-end machine learning solution for predicting freight load rates.

The workflow covers:

- data auditing and exploratory data analysis
- anomaly investigation
- leakage-safe preprocessing
- geographic and temporal feature engineering
- chronological forward validation
- CatBoost feature-set comparison
- hyperparameter tuning
- target-anomaly strategy testing
- final model training
- validation prediction generation
- fixed December prediction generation
- official scorer validation
- reproducible model and preprocessing artifacts

The final solution uses a tuned **CatBoostRegressor** trained on all 48,000 labeled development records.

---

## Assessment Objective

The objective is to predict `posted_rate` for:

1. **12,000 unseen validation loads**
2. **31 fixed December loads**

The required final validation file is:

```text
validation_predictions.csv
````

with exactly:

```text
load_id,predicted_rate
```

The December output is:

```text
data/december_chart_inputs.csv
```

The provided `score.py` validates both output files and generates the required December chart.

---

## Dataset

### Training data

```text
data/raw/train-test.csv
```

* Rows: **48,000**
* Columns: **14**
* Date range: **2025-01-01 to 2025-10-31**
* Target: `posted_rate`

### Validation data

```text
data/raw/validation.csv
```

* Rows: **12,000**
* Columns: **13**
* Date range: **2025-11-01 to 2025-12-31**
* Target unavailable

### Validation prediction template

```text
data/raw/validation-predictions-template.csv
```

* Rows: **12,000**
* Required columns:

  * `load_id`
  * `predicted_rate`

### Fixed December input

```text
data/raw/december-chart-inputs.csv
```

* Rows: **31**
* Dates: **2025-12-01 to 2025-12-31**
* Fixed route:

  * Pickup: Lexington
  * Delivery: Fort Wayne
  * Distance: 360 miles
  * Equipment: Dry Van
  * Weight: 32,000 lb

The original assessment files are retained unchanged.

---

# Methodology

## 1. Data Audit

The data was first checked for:

* duplicate rows
* duplicate `load_id` values
* missing values
* invalid dates
* non-positive distances
* invalid coordinates
* negative or missing weights
* missing market indicators
* target extremes
* unseen categorical values in validation

### Key findings

Training data contained:

* 300 missing weights
* 292 negative weights
* 374 missing `market_index` values
* no duplicate rows
* no duplicate load IDs
* no invalid dates
* no non-positive distances

Validation data contained:

* 165 missing weights
* 145 negative weights
* 249 missing `market_index` values

Approximately 12% of validation observations involved at least one pickup or delivery city not observed in the training data.

This was an important reason for emphasizing geographic coordinates rather than depending entirely on raw city-name categories.

---

## 2. Weight Cleaning

Negative weights were not deleted.

Their absolute magnitudes had nearly the same distribution as valid positive weights, indicating that the negative sign was most likely a data-entry issue rather than evidence of invalid loads.

The transformation used was:

```python
weight_clean = abs(weight)
```

Two indicator variables were retained:

```text
weight_negative_flag
weight_missing_flag
```

Missing weights were imputed using the median absolute weight for the corresponding equipment type.

A global training median was retained as a fallback.

All imputation statistics were learned from training data only.

---

## 3. Missing Market Index

Missing `market_index` values were initially imputed with the training-derived median and accompanied by:

```text
market_index_missing_flag
```

However, later chronological validation showed that market signals did not improve forward predictive performance.

The selected final feature set therefore excludes:

```text
market_index
quote_signal
```

This decision was made empirically rather than by assumption.

---

## 4. Target Anomaly Analysis

Both raw `posted_rate` and rate-per-mile were examined for extreme values.

A separate residual diagnostic was also performed using a simple distance + equipment regression baseline.

Strong residual anomalies existed, including a small number of very large positive and negative residuals.

However, statistical outliers were **not automatically removed**.

Three strategies were later tested using chronological validation:

1. keep all targets
2. remove strong residual anomalies
3. cap extreme target values

The best-performing approach was:

```text
keep_all
```

Therefore, all labeled observations were retained for final model training.

---

# Feature Engineering

The final feature pipeline generates geographic, temporal, load, and interaction features.

## Geographic features

Examples include:

* pickup latitude/longitude
* delivery latitude/longitude
* latitude change
* longitude change
* absolute latitude change
* absolute longitude change
* route midpoint latitude
* route midpoint longitude
* Haversine distance
* route detour ratio
* route bearing sine
* route bearing cosine

Geographic coordinates were preferred over sole reliance on city labels because validation contains previously unseen cities.

---

## Distance features

Features include:

```text
distance
log_distance
```

and route-distance relationships such as:

```text
haversine_distance
detour_ratio
```

---

## Weight features

Features include:

```text
weight_clean
weight_missing_flag
weight_negative_flag
distance_x_weight
weight_per_1000_miles
```

---

## Temporal features

The date was decomposed into:

```text
year
month
day_of_month
day_of_week
day_of_year
week_of_year
quarter
is_weekend
is_month_start
is_month_end
```

Cyclical encodings were also generated:

```text
day_of_week_sin
day_of_week_cos
month_sin
month_cos
day_of_year_sin
day_of_year_cos
```

Cyclical features allow calendar positions such as Sunday/Monday or December/January to be represented as neighboring positions rather than unrelated integers.

---

# Feature-Set Comparison

Three feature configurations were evaluated using the same CatBoost setup.

### Feature Set A

```text
Coordinates + engineered features + market signals
```

* 41 features
* categorical variable: `equipment`

### Feature Set B

```text
Coordinates + engineered features
No market signals
No raw city categories
```

* **38 features**
* categorical variable: `equipment`

### Feature Set C

```text
Coordinates + engineered features + raw city categories + market signals
```

* 43 features
* categorical variables:

  * `equipment`
  * `pickup`
  * `delivery`

The winning configuration was:

```text
B_coordinates_no_market
```

Initial chronological-validation performance:

| Metric    |  Result |
| --------- | ------: |
| Mean MAE  | $103.29 |
| Mean RMSE | $626.47 |
| Mean R²   |  0.8276 |

This result showed that adding market signals and raw city categories did not improve future-period generalization.

---

# Validation Strategy

A random train/test split was intentionally avoided.

The labeled development data covers January through October 2025, while the actual unseen validation data occurs later in time.

Random splitting could therefore mix future observations into model development and produce an overly optimistic estimate of real deployment performance.

An **expanding-window chronological validation strategy** was used instead.

## Fold 1

Training:

```text
2025-01-01 through 2025-07-31
```

Validation:

```text
2025-08-01 through 2025-08-31
```

* Training rows: 33,718
* Holdout rows: 4,759

## Fold 2

Training:

```text
2025-01-01 through 2025-08-31
```

Validation:

```text
2025-09-01 through 2025-09-30
```

* Training rows: 38,477
* Holdout rows: 4,670

## Fold 3

Training:

```text
2025-01-01 through 2025-09-30
```

Validation:

```text
2025-10-01 through 2025-10-31
```

* Training rows: 43,147
* Holdout rows: 4,853

For every fold:

1. the chronological split was created first
2. preprocessing statistics were fitted on the training fold only
3. those statistics were applied to the future holdout
4. feature engineering was performed
5. the model was trained
6. the future month was evaluated

This prevents preprocessing leakage.

---

# Model Selection

## Why CatBoost?

CatBoost was selected because the problem is structured tabular regression containing:

* nonlinear distance/rate relationships
* geographic variables
* temporal effects
* interactions
* categorical equipment information
* moderate missingness
* potentially complex feature relationships

Tree-based gradient boosting is well suited to these characteristics.

CatBoost additionally supports categorical features directly and performs well on medium-sized tabular datasets.

---

# Hyperparameter Tuning

After Feature Set B was selected, six CatBoost configurations were evaluated using the same three chronological folds.

The best configuration was:

```text
T4_depth8_regularized
```

with:

| Parameter              | Selected value |
| ---------------------- | -------------: |
| Depth                  |              8 |
| Learning rate          |          0.025 |
| L2 leaf regularization |           10.0 |
| Random strength        |            1.0 |
| Loss                   |            MAE |
| Random seed            |             42 |

Performance:

| Metric              |      Result |
| ------------------- | ----------: |
| Mean MAE            | **$100.99** |
| Worst-fold MAE      | **$107.46** |
| Mean RMSE           | **$626.25** |
| Mean R²             |  **0.8277** |
| Mean best iteration |  **1556.3** |

MAE was used as the primary optimization criterion because the target distribution contains a small number of extreme freight rates.

Compared with squared-error loss, MAE is less dominated by unusually large observations.

RMSE is still reported because it highlights the effect of large prediction errors.

---

# Target-Anomaly Experiment

After tuning, the selected model was tested under three target-handling strategies.

```text
keep_all
remove_strong_anomalies
cap_extreme_targets
```

The best result was obtained by:

```text
keep_all
```

with:

| Metric         |      Result |
| -------------- | ----------: |
| Mean MAE       | **$100.99** |
| Worst-fold MAE | **$107.46** |
| Mean RMSE      | **$626.25** |
| Mean R²        |  **0.8277** |

This experiment provided evidence that automatically deleting or modifying extreme freight-rate observations would not improve forward generalization.

All 48,000 labeled observations were therefore retained.

---

# Final Model

The final CatBoost model was trained on all:

```text
48,000 labeled observations
```

using:

```text
Feature Set B
```

and the tuned configuration.

The number of final boosting iterations was based on the mean best iteration observed during chronological validation.

Final training used approximately:

```text
1,557 boosting iterations
```

Because all labeled observations were used for final training, there was no holdout set at this stage.

The final model is stored as:

```text
models/final_catboost_model.cbm
```

---

# Saved Reproducibility Artifacts

The following files are saved together with the final model:

```text
models/
├── final_catboost_model.cbm
├── preprocessing_stats.json
├── feature_metadata.json
└── final_model_config.json
```

### `preprocessing_stats.json`

Contains training-derived:

* equipment-specific weight medians
* global weight median
* market-index median

### `feature_metadata.json`

Contains:

* selected feature-set name
* exact feature options
* exact feature-column order
* categorical feature definitions

### `final_model_config.json`

Contains:

* model type
* hyperparameters
* boosting iterations
* random seed
* target-handling strategy
* selected feature set

This ensures that inference uses exactly the same transformations and feature structure as training.

---

# Validation Predictions

The final model generated predictions for all:

```text
12,000 validation rows
```

The prediction script verifies:

* exactly 12,000 records
* unique load IDs
* matching validation/template IDs
* finite predictions
* positive predictions
* correct column names
* exact required ID sequence:

  * `TE-000001`
  * through
  * `TE-012000`

The final submission file is:

```text
validation_predictions.csv
```

with exactly:

```text
load_id,predicted_rate
```

Example:

```text
TE-000001,889.384721
TE-000002,4958.804645
TE-000003,5156.546813
TE-000004,3932.170567
TE-000005,1852.669623
```

Final validation metrics are calculated by Spotter after submission and are therefore not available locally.

---

# December Predictions

The assessment provides 31 fixed December records for:

```text
Lexington -> Fort Wayne
Distance: 360 miles
Equipment: Dry Van
Weight: 32,000 lb
Dates: 2025-12-01 through 2025-12-31
```

The supplied December file does not contain route coordinates.

Because the selected model uses geographic features, the prediction pipeline derives Lexington and Fort Wayne coordinates from the original labeled training data.

The source file remains unchanged.

The scorer-ready output is:

```text
data/december_chart_inputs.csv
```

Prediction summary:

| Statistic | Predicted rate |
| --------- | -------------: |
| Minimum   |        $842.56 |
| Mean      |        $858.78 |
| Median    |        $857.27 |
| Maximum   |        $889.45 |

---

# Official Scorer Validation

The provided scorer was executed using:

```powershell
python .\score.py `
    --predictions .\validation_predictions.csv `
    --december-predictions .\data\december_chart_inputs.csv
```

Scorer output:

```text
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results\candidate_december.png
Final validation metrics are calculated by Spotter after submission.
```

Therefore, both required prediction artifacts successfully passed the provided structural and integrity checks.

The generated December chart is available at:

```text
scorer_results/candidate_december.png
```

---

# Repository Structure

```text
.
├── score.py
├── requirements.txt
├── README.md
├── validation_predictions.csv
│
├── data/
│   ├── december_chart_inputs.csv
│   └── raw/
│       ├── train-test.csv
│       ├── validation.csv
│       ├── validation-predictions-template.csv
│       └── december-chart-inputs.csv
│
├── src/
│   ├── data_audit.py
│   ├── eda.py
│   ├── anomaly_analysis.py
│   ├── residual_analysis.py
│   ├── preprocessing.py
│   ├── preprocessing_check.py
│   ├── split.py
│   ├── split_check.py
│   ├── features.py
│   ├── features_check.py
│   ├── train_catboost.py
│   ├── catboost_tuning.py
│   ├── catboost_anomaly_test.py
│   ├── train_final_model.py
│   ├── predict_validation.py
│   └── predict_december.py
│
├── models/
│   ├── final_catboost_model.cbm
│   ├── preprocessing_stats.json
│   ├── feature_metadata.json
│   └── final_model_config.json
│
├── reports/
│   ├── figures/
│   └── tables/
│
└── scorer_results/
    └── candidate_december.png
```

---

# Environment Setup

## 1. Create a virtual environment

Example:

```powershell
python -m venv .venv
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Dependencies:

```text
matplotlib>=3.8,<4
numpy>=1.26,<3
pandas>=2.0,<3
scikit-learn>=1.4,<2
catboost>=1.2,<2
```

---

# Reproducing the Analysis

Run commands from the repository root.

## Data audit

```powershell
python .\src\data_audit.py
```

## Exploratory data analysis

```powershell
python .\src\eda.py
```

## Detailed anomaly investigation

```powershell
python .\src\anomaly_analysis.py
```

## Residual anomaly diagnostics

```powershell
python .\src\residual_analysis.py
```

## Preprocessing verification

```powershell
python .\src\preprocessing_check.py
```

## Chronological split verification

```powershell
python .\src\split_check.py
```

## Feature-engineering verification

```powershell
python .\src\features_check.py
```

---

# Reproducing Model Development

## Feature-set comparison

```powershell
python .\src\train_catboost.py
```

## CatBoost hyperparameter tuning

```powershell
python .\src\catboost_tuning.py
```

## Target-anomaly strategy comparison

```powershell
python .\src\catboost_anomaly_test.py
```

## Train final model

```powershell
python .\src\train_final_model.py
```

---

# Reproducing Final Predictions

## Validation predictions

```powershell
python .\src\predict_validation.py
```

This generates:

```text
validation_predictions.csv
```

## December predictions

```powershell
python .\src\predict_december.py
```

This generates:

```text
data/december_chart_inputs.csv
```

---

# Run Official Scorer

```powershell
python .\score.py `
    --predictions .\validation_predictions.csv `
    --december-predictions .\data\december_chart_inputs.csv
```

Expected structural-validation output:

```text
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results\candidate_december.png
Final validation metrics are calculated by Spotter after submission.
```

---

# Key Modeling Decisions

| Decision                                             | Rationale                                                                           |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Chronological validation instead of random splitting | Mimics prediction of genuinely future freight loads and avoids temporal leakage     |
| Convert negative weights to absolute values          | Their magnitudes match valid weights and indicate likely sign-entry errors          |
| Training-derived median imputation                   | Prevents information leakage from future data                                       |
| Preserve missing/negative flags                      | Allows the model to learn whether data-quality conditions themselves contain signal |
| Use coordinates and geographic features              | Supports generalization to unseen pickup/delivery cities                            |
| Test market signals rather than assume usefulness    | Forward validation showed they reduced generalization performance                   |
| Exclude raw city categories from final feature set   | Coordinate-based representation generalized better to unseen locations              |
| CatBoost                                             | Strong fit for nonlinear structured tabular regression with categorical information |
| MAE loss                                             | More robust to extreme freight-rate observations                                    |
| Expanding-window folds                               | Reproduces realistic train-on-past, predict-future behavior                         |
| Early stopping during tuning                         | Identifies effective boosting length while limiting unnecessary training            |
| Keep all target observations                         | Empirical anomaly-strategy testing showed removal/capping did not improve MAE       |
| Save model + preprocessing + feature metadata        | Ensures reproducible inference                                                      |
| Merge validation predictions by `load_id`            | Avoids dependence on accidental row ordering                                        |
| Keep raw assessment files unchanged                  | Preserves source data integrity                                                     |

---

# Main Result

The selected model achieved the following internal forward-validation performance:

```text
Mean MAE:       $100.99
Worst-fold MAE: $107.46
Mean RMSE:      $626.25
Mean R²:        0.8277
```

These results are based only on the labeled January–October development data.

The final November–December validation target values are hidden, so official final metrics are calculated by Spotter after submission.

---

# Limitations

Several limitations should be noted:

1. The official validation labels are unavailable locally.
2. A small number of extreme target observations increase RMSE substantially even though typical absolute errors are much smaller.
3. The December input does not contain the complete feature schema used during model development, requiring coordinates to be recovered from training data.
4. Future freight markets may experience distribution shifts not represented in January–October 2025.
5. Hyperparameter exploration was deliberately targeted rather than exhaustive to balance model quality, runtime, and reproducibility.

---

# Deliverables

The assessment submission includes:

* `validation_predictions.csv`
* completed December predictions
* trained CatBoost model
* preprocessing and feature metadata
* source code
* EDA figures and analysis tables
* scorer-generated December chart
* assessment report
* GitHub repository
* Loom walkthrough

---

## Author

**Muhammad Kamran**

Machine Learning Engineer Recruitment Assessment

```

After saving it, do **not do anything else yet**.

Tell me **saved**, and our next single step will be to verify how the README renders before we initialize/push the Git repository. 
```
