# Premium Prediction

A machine learning pipeline for predicting insurance **pure premiums** using a two-stage frequency-severity model, built on the French Motor Third-Party Liability (freMTPL2) claims dataset.

## Overview

Insurance pricing traditionally splits risk into two components:

- **Frequency** — the probability that a policyholder files a claim
- **Severity** — the expected claim amount, given that a claim occurs

This project models both components separately and combines them into a **pure premium** estimate (`Expected Frequency × Expected Severity`), which represents the expected cost of claims per policy.

## Data Pipeline

1. **Raw data**: `freMTPL2freq.csv` (policy/claim frequency data) and `freMTPL2sev.csv` (claim severity data) are loaded and pushed into a PostgreSQL database.
2. **SQL join**: A `Claim_Join` view (see [`ClaimDatabase.sql`](./ClaimDatabase.sql)) aggregates total claim amount per policy from `ClaimSev` and left-joins it onto `ClaimFreq`, producing one row per policy with a `ClaimAmount` column (0 if no claim).
3. **Feature engineering**:
   - One-hot encoding of `Region`, `VehBrand`, and `VehGas`
   - Ordinal encoding of `Area` (A–D → 0–3)
   - Claim severity capped at the 95th percentile to reduce the influence of extreme outliers

## Exploratory Analysis

- Distribution of claim amounts
- Claim frequency by vehicle brand (highest: **B12**)

  ![Claim Frequency by Vehicle Brand](images/claim_freq_by_vehicle_brand.png)

- Average claim severity by region (highest: **Rhône-Alpes**)

  ![Average Claim Severity by Region](images/claim_severity_by_region.png)

- Claim frequency by region

  ![Claim Frequency by Region](images/claim_frequency_by_region.png)

## Modeling

Two Random Forest models form the pricing pipeline:

| Model | Type | Target | Purpose |
|---|---|---|---|
| `rf_freq` | `RandomForestClassifier` | `Claim_Status` (binary) | Predicts probability a policy has a claim |
| `rf_sev` | `RandomForestRegressor` | `ClaimAmount` (claims > 0 only) | Predicts claim severity, trained only on policies with claims |

**Pure Premium** = `freq_pred_prob × sev_pred`

## Results

- Evaluated using **MAE** and **RMSE** against actual claim amounts on the test set
- Predicted total premium compared against total actual claims, with an adjustment ratio applied to calibrate overall premium levels
- Feature importance analysis identifies the top risk factors driving claim probability — **Exposure**, **DrivAge**, and **Density** rank highest

  ![Top 10 Factors Influencing Claim Probability](images/feature_importance.png)

## What I Learned

- **Frequency-severity modeling**: How actuarial pricing splits risk into "how often" and "how much," and why modeling these separately (rather than predicting claim cost directly) produces more stable, interpretable results — especially with highly imbalanced, zero-inflated claims data.
- **SQL + Python integration**: Writing SQL views to aggregate and join relational data (via SQLAlchemy/PostgreSQL) before pulling it into pandas, instead of doing all transformations in-memory.
- **Handling imbalanced classification**: Using `class_weight='balanced'` in the Random Forest classifier to address the fact that most policies have no claim.
- **Outlier treatment**: Capping claim severity at the 95th percentile to prevent a small number of extreme claims from distorting the severity model.
- **Feature engineering choices**: Deciding between one-hot encoding (nominal categories like Region/VehBrand) and ordinal encoding (Area, which has a natural order), and seeing how that choice affects model input.
- **Model evaluation beyond accuracy**: Using MAE/RMSE to judge severity predictions, and comparing aggregate predicted premium vs. actual claims to sanity-check the model at a portfolio level, not just per-row error.
- **Feature importance for explainability**: Extracting and visualizing which features (Exposure, DrivAge, Density) actually drive claim probability — important in insurance, where models often need to be explainable to regulators/stakeholders, not just accurate.



- **Python**: pandas, numpy, scikit-learn, matplotlib
- **Database**: PostgreSQL (via SQLAlchemy)
- **Modeling**: Random Forest (classification + regression)

## Project Structure

```
├── ClaimDatabase.sql        # SQL view joining frequency and severity claim tables
├── Claim_Prediciton.ipynb   # Full analysis and modeling notebook
├── images/                  # Exported charts embedded in this README
└── README.md
```

## Setup

1. Load `freMTPL2freq.csv` and `freMTPL2sev.csv` into a PostgreSQL database
2. Run `ClaimDatabase.sql` to create the `Claim_Join` view
3. Run `Claim_Prediciton.ipynb` to reproduce the encoding, EDA, and modeling pipeline

## Notes

- This is a portfolio/learning project based on a public dataset (freMTPL2) and is not intended for production actuarial use.
