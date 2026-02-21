# Credit Card Fraud Detection (End-to-End)

A production-style machine learning project to detect fraudulent credit card transactions on a highly imbalanced dataset.

## Business Goal
Detect as many fraudulent transactions as possible (high recall) while minimizing false alarms to avoid blocking legitimate customer payments.

## Dataset
- Source: mlg-ulb/creditcardfraud
- Size: 284,807 transactions
- Target: `Class` (0 = Normal, 1 = Fraud)
- Extreme imbalance: ~0.17% fraud

## Key Steps
1. **EDA & Problem Framing**
   - Confirmed extreme class imbalance
   - Analyzed `Amount` distribution (mean vs median due to outliers)
   - Feature engineering from time: `Hour = (Time // 3600) % 24`

2. **Preprocessing**
   - Created `Hour`, dropped raw `Time`
   - Scaled `Amount` (fit on train only to avoid leakage)
   - Stratified train/test split

3. **Modeling**
   - Baseline: Logistic Regression with `class_weight="balanced"`
   - Compared: Random Forest, HistGradientBoosting
   - Performed **threshold tuning** for business-driven decision making

## Results (Test Set)
### Logistic Regression (baseline)
- Strong recall (~0.91) but high false positives (~1500)

### Random Forest (final model)
- **PR-AUC:** ~0.86
- **Precision:** ~0.96
- **Recall:** ~0.76
- **False Positives:** reduced to ~3

**Final choice:** Random Forest — best balance between fraud detection and customer experience (near-zero false alarms).

## Visuals
- Precision-Recall Curve: `images/pr_curve.png`
- ROC Curve: `images/roc_curve.png`
- Feature Importance: `images/feature_importance.png`

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
