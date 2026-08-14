# Financial Crimes & Anomaly Detection Pipeline

Unsupervised fraud detection on real, anonymized credit card transactions using an Isolation Forest — with no fraud labels used during training.

## Overview

Fraud detection is fundamentally an imbalanced, label-scarce problem: confirmed fraud cases are rare, expensive to label, and constantly evolving, so systems that depend entirely on historical labels struggle to catch new fraud patterns. This project builds an **unsupervised anomaly detection model** that flags suspicious transactions purely from structure in the feature space, then evaluates it against ground-truth labels to measure how well "unsupervised anomaly" lines up with "actual fraud."

## Dataset

[Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 real European card transactions from September 2013, of which 492 (0.173%) are confirmed fraud. Features `V1`–`V28` are PCA-transformed for confidentiality; `Time` and `Amount` are the only raw features.

> The raw CSV (~150MB) is not included in this repo — download it from Kaggle and place it in the project root as `creditcard.csv` before running the notebook.

## Approach

1. **Preprocessing** — standard-scale `Time` and `Amount` (the PCA components are already scaled).
2. **Model** — `IsolationForest` (scikit-learn), trained on all transaction features with the labels withheld. Contamination is set to the true fraud rate (0.173%).
3. **Evaluation** — ROC-AUC to measure overall ranking quality, then a precision/recall sweep across thresholds to quantify the *operational* tradeoff a fraud-ops team actually faces: how many transactions need manual review to catch a given percentage of fraud.

## Key Results

- **ROC-AUC: 0.947** despite zero labeled training examples and extreme class imbalance.
- Catching **80% of fraud requires flagging ~18.7x** as many transactions as there are actual frauds (9,206 flagged vs. 492 real fraud cases) — the review-queue cost of raising detection sensitivity.
- Pushing to 90% recall inflates the queue to ~77x, for a 10-point recall gain — a sharply diminishing return that should inform staffing/threshold decisions.

| Target Recall | Flagged | Precision | Review Queue (× fraud count) |
|---|---|---|---|
| 50% | 1,959 | 12.6% | 4.0x |
| 70% | 5,870 | 5.9% | 11.9x |
| 80% | 9,206 | 4.3% | 18.7x |
| 90% | 38,037 | 1.2% | 77.3x |

## Running It

```bash
git clone https://github.com/yashdesai241me320-max/fraud-anomaly-detection.git
cd fraud-anomaly-detection
pip install -r requirements.txt
# download creditcard.csv from the Kaggle link above and place it in this folder
jupyter notebook Financial_Crimes_Anomaly_Detection.ipynb
```

## Possible Extensions

- Compare against a supervised baseline (e.g. XGBoost trained on labels) to quantify the cost of going label-free.
- Try Local Outlier Factor or a deep autoencoder as alternative unsupervised detectors.
- Cost-weight the threshold choice using estimated dollar loss per missed fraud vs. analyst-hour cost per false positive.

## Tech Stack

Python · pandas · scikit-learn · matplotlib · seaborn
