# Transaction Fraud Detector

End-to-end machine-learning project that detects fraudulent credit-card transactions
with **Logistic Regression** and **XGBoost**, handles the extreme class imbalance
(~0.17 % fraud) with **SMOTE** or class weights, explains the model with **SHAP**,
and ships a **Streamlit** app for uploading transactions and scoring them.

## Project structure

```
transaction-fraud-detector/
├── app/streamlit_app.py        # Streamlit UI (predict + metrics tabs)
├── data/
│   ├── README.md               # dataset download instructions
│   └── sample_transactions.csv # SYNTHETIC sample (2,500 rows) for smoke tests
├── scripts/make_sample_data.py # regenerates the synthetic sample
├── src/
│   ├── config.py               # paths, columns, constants
│   ├── data.py                 # load, clean, EDA plots, stratified split
│   ├── models.py               # scaler -> [SMOTE] -> classifier pipelines
│   ├── evaluate.py             # metrics + confusion matrix / ROC / PR charts
│   ├── explain.py              # SHAP for XGBoost
│   ├── predict.py              # inference helpers (used by the app)
│   ├── sample_data.py          # synthetic data generator
│   └── train.py                # training / evaluation entry point
├── tests/test_pipeline.py
├── models/                     # trained models are written here (git-ignored)
├── reports/                    # metrics.json, figures/ (git-ignored)
├── requirements.txt
└── .gitignore
```

## Setup

Requires Python 3.9+.

```bash
cd transaction-fraud-detector
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download the dataset (see `data/README.md`) so that `data/creditcard.csv` exists:

```bash
pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip
```

## Train and evaluate

```bash
python -m src.train                            # SMOTE on the training split (default)
python -m src.train --imbalance class_weight   # class weights instead of SMOTE
python -m src.train --threshold 0.5            # decision threshold for reported metrics
python -m src.train --skip-shap                # skip the SHAP step (faster)
```

Quick pipeline check without the real dataset (synthetic data, numbers are meaningless):

```bash
python -m src.train --data data/sample_transactions.csv
```

Outputs:

| Path | Content |
|------|---------|
| `models/logistic_regression.joblib`, `models/xgboost.joblib` | full fitted pipelines (scaler + classifier) |
| `reports/metrics.json` | precision, recall, F1, ROC-AUC, PR-AUC, confusion-matrix counts, split sizes, library versions |
| `reports/prediction_comparison.csv` | per-transaction probabilities/predictions of both models on the test set |
| `reports/shap_feature_ranking.csv` | mean \|SHAP\| per feature (XGBoost) |
| `reports/figures/*.png` | EDA, confusion matrices, ROC, PR, metric comparison, SHAP plots |

## Run the Streamlit app

```bash
python -m src.train          # once, so models/ exists
streamlit run app/streamlit_app.py
```

Then open http://localhost:8501. Upload a CSV with columns `Time, V1…V28, Amount`
(an optional `Class` column enables precision/recall/F1/ROC-AUC/confusion matrix on your file).
Pick a model and a fraud threshold in the sidebar; the **Model metrics** tab shows the
evaluation produced by `src.train`.

## Tests

```bash
pytest -q
```

## Methodology and leakage prevention

* Exact duplicate rows are dropped **before** the split, so identical transactions cannot
  appear in both train and test.
* Stratified 80/20 split (`random_state=42`) keeps the fraud ratio equal in both sets.
* `Time` and `Amount` are scaled with `RobustScaler`, **fit on the training split only**.
* SMOTE is a step inside an `imblearn` `Pipeline`, so it runs **only during `fit()` on training
  data**; the test set and any data passed to `predict()` are never resampled.
* The test set is used once, for the final metrics. The decision threshold defaults to 0.5
  and is not tuned on the test set; the Streamlit slider is for exploration only.
* Metrics are computed at the stated threshold. Because fraud is rare, **PR-AUC, precision and
  recall matter far more than accuracy** (accuracy is not reported).

## Results

No scores are hard-coded in this README. Run `python -m src.train` on the real dataset and
read `reports/metrics.json` (or the **Model metrics** tab in the app) for the measured values.

## Notes and limitations

* Features `V1…V28` are anonymised PCA components, so SHAP explains them by name only.
* A single random split gives a noisy estimate with only ~100 fraud cases in the test set;
  consider repeated stratified cross-validation for firmer conclusions.
* The dataset is from European card transactions in 2013; real deployments need monitoring,
  drift checks and a cost-based threshold.
