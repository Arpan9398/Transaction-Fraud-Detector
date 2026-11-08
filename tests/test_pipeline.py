import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import clean_data, split_data
from src.evaluate import compute_metrics
from src.predict import validate_input
from src.sample_data import generate


@pytest.fixture(scope="module")
def df():
    return generate(n_rows=1500, fraud_rate=0.04, seed=1)


def test_clean_drops_duplicates(df):
    dup = pd.concat([df, df.iloc[:10]], ignore_index=True)
    cleaned, summary = clean_data(dup)
    assert summary["rows_dropped_duplicates"] == 10
    assert len(cleaned) == len(df)


def test_split_is_stratified(df):
    _, _, y_tr, y_te = split_data(df)
    assert abs(y_tr.mean() - y_te.mean()) < 0.01


def test_validate_input_reports_missing_columns(df):
    with pytest.raises(ValueError, match="missing"):
        validate_input(df.drop(columns=["V5"]))


def test_validate_input_orders_columns(df):
    X = validate_input(df[::-1].reset_index(drop=True).sample(frac=1, axis=1, random_state=0))
    assert list(X.columns) == config.FEATURE_COLUMNS


def test_compute_metrics_values():
    m = compute_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], threshold=0.5)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5
    assert m["roc_auc"] == 0.75


def test_smote_only_resamples_training_data(df):
    pytest.importorskip("xgboost")
    pytest.importorskip("imblearn")
    from src.models import build_pipeline

    X_tr, X_te, y_tr, _ = split_data(df)
    pipe = build_pipeline("logistic_regression", imbalance="smote")
    assert [n for n, _ in pipe.steps] == ["prep", "smote", "clf"]

    pipe.fit(X_tr, y_tr)
    # Predicting never resamples: output length equals the input length exactly.
    assert pipe.predict_proba(X_te).shape == (len(X_te), 2)
    # SMOTE balanced the training set inside fit(); the original X_tr is untouched.
    assert len(X_tr) == len(y_tr)


def test_class_weight_pipeline_has_no_sampler(df):
    pytest.importorskip("xgboost")
    pytest.importorskip("imblearn")
    from src.models import build_pipeline

    pipe = build_pipeline("xgboost", imbalance="class_weight", scale_pos_weight=20.0)
    assert [n for n, _ in pipe.steps] == ["prep", "clf"]
