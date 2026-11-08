"""Inference helpers shared by the CLI and the Streamlit app."""
import joblib
import pandas as pd

from . import config


def load_model(name: str):
    path = config.model_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run `python -m src.train` first."
        )
    return joblib.load(path)


def available_models() -> list:
    return [n for n in config.MODEL_LABELS if config.model_path(n).exists()]


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """Return the feature frame (correct column order, numeric) or raise ValueError."""
    missing = [c for c in config.FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Uploaded file is missing {len(missing)} required column(s): {missing}. "
            f"Expected: {config.FEATURE_COLUMNS}"
        )
    X = df[config.FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    bad = int(X.isna().any(axis=1).sum())
    if bad:
        raise ValueError(f"{bad} row(s) contain missing or non-numeric feature values.")
    return X


def predict_df(df: pd.DataFrame, model, threshold: float = 0.5) -> pd.DataFrame:
    X = validate_input(df)
    proba = model.predict_proba(X)[:, 1]
    out = df.copy()
    out["fraud_probability"] = proba
    out["prediction"] = (proba >= threshold).astype(int)
    out["prediction_label"] = out["prediction"].map({0: "Legitimate", 1: "Fraud"})
    return out
