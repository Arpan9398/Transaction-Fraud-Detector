"""Central configuration: paths, column names and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DEFAULT_DATA_PATH = DATA_DIR / "creditcard.csv"
SAMPLE_DATA_PATH = DATA_DIR / "sample_transactions.csv"
METRICS_PATH = REPORTS_DIR / "metrics.json"

TARGET = "Class"
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
FEATURE_COLUMNS = ["Time"] + PCA_FEATURES + ["Amount"]
SCALED_COLUMNS = ["Time", "Amount"]  # V1..V28 are already PCA-transformed

RANDOM_STATE = 42
TEST_SIZE = 0.2

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "xgboost": "XGBoost",
}


def model_path(name: str) -> Path:
    return MODELS_DIR / f"{name}.joblib"
