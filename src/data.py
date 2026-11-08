"""Data loading, cleaning, exploration and splitting."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from . import config  # noqa: E402

REQUIRED_COLUMNS = config.FEATURE_COLUMNS + [config.TARGET]


def load_data(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Download the Kaggle 'Credit Card Fraud Detection' dataset and place "
            "creditcard.csv in data/ (see data/README.md), or smoke-test with "
            "--data data/sample_transactions.csv (synthetic)."
        )
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def clean_data(df: pd.DataFrame):
    """Drop rows with missing values and exact duplicates.

    Duplicates are removed *before* the train/test split so identical rows can
    never end up on both sides of the split.
    """
    summary = {"rows_raw": int(len(df))}
    df = df.copy()

    na_rows = int(df[REQUIRED_COLUMNS].isna().any(axis=1).sum())
    df = df.dropna(subset=REQUIRED_COLUMNS)
    summary["rows_dropped_missing"] = na_rows

    dups = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    summary["rows_dropped_duplicates"] = dups

    df[config.TARGET] = df[config.TARGET].astype(int)
    labels = set(df[config.TARGET].unique())
    if not labels <= {0, 1}:
        raise ValueError(f"Target column must contain only 0/1, found {sorted(labels)}")

    summary["rows_clean"] = int(len(df))
    summary["fraud_count"] = int(df[config.TARGET].sum())
    summary["fraud_rate"] = float(df[config.TARGET].mean())
    return df, summary


def explore(df: pd.DataFrame, out_dir=config.FIGURES_DIR) -> dict:
    """Save a few exploratory plots and return summary statistics."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y = config.TARGET

    counts = df[y].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(x=["Legitimate (0)", "Fraud (1)"][: len(counts)], y=counts.values, ax=ax)
    ax.set_yscale("log")
    ax.set_title("Class distribution (log scale)")
    ax.set_ylabel("Transactions")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_dir / "eda_class_distribution.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    plot_df = pd.DataFrame({"log1p(Amount)": np.log1p(df["Amount"]), "Class": df[y].map({0: "Legit", 1: "Fraud"})})
    sns.boxplot(data=plot_df, x="Class", y="log1p(Amount)", ax=ax)
    ax.set_title("Transaction amount by class")
    fig.tight_layout()
    fig.savefig(out_dir / "eda_amount_by_class.png", dpi=120)
    plt.close(fig)

    corr = df[config.FEATURE_COLUMNS + [y]].corr()[y].drop(y)
    top = corr.reindex(corr.abs().sort_values(ascending=False).index).head(12)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x=top.values, y=top.index, ax=ax)
    ax.set_title("Top feature correlations with Class")
    ax.set_xlabel("Pearson correlation")
    fig.tight_layout()
    fig.savefig(out_dir / "eda_top_correlations.png", dpi=120)
    plt.close(fig)

    stats = {
        "amount_describe": df["Amount"].describe().round(4).to_dict(),
        "top_correlations_with_class": top.round(4).to_dict(),
    }
    return stats


def split_data(df: pd.DataFrame):
    """Stratified train/test split. SMOTE is applied later, inside the training
    pipeline, so the test set is never resampled."""
    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET]
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
