"""Metrics and plots."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) == 2
    return {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)) if both else None,
        "pr_auc": float(average_precision_score(y_true, proba)) if both else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def plot_confusion_matrix(metrics: dict, title: str = "Confusion matrix"):
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{title}\n(threshold = {metrics['threshold']:.2f})")
    fig.tight_layout()
    return fig


def plot_roc_curves(y_true, probas: dict):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, p):.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_pr_curves(y_true, probas: dict):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, p in probas.items():
        prec, rec, _ = precision_recall_curve(y_true, p)
        ax.plot(rec, prec, label=f"{name} (AP={average_precision_score(y_true, p):.4f})")
    ax.axhline(np.mean(y_true), color="k", linestyle="--", alpha=0.4, label="Baseline")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_metric_comparison(results: dict):
    """results: {model label: metrics dict}"""
    keys = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    labels = list(results)
    x = np.arange(len(keys))
    width = 0.8 / max(len(labels), 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, lab in enumerate(labels):
        vals = [results[lab][k] or 0 for k in keys]
        bars = ax.bar(x + i * width, vals, width, label=lab)
        ax.bar_label(bars, fmt="%.3f", fontsize=7)
    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels(["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"])
    ax.set_ylim(0, 1.1)
    ax.set_title("Model comparison (test set)")
    ax.legend()
    fig.tight_layout()
    return fig
