"""Streamlit UI: upload transactions, predict fraud, view model metrics."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import config  # noqa: E402
from src.evaluate import compute_metrics, plot_confusion_matrix  # noqa: E402
from src.predict import available_models, load_model, predict_df  # noqa: E402

st.set_page_config(page_title="Transaction Fraud Detector", page_icon="🛡️", layout="wide")
st.title("🛡️ Transaction Fraud Detector")


@st.cache_resource
def get_model(name: str):
    return load_model(name)


models = available_models()
if not models:
    st.error("No trained models found in `models/`. Run `python -m src.train` first (see README).")
    st.stop()

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", models, format_func=lambda n: config.MODEL_LABELS[n])
    threshold = st.slider("Fraud threshold", 0.05, 0.95, 0.50, 0.05,
                          help="Transactions with fraud probability >= threshold are flagged.")
    st.caption("Changing the threshold trades precision against recall.")

tab_predict, tab_metrics = st.tabs(["🔍 Predict", "📊 Model metrics"])

with tab_predict:
    st.write(
        "Upload a CSV with columns `Time`, `V1`–`V28`, `Amount` "
        "(same schema as the Kaggle Credit Card Fraud dataset). "
        "An optional `Class` column (0/1) enables an accuracy check."
    )
    uploaded = st.file_uploader("Transaction CSV", type=["csv"])
    use_sample = st.checkbox("Use the bundled sample file instead (synthetic data)")

    df = None
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    elif use_sample and config.SAMPLE_DATA_PATH.exists():
        df = pd.read_csv(config.SAMPLE_DATA_PATH)
        st.info("The sample file is synthetic. Predictions on it only demonstrate the workflow.")

    if df is not None:
        try:
            result = predict_df(df, get_model(model_name), threshold)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        flagged = int(result["prediction"].sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Transactions", f"{len(result):,}")
        c2.metric("Flagged as fraud", f"{flagged:,}")
        c3.metric("Flagged share", f"{flagged / len(result):.2%}")

        left, right = st.columns([3, 2])
        with left:
            only_flagged = st.checkbox("Show only flagged transactions", value=flagged > 0)
            view = result[result["prediction"] == 1] if only_flagged else result
            view = view.sort_values("fraud_probability", ascending=False)
            st.dataframe(view, use_container_width=True, height=380)
            st.download_button("Download all predictions (CSV)",
                               result.to_csv(index=False).encode(), "predictions.csv", "text/csv")
        with right:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.hist(result["fraud_probability"], bins=40)
            ax.axvline(threshold, color="red", linestyle="--", label="threshold")
            ax.set_yscale("log")
            ax.set_xlabel("Predicted fraud probability")
            ax.set_ylabel("Transactions (log)")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)

        if config.TARGET in result.columns and result[config.TARGET].isin([0, 1]).all():
            st.subheader("Performance on uploaded labels")
            m = compute_metrics(result[config.TARGET], result["fraud_probability"], threshold)
            k = st.columns(5)
            k[0].metric("Precision", f"{m['precision']:.3f}")
            k[1].metric("Recall", f"{m['recall']:.3f}")
            k[2].metric("F1", f"{m['f1']:.3f}")
            k[3].metric("ROC-AUC", "n/a" if m["roc_auc"] is None else f"{m['roc_auc']:.3f}")
            k[4].metric("PR-AUC", "n/a" if m["pr_auc"] is None else f"{m['pr_auc']:.3f}")
            st.pyplot(plot_confusion_matrix(m, config.MODEL_LABELS[model_name]))
            st.caption("If the uploaded rows were part of the training data, these numbers are optimistic.")

with tab_metrics:
    if not config.METRICS_PATH.exists():
        st.warning("No `reports/metrics.json` found. Run `python -m src.train`.")
    else:
        report = json.loads(config.METRICS_PATH.read_text())
        split = report["split"]
        st.write(
            f"**Data:** `{report['data_path']}` · **Imbalance handling:** `{report['imbalance_strategy']}` · "
            f"**Test set:** {split['test_rows']:,} rows ({split['test_fraud']} fraud) · "
            f"**Threshold:** {report['threshold']}"
        )
        table = pd.DataFrame(report["metrics"]).T[
            ["precision", "recall", "f1", "roc_auc", "pr_auc", "tp", "fp", "fn", "tn"]
        ]
        st.dataframe(table.style.format({c: "{:.4f}" for c in ["precision", "recall", "f1", "roc_auc", "pr_auc"]}))

        fig_dir = config.FIGURES_DIR
        for title, files in [
            ("Performance charts", ["metric_comparison.png", "roc_curves.png", "pr_curves.png"]),
            ("Confusion matrices", [f"confusion_matrix_{n}.png" for n in config.MODEL_LABELS]),
            ("SHAP explanations (XGBoost)", ["shap_summary.png", "shap_importance.png"]),
        ]:
            present = [fig_dir / f for f in files if (fig_dir / f).exists()]
            if present:
                st.subheader(title)
                cols = st.columns(len(present))
                for col, path in zip(cols, present):
                    col.image(str(path), use_container_width=True)
