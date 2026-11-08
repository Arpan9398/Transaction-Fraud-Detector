"""Train, evaluate and compare Logistic Regression and XGBoost.

Usage:
    python -m src.train                          # full Kaggle dataset, SMOTE
    python -m src.train --imbalance class_weight # class weights instead of SMOTE
    python -m src.train --data data/sample_transactions.csv   # synthetic smoke test
"""
import argparse
import platform
import sys
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from . import config
from .data import clean_data, explore, load_data, save_json, split_data
from .evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_metric_comparison,
    plot_pr_curves,
    plot_roc_curves,
)
from .models import IMBALANCE_STRATEGIES, build_pipeline


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(config.DEFAULT_DATA_PATH), help="Path to creditcard.csv")
    p.add_argument("--imbalance", choices=IMBALANCE_STRATEGIES, default="smote",
                   help="How to handle class imbalance (default: smote, applied to training data only)")
    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for reported metrics")
    p.add_argument("--skip-eda", action="store_true")
    p.add_argument("--skip-shap", action="store_true")
    return p.parse_args(argv)


def lib_versions() -> dict:
    out = {"python": platform.python_version()}
    for mod in ["pandas", "numpy", "sklearn", "xgboost", "imblearn", "shap"]:
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def main(argv=None):
    args = parse_args(argv)
    for d in (config.MODELS_DIR, config.REPORTS_DIR, config.FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.data} ...")
    raw = load_data(args.data)
    df, clean_summary = clean_data(raw)
    print(f"Clean data: {clean_summary}")

    eda = {}
    if not args.skip_eda:
        eda = explore(df)

    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {len(X_train):,} rows ({int(y_train.sum())} fraud) | "
          f"Test: {len(X_test):,} rows ({int(y_test.sum())} fraud)")

    n_neg, n_pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    spw = n_neg / max(n_pos, 1)

    probas, results, train_seconds = {}, {}, {}
    for name, label in config.MODEL_LABELS.items():
        print(f"\nTraining {label} (imbalance={args.imbalance}) ...")
        pipe = build_pipeline(name, imbalance=args.imbalance, scale_pos_weight=spw)
        t0 = time.time()
        pipe.fit(X_train, y_train)  # SMOTE (if used) only ever sees X_train / y_train
        train_seconds[label] = round(time.time() - t0, 2)

        proba = pipe.predict_proba(X_test)[:, 1]
        probas[label] = proba
        results[label] = compute_metrics(y_test, proba, args.threshold)
        joblib.dump(pipe, config.model_path(name))

        fig = plot_confusion_matrix(results[label], f"{label} - confusion matrix")
        fig.savefig(config.FIGURES_DIR / f"confusion_matrix_{name}.png", dpi=120)
        fig.clf()

        m = results[label]
        print(f"  precision={m['precision']:.4f} recall={m['recall']:.4f} f1={m['f1']:.4f} "
              f"roc_auc={m['roc_auc']} pr_auc={m['pr_auc']}  "
              f"[TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}]")

    for fn, fname in [(plot_roc_curves, "roc_curves.png"), (plot_pr_curves, "pr_curves.png")]:
        fn(y_test, probas).savefig(config.FIGURES_DIR / fname, dpi=120)
    plot_metric_comparison(results).savefig(config.FIGURES_DIR / "metric_comparison.png", dpi=120)

    # --- Compare fraud predictions side by side ---
    labels = list(config.MODEL_LABELS.values())
    comp = pd.DataFrame({"y_true": y_test.values})
    for lab in labels:
        comp[f"{lab}_proba"] = probas[lab]
        comp[f"{lab}_pred"] = (probas[lab] >= args.threshold).astype(int)
    comp.to_csv(config.REPORTS_DIR / "prediction_comparison.csv", index=False)

    frauds = comp[comp.y_true == 1]
    a, b = (frauds[f"{labels[0]}_pred"] == 1), (frauds[f"{labels[1]}_pred"] == 1)
    agreement = {
        "true_frauds_in_test": int(len(frauds)),
        "caught_by_both": int((a & b).sum()),
        f"caught_only_by_{labels[0]}": int((a & ~b).sum()),
        f"caught_only_by_{labels[1]}": int((~a & b).sum()),
        "missed_by_both": int((~a & ~b).sum()),
        "models_disagree_on_pct_of_all_test_rows": float(
            (comp[f"{labels[0]}_pred"] != comp[f"{labels[1]}_pred"]).mean() * 100
        ),
    }
    print(f"\nFraud-catch comparison on test set: {agreement}")

    # --- SHAP for XGBoost ---
    shap_ranking = None
    if not args.skip_shap:
        try:
            from .explain import explain_xgboost

            xgb_pipe = joblib.load(config.model_path("xgboost"))
            shap_ranking = explain_xgboost(xgb_pipe, X_test).head(10).round(5).to_dict()
            print(f"Top SHAP features: {shap_ranking}")
        except Exception as exc:  # never lose trained models / metrics because of plotting
            print(f"[warn] SHAP step failed and was skipped: {exc!r}", file=sys.stderr)

    save_json({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_path": str(args.data),
        "imbalance_strategy": args.imbalance,
        "threshold": args.threshold,
        "cleaning": clean_summary,
        "split": {
            "train_rows": int(len(X_train)), "train_fraud": int(y_train.sum()),
            "test_rows": int(len(X_test)), "test_fraud": int(y_test.sum()),
            "test_size": config.TEST_SIZE, "random_state": config.RANDOM_STATE, "stratified": True,
        },
        "metrics": results,
        "train_seconds": train_seconds,
        "fraud_catch_comparison": agreement,
        "shap_top_features": shap_ranking,
        "eda": eda,
        "versions": lib_versions(),
    }, config.METRICS_PATH)
    print(f"\nSaved models to {config.MODELS_DIR}/ and reports to {config.REPORTS_DIR}/")


if __name__ == "__main__":
    main()
