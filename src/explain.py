"""SHAP explanations for the XGBoost pipeline."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402


def _shap_values(clf, X_t, feature_names):
    import shap

    try:
        values = shap.TreeExplainer(clf).shap_values(X_t)
        if isinstance(values, list):  # older shap: one array per class
            values = values[1]
        return np.asarray(values)
    except Exception as exc:  # fall back to XGBoost's native SHAP contributions
        print(f"[explain] shap.TreeExplainer failed ({exc!r}); using XGBoost pred_contribs instead.")
        import xgboost as xgb

        dm = xgb.DMatrix(np.asarray(X_t))  # booster was trained on unnamed arrays
        return clf.get_booster().predict(dm, pred_contribs=True)[:, :-1]


def explain_xgboost(pipeline, X_test: pd.DataFrame, out_dir=config.FIGURES_DIR, max_samples: int = 2000):
    import shap

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]
    sample = X_test.sample(min(max_samples, len(X_test)), random_state=config.RANDOM_STATE)
    X_t = prep.transform(sample)
    names = list(prep.get_feature_names_out())

    values = _shap_values(clf, X_t, names)

    plt.figure()
    shap.summary_plot(values, X_t, feature_names=names, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=130, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(values, X_t, feature_names=names, plot_type="bar", show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_importance.png", dpi=130, bbox_inches="tight")
    plt.close()

    ranking = (
        pd.Series(np.abs(values).mean(axis=0), index=names)
        .sort_values(ascending=False)
        .rename("mean_abs_shap")
    )
    ranking.to_csv(config.REPORTS_DIR / "shap_feature_ranking.csv")
    return ranking
