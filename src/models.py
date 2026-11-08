"""Model pipelines.

Each pipeline is: scale (Time, Amount) -> [SMOTE] -> classifier.
Because SMOTE sits inside an imbalanced-learn Pipeline, it is applied ONLY
during fit() (i.e. to training data) and is skipped at predict() time.
The scaler is likewise fit on training data only.
"""
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

from . import config

IMBALANCE_STRATEGIES = ("smote", "class_weight")


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [("scale", RobustScaler(), config.SCALED_COLUMNS)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def build_pipeline(name: str, imbalance: str = "smote", scale_pos_weight: float = 1.0) -> ImbPipeline:
    if imbalance not in IMBALANCE_STRATEGIES:
        raise ValueError(f"imbalance must be one of {IMBALANCE_STRATEGIES}")
    use_smote = imbalance == "smote"

    if name == "logistic_regression":
        clf = LogisticRegression(
            max_iter=1000,
            class_weight=None if use_smote else "balanced",
            random_state=config.RANDOM_STATE,
        )
    elif name == "xgboost":
        clf = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="aucpr",
            scale_pos_weight=1.0 if use_smote else scale_pos_weight,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model: {name}")

    steps = [("prep", build_preprocessor())]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=config.RANDOM_STATE)))
    steps.append(("clf", clf))
    return ImbPipeline(steps)
