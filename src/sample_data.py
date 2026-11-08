"""Generate a small SYNTHETIC dataset with the same schema as the Kaggle
Credit Card Fraud dataset (Time, V1..V28, Amount, Class).

This is NOT real data. It exists only so the pipeline and the Streamlit app
can be smoke-tested without downloading the real dataset.
"""
import numpy as np
import pandas as pd

from . import config


def generate(n_rows: int = 2500, fraud_rate: float = 0.02, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = int(round(n_rows * fraud_rate))
    n_legit = n_rows - n_fraud

    scales = rng.uniform(0.5, 2.0, 28)
    legit_v = rng.normal(0.0, scales, size=(n_legit, 28))

    shift = np.zeros(28)
    # indices of V1, V3, V4, V10, V11, V12, V14, V17
    for idx, val in {0: -3.0, 2: -4.0, 3: 3.0, 9: -3.5, 10: 3.0, 11: -4.0, 13: -5.0, 16: -3.0}.items():
        shift[idx] = val
    fraud_v = rng.normal(shift, scales * 1.3, size=(n_fraud, 28))

    legit_amt = rng.lognormal(3.5, 1.4, n_legit)
    fraud_amt = rng.lognormal(3.0, 1.8, n_fraud)

    df = pd.DataFrame(np.vstack([legit_v, fraud_v]), columns=config.PCA_FEATURES)
    df.insert(0, "Time", rng.uniform(0, 172792, n_rows))
    df["Amount"] = np.concatenate([legit_amt, fraud_amt])
    df[config.TARGET] = np.concatenate([np.zeros(n_legit, int), np.ones(n_fraud, int)])

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df.round(4)
