"""Regenerate data/sample_transactions.csv (synthetic, for smoke tests only)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.sample_data import generate  # noqa: E402

if __name__ == "__main__":
    df = generate()
    config.DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(config.SAMPLE_DATA_PATH, index=False)
    print(f"Wrote {config.SAMPLE_DATA_PATH} ({len(df)} rows, {int(df[config.TARGET].sum())} fraud)")
