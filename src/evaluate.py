from __future__ import annotations

import json

import pandas as pd

from .config import MODEL_PATH, RESULTS_DIR
from .data import load_data, model_xy
from .predict import load_artifacts


def main():
    model, metadata = load_artifacts()
    frame = load_data()
    X, _ = model_xy(frame)
    print(f"Loaded: {MODEL_PATH}")
    print(f"Model: {metadata['model_name']}")
    print("Saved untouched-test metrics:")
    print(json.dumps(metadata["test_metrics"], indent=2))
    print(f"Full-dataset prediction counts: {pd.Series(model.predict(X)).value_counts().sort_index().to_dict()}")
    print(f"Detailed comparison: {RESULTS_DIR / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
