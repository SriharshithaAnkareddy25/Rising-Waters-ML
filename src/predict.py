from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from .config import FEATURE_RANGES, METADATA_PATH, MODEL_FEATURES, MODEL_PATH


class InputValidationError(ValueError):
    pass


def load_artifacts(model_path=MODEL_PATH, metadata_path=METADATA_PATH):
    model = joblib.load(model_path)
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if metadata["feature_names"] != MODEL_FEATURES:
        raise RuntimeError("Saved model metadata does not match the application schema")
    return model, metadata


def validate_payload(payload: dict[str, object]) -> pd.DataFrame:
    missing = [name for name in MODEL_FEATURES if name not in payload]
    unexpected = sorted(set(payload) - set(MODEL_FEATURES))
    if missing or unexpected:
        raise InputValidationError(f"Schema mismatch; missing={missing}, unexpected={unexpected}")

    values = {}
    for name in MODEL_FEATURES:
        try:
            value = float(payload[name])
        except (TypeError, ValueError) as exc:
            raise InputValidationError(f"{name} must be numeric") from exc
        low, high = FEATURE_RANGES[name]
        if not low <= value <= high:
            raise InputValidationError(
                f"{name} must be within the observed dataset range [{low:g}, {high:g}]"
            )
        values[name] = value
    return pd.DataFrame([values], columns=MODEL_FEATURES)


def predict_payload(model, payload: dict[str, object]) -> dict[str, object]:
    frame = validate_payload(payload)
    probability = float(model.predict_proba(frame)[0, 1])
    prediction = int(model.predict(frame)[0])
    return {"prediction": prediction, "probability": probability}
