import joblib
import pytest

from src.config import METADATA_PATH, MODEL_FEATURES, MODEL_PATH
from src.data import load_data, model_xy
from src.predict import InputValidationError, load_artifacts, predict_payload, validate_payload


@pytest.fixture
def valid_payload():
    return {
        "Temp": 29,
        "Humidity": 74,
        "Cloud Cover": 36,
        "Jan-Feb": 20.5,
        "Mar-May": 342.0,
        "Oct-Dec": 501.5,
        "avgjune": 211.03,
        "sub": 430.6,
    }


def test_input_schema_and_order(valid_payload):
    frame = validate_payload(valid_payload)
    assert frame.columns.tolist() == MODEL_FEATURES
    assert frame.shape == (1, len(MODEL_FEATURES))


def test_missing_feature_is_rejected(valid_payload):
    valid_payload.pop("Temp")
    with pytest.raises(InputValidationError, match="missing"):
        validate_payload(valid_payload)


def test_unexpected_feature_is_rejected(valid_payload):
    valid_payload["ANNUAL"] = 3000
    with pytest.raises(InputValidationError, match="unexpected"):
        validate_payload(valid_payload)


def test_non_numeric_and_out_of_range_are_rejected(valid_payload):
    valid_payload["Humidity"] = "wet"
    with pytest.raises(InputValidationError, match="numeric"):
        validate_payload(valid_payload)
    valid_payload["Humidity"] = 120
    with pytest.raises(InputValidationError, match="observed dataset range"):
        validate_payload(valid_payload)


def test_preprocessing_output_shape():
    model = joblib.load(MODEL_PATH)
    X, _ = model_xy(load_data())
    transformed = model.named_steps["preprocess"].transform(X.head(3))
    assert transformed.shape == (3, len(MODEL_FEATURES))


def test_model_loading_and_prediction_smoke(valid_payload):
    model, metadata = load_artifacts(MODEL_PATH, METADATA_PATH)
    result = predict_payload(model, valid_payload)
    assert metadata["feature_names"] == MODEL_FEATURES
    assert result["prediction"] in {0, 1}
    assert 0.0 <= result["probability"] <= 1.0
