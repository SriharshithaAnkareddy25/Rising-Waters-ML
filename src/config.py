from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "flood_dataset_raw.xlsx"
MODEL_PATH = ROOT / "models" / "flood_pipeline.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
RESULTS_DIR = ROOT / "reports"
FIGURES_DIR = RESULTS_DIR / "figures"

TARGET = "flood"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

ALL_FEATURES = [
    "Temp", "Humidity", "Cloud Cover", "ANNUAL", "Jan-Feb",
    "Mar-May", "Jun-Sep", "Oct-Dec", "avgjune", "sub",
]

# ANNUAL is a near-exact sum of seasonal rainfall and Jun-Sep perfectly
# separates the provided label. They are excluded from the primary experiment
# because their availability at a genuine forecast time is undocumented.
MODEL_FEATURES = [
    "Temp", "Humidity", "Cloud Cover", "Jan-Feb", "Mar-May",
    "Oct-Dec", "avgjune", "sub",
]

FEATURE_RANGES = {
    "Temp": (28.0, 31.0),
    "Humidity": (70.0, 79.0),
    "Cloud Cover": (30.0, 44.0),
    "Jan-Feb": (0.3, 98.1),
    "Mar-May": (89.9, 915.2),
    "Oct-Dec": (166.6, 823.3),
    "avgjune": (65.6, 366.066667),
    "sub": (34.2, 982.7),
}
