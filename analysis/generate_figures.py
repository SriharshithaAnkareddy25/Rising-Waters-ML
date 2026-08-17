from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from src.config import FIGURES_DIR, METADATA_PATH, MODEL_FEATURES, MODEL_PATH, RESULTS_DIR
from src.data import load_data


def save(name):
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / name, dpi=160, bbox_inches="tight")
    plt.close()


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_data()

    counts = frame["flood"].value_counts().sort_index()
    plt.figure(figsize=(5, 4))
    bars = plt.bar(["No flood (0)", "Flood (1)"], counts.values, color=["#4472C4", "#ED7D31"])
    for bar, count in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, count + 1, str(count), ha="center")
    plt.ylabel("Rows")
    plt.title("Target class distribution")
    save("class_distribution.png")

    features = ["Temp", "Humidity", "Cloud Cover", "ANNUAL", "Jan-Feb", "Mar-May", "Jun-Sep", "Oct-Dec", "avgjune", "sub"]
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    for ax, feature in zip(axes.ravel(), features):
        ax.hist(frame[feature], bins=min(15, frame[feature].nunique()), color="#4472C4", alpha=0.8, edgecolor="white")
        ax.set_title(feature)
    fig.suptitle("Feature distributions", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_distributions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    corr = frame.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    image = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr)), corr.columns, rotation=60, ha="right")
    plt.yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    plt.title("Pearson correlation matrix")
    save("correlation_matrix.png")

    predictions = pd.read_csv(RESULTS_DIR / "test_predictions.csv")
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    selected = metadata["model_name"]
    actual = predictions["actual"]
    selected_predictions = predictions[f"prediction__{selected}"]
    ConfusionMatrixDisplay.from_predictions(actual, selected_predictions, display_labels=["No flood", "Flood"], cmap="Blues")
    plt.title(f"Untouched-test confusion matrix\n{selected}")
    save("confusion_matrix.png")

    plt.figure(figsize=(7, 6))
    for name in ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]:
        RocCurveDisplay.from_predictions(actual, predictions[f"probability__{name}"], name=name, ax=plt.gca())
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.title("ROC curves on untouched test set")
    save("roc_curve.png")

    model = joblib.load(MODEL_PATH)
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
        label = "Model feature importance"
    elif hasattr(estimator, "coef_"):
        importance = estimator.coef_[0]
        label = "Standardized logistic coefficient"
    else:
        importance = np.zeros(len(MODEL_FEATURES))
        label = "Feature contribution unavailable"
    order = np.argsort(np.abs(importance))
    plt.figure(figsize=(8, 5))
    plt.barh(np.array(MODEL_FEATURES)[order], np.array(importance)[order], color="#4472C4")
    plt.xlabel(label)
    plt.title(f"Selected-model interpretation\n{selected}")
    save("feature_importance.png")

    # Logistic coefficients are included regardless of which model is selected.
    metrics = json.loads((RESULTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    from src.modeling import model_candidates
    from src.config import RANDOM_STATE, TEST_SIZE
    from sklearn.model_selection import train_test_split
    X = frame[MODEL_FEATURES]
    y = frame["flood"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    logistic = model_candidates(ratio)["Logistic Regression"].fit(X_train, y_train)
    coef = logistic.named_steps["model"].coef_[0]
    order = np.argsort(np.abs(coef))
    plt.figure(figsize=(8, 5))
    plt.barh(np.array(MODEL_FEATURES)[order], coef[order], color=np.where(coef[order] >= 0, "#ED7D31", "#4472C4"))
    plt.xlabel("Coefficient after standardization")
    plt.title("Logistic Regression coefficients")
    save("logistic_coefficients.png")

    print(f"Generated figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
