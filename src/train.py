from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    make_scorer,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split

from .config import (
    CV_FOLDS,
    DATA_PATH,
    METADATA_PATH,
    MODEL_FEATURES,
    MODEL_PATH,
    RANDOM_STATE,
    RESULTS_DIR,
    TEST_SIZE,
)
from .data import load_data, model_xy
from .modeling import TUNING_GRIDS, model_candidates

SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}


def scalar_metrics(y_true, predictions, probabilities):
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }


def timed_inference(estimator, frame, repeats=200):
    start = time.perf_counter()
    for _ in range(repeats):
        estimator.predict_proba(frame)
    return (time.perf_counter() - start) * 1000 / (repeats * len(frame))


def main(data_path=DATA_PATH):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    frame = load_data(data_path)
    X, y = model_xy(frame)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    candidates = model_candidates(ratio)

    results = {}
    fitted = {}
    test_predictions = pd.DataFrame({"row_index": X_test.index, "actual": y_test.values})

    for name, estimator in candidates.items():
        cv_result = cross_validate(
            estimator, X_train, y_train, scoring=SCORING, cv=cv, n_jobs=1, return_train_score=False
        )
        start = time.perf_counter()
        estimator.fit(X_train, y_train)
        training_seconds = time.perf_counter() - start
        predictions = estimator.predict(X_test)
        probabilities = estimator.predict_proba(X_test)[:, 1]
        fitted[name] = estimator
        test_predictions[f"probability__{name}"] = probabilities
        test_predictions[f"prediction__{name}"] = predictions
        results[name] = {
            "kind": "untuned",
            "cv": {
                metric: {
                    "mean": float(np.mean(cv_result[f"test_{metric}"])),
                    "std": float(np.std(cv_result[f"test_{metric}"])),
                    "fold_values": [float(v) for v in cv_result[f"test_{metric}"]],
                }
                for metric in SCORING
            },
            "test": scalar_metrics(y_test, predictions, probabilities),
            "training_seconds": float(training_seconds),
            "inference_ms_per_row": float(timed_inference(estimator, X_test)),
            "parameters": estimator.get_params(deep=False),
        }

    eligible = [name for name in TUNING_GRIDS if name in candidates]
    tune_names = sorted(
        eligible,
        key=lambda name: (
            results[name]["cv"]["recall"]["mean"],
            results[name]["cv"]["f1"]["mean"],
            results[name]["cv"]["roc_auc"]["mean"],
        ),
        reverse=True,
    )[:2]

    tuned_estimators = {}
    for name in tune_names:
        search = GridSearchCV(
            clone(candidates[name]),
            TUNING_GRIDS[name],
            scoring=SCORING,
            refit="f1",
            cv=cv,
            n_jobs=1,
            return_train_score=False,
        )
        start = time.perf_counter()
        search.fit(X_train, y_train)
        training_seconds = time.perf_counter() - start
        tuned_name = f"{name} (tuned)"
        estimator = search.best_estimator_
        predictions = estimator.predict(X_test)
        probabilities = estimator.predict_proba(X_test)[:, 1]
        best = search.best_index_
        tuned_estimators[tuned_name] = estimator
        test_predictions[f"probability__{tuned_name}"] = probabilities
        test_predictions[f"prediction__{tuned_name}"] = predictions
        results[tuned_name] = {
            "kind": "tuned",
            "source_model": name,
            "cv": {
                metric: {
                    "mean": float(search.cv_results_[f"mean_test_{metric}"][best]),
                    "std": float(search.cv_results_[f"std_test_{metric}"][best]),
                    "fold_values": [
                        float(search.cv_results_[f"split{i}_test_{metric}"][best])
                        for i in range(CV_FOLDS)
                    ],
                }
                for metric in SCORING
            },
            "test": scalar_metrics(y_test, predictions, probabilities),
            "training_seconds": float(training_seconds),
            "inference_ms_per_row": float(timed_inference(estimator, X_test)),
            "best_parameters": search.best_params_,
            "search_space": TUNING_GRIDS[name],
            "candidate_count": int(len(search.cv_results_["params"])),
        }

    selectable = [name for name in results if name != "Dummy majority"]
    selected_name = max(
        selectable,
        key=lambda name: (
            results[name]["cv"]["recall"]["mean"],
            results[name]["cv"]["f1"]["mean"],
            results[name]["cv"]["roc_auc"]["mean"],
            -results[name]["inference_ms_per_row"],
        ),
    )
    selected = tuned_estimators.get(selected_name, fitted.get(selected_name))
    joblib.dump(selected, MODEL_PATH)

    comparison = []
    for name, item in results.items():
        comparison.append({
            "model": name,
            **{f"cv_{m}_mean": item["cv"][m]["mean"] for m in SCORING},
            **{f"cv_{m}_std": item["cv"][m]["std"] for m in SCORING},
            **{f"test_{m}": item["test"][m] for m in SCORING},
            "training_seconds": item["training_seconds"],
            "inference_ms_per_row": item["inference_ms_per_row"],
        })
    pd.DataFrame(comparison).to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    test_predictions.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

    audit = {
        "rows": int(len(frame)),
        "features_original": int(len(frame.columns) - 1),
        "features_modeled": len(MODEL_FEATURES),
        "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "missing_values": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "annual_seasonal_sum_max_absolute_difference": float(
            (frame["ANNUAL"] - frame[["Jan-Feb", "Mar-May", "Jun-Sep", "Oct-Dec"]].sum(axis=1)).abs().max()
        ),
        "jun_sep_2400_rule_accuracy": float(accuracy_score(y, frame["Jun-Sep"] > 2400)),
    }
    payload = {
        "methodology": {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "stratified_holdout": True,
            "cv": f"{CV_FOLDS}-fold StratifiedKFold on training partition only",
            "selection_metric": "mean cross-validation recall; F1 then ROC-AUC as tie-breakers",
            "test_set_used_for_selection": False,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "train_class_counts": {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
            "test_class_counts": {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()},
        },
        "audit": audit,
        "excluded_features": {
            "ANNUAL": "Near-exact sum of seasonal rainfall and unavailable until year end.",
            "Jun-Sep": "A >2400 mm threshold perfectly reproduces the provided target; label provenance is undocumented.",
        },
        "tuned_models": tune_names,
        "selected_model": selected_name,
        "results": results,
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    metadata = {
        "model_name": selected_name,
        "feature_names": MODEL_FEATURES,
        "positive_class": 1,
        "decision_threshold": 0.5,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "python_version": platform.python_version(),
        "test_metrics": results[selected_name]["test"],
        "warning": "Research prototype; probabilities are not calibrated flood-warning probabilities.",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"selected_model": selected_name, "test": results[selected_name]["test"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the reproducible Rising Waters experiment")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    args = parser.parse_args()
    main(args.data)
