from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import MODEL_FEATURES, RANDOM_STATE


def preprocessing(scale: bool) -> ColumnTransformer:
    transformer = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [("numeric", transformer, MODEL_FEATURES)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def pipeline(model, *, scale: bool = False) -> Pipeline:
    return Pipeline([("preprocess", preprocessing(scale)), ("model", model)])


def model_candidates(negative_to_positive_ratio: float):
    return {
        "Dummy majority": pipeline(DummyClassifier(strategy="prior")),
        "Logistic Regression": pipeline(
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
            scale=True,
        ),
        "Decision Tree": pipeline(
            DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)
        ),
        "Random Forest": pipeline(
            RandomForestClassifier(
                n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1
            )
        ),
        "XGBoost": pipeline(
            XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                scale_pos_weight=negative_to_positive_ratio,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
        ),
    }


TUNING_GRIDS = {
    "Logistic Regression": {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__class_weight": [None, "balanced"],
    },
    "Random Forest": {
        "model__n_estimators": [200, 500],
        "model__max_depth": [None, 3, 5],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", 0.75],
        "model__class_weight": ["balanced", "balanced_subsample"],
    },
    "XGBoost": {
        "model__n_estimators": [100, 250],
        "model__max_depth": [1, 2, 3],
        "model__learning_rate": [0.03, 0.1],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
    },
}
