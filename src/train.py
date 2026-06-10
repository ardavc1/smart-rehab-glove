"""Model eğitimi CLI."""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.features import (
    DEFAULT_DATA_PATH,
    FEATURE_NAMES,
    build_feature_dataset,
    load_raw_data,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def train_models(data_path: Path = DEFAULT_DATA_PATH) -> dict:
    df = load_raw_data(data_path)
    X, y_state, y_interp = build_feature_dataset(df)

    X_train, X_test, ys_train, ys_test = train_test_split(
        X, y_state, test_size=0.2, random_state=42, stratify=y_state
    )
    _, _, yi_train, yi_test = train_test_split(
        X, y_interp, test_size=0.2, random_state=42, stratify=y_interp
    )

    movement_clf = RandomForestClassifier(
        n_estimators=100, max_depth=12, random_state=42, class_weight="balanced"
    )
    movement_clf.fit(X_train, ys_train)

    interp_clf = RandomForestClassifier(
        n_estimators=100, max_depth=12, random_state=42, class_weight="balanced"
    )
    interp_clf.fit(X_train, yi_train)

    ys_pred = movement_clf.predict(X_test)
    yi_pred = interp_clf.predict(X_test)

    print("=== Hareket Sınıflandırıcı ===")
    print(classification_report(ys_test, ys_pred, zero_division=0))
    print("=== Yorum Sınıflandırıcı ===")
    print(classification_report(yi_test, yi_pred, zero_division=0))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    movement_path = MODELS_DIR / "movement_classifier.joblib"
    interp_path = MODELS_DIR / "interpretation_classifier.joblib"
    meta_path = MODELS_DIR / "model_meta.joblib"

    joblib.dump(movement_clf, movement_path)
    joblib.dump(interp_clf, interp_path)
    joblib.dump({"feature_names": FEATURE_NAMES}, meta_path)

    print(f"Modeller kaydedildi: {MODELS_DIR}")
    return {
        "movement_path": movement_path,
        "interpretation_path": interp_path,
        "movement_report": classification_report(ys_test, ys_pred, zero_division=0, output_dict=True),
        "interpretation_report": classification_report(yi_test, yi_pred, zero_division=0, output_dict=True),
    }


if __name__ == "__main__":
    train_models()
