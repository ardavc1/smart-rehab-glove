"""Anlık tahmin katmanı."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import FEATURE_NAMES, features_from_buffer

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

STATE_LABELS_TR = {
    "rest": "Dinlenme",
    "hand_opening": "El Açma",
    "hand_closing": "El Kapama",
    "grip_hold": "Kavrama Tutma",
    "release": "Bırakma",
    "fatigue_attempt": "Yorgunluk Denemesi",
}

INTERP_LABELS_TR = {
    "no_exercise": "Egzersiz Yok",
    "movement_phase": "Hareket Fazı",
    "successful_grip": "Başarılı Kavrama",
    "high_effort_low_motion": "Yüksek Efor / Düşük Hareket",
}


class Predictor:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.movement_clf = joblib.load(models_dir / "movement_classifier.joblib")
        self.interp_clf = joblib.load(models_dir / "interpretation_classifier.joblib")
        meta = joblib.load(models_dir / "model_meta.joblib")
        self.feature_names = meta.get("feature_names", FEATURE_NAMES)

    def predict_from_buffer(self, buffer: list[dict]) -> dict | None:
        features = features_from_buffer(buffer)
        if features is None:
            return None

        X = pd.DataFrame([features], columns=self.feature_names)
        state = self.movement_clf.predict(X)[0]
        interp = self.interp_clf.predict(X)[0]
        state_proba = self.movement_clf.predict_proba(X)[0]
        interp_proba = self.interp_clf.predict_proba(X)[0]

        return {
            "exercise_state": state,
            "exercise_state_tr": STATE_LABELS_TR.get(state, state),
            "interpretation": interp,
            "interpretation_tr": INTERP_LABELS_TR.get(interp, interp),
            "state_confidence": float(state_proba.max()),
            "interp_confidence": float(interp_proba.max()),
            "features": dict(zip(self.feature_names, features)),
        }
