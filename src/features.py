"""Özellik mühendisliği: sliding window ve sentetik augmentasyon."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ANGLE_COLS = [
    "thumb_angle",
    "index_angle",
    "middle_angle",
    "ring_angle",
    "pinky_angle",
]
SENSOR_COLS = ANGLE_COLS + [
    "grip_force",
    "emg_level",
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
]
LABEL_STATE = "exercise_state"
LABEL_INTERP = "interpretation"
DEFAULT_WINDOW = 10
DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "rehab_glove_sample_300.csv"


def load_raw_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def augment_dataframe(df: pd.DataFrame, copies: int = 5, noise_pct: float = 0.05) -> pd.DataFrame:
    """Sensör sütunlarına gürültü ekleyerek sentetik kopyalar üretir."""
    frames = [df.copy()]
    rng = np.random.default_rng(42)

    for i in range(copies):
        aug = df.copy()
        for col in SENSOR_COLS:
            scale = df[col].std()
            if scale == 0 or np.isnan(scale):
                scale = 1.0
            noise = rng.normal(0, scale * noise_pct, size=len(df))
            aug[col] = aug[col] + noise
        aug["time_ms"] = aug["time_ms"] + (i + 1) * 1_000_000
        frames.append(aug)

    return pd.concat(frames, ignore_index=True)


def _window_slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values))
    return float(np.polyfit(x, values, 1)[0])


def extract_window_features(window: pd.DataFrame) -> dict[str, float]:
    """Son N ölçümden model özellikleri çıkarır."""
    angles = window[ANGLE_COLS].values
    avg_angles = angles.mean(axis=1)
    angle_range = float(angles.max() - angles.min())
    avg_finger_angle = float(avg_angles.mean())
    finger_angle_std = float(avg_angles.std())

    grip_mean = float(window["grip_force"].mean())
    grip_max = float(window["grip_force"].max())
    emg_mean = float(window["emg_level"].mean())
    emg_max = float(window["emg_level"].max())

    gyro_mag = np.sqrt(
        window["gyro_x"] ** 2 + window["gyro_y"] ** 2 + window["gyro_z"] ** 2
    )
    motion_intensity = float(gyro_mag.mean())
    motion_peak = float(gyro_mag.max())

    acc_mag = np.sqrt(
        window["acc_x"] ** 2 + window["acc_y"] ** 2 + window["acc_z"] ** 2
    )
    acc_stability = float(acc_mag.std())

    angle_slope = _window_slope(avg_angles)
    grip_slope = _window_slope(window["grip_force"].values)
    emg_slope = _window_slope(window["emg_level"].values)

    epsilon = 1e-6
    effort_motion_ratio = emg_mean / (angle_range + epsilon)

    return {
        "avg_finger_angle": avg_finger_angle,
        "finger_angle_std": finger_angle_std,
        "angle_range": angle_range,
        "grip_force_mean": grip_mean,
        "grip_force_max": grip_max,
        "emg_mean": emg_mean,
        "emg_max": emg_max,
        "motion_intensity": motion_intensity,
        "motion_peak": motion_peak,
        "acc_stability": acc_stability,
        "angle_slope": angle_slope,
        "grip_slope": grip_slope,
        "emg_slope": emg_slope,
        "effort_motion_ratio": effort_motion_ratio,
    }


FEATURE_NAMES = list(extract_window_features(
    pd.DataFrame({col: [0.0] * DEFAULT_WINDOW for col in SENSOR_COLS})
).keys())


def build_feature_dataset(
    df: pd.DataFrame,
    window_size: int = DEFAULT_WINDOW,
    augment: bool = True,
    augment_copies: int = 5,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Sliding window ile eğitim veri seti oluşturur."""
    source = augment_dataframe(df, copies=augment_copies) if augment else df

    rows: list[dict[str, float]] = []
    states: list[str] = []
    interpretations: list[str] = []

    for start in range(len(source) - window_size + 1):
        window = source.iloc[start : start + window_size]
        feats = extract_window_features(window)
        label_row = source.iloc[start + window_size - 1]
        rows.append(feats)
        states.append(label_row[LABEL_STATE])
        interpretations.append(label_row[LABEL_INTERP])

    X = pd.DataFrame(rows)
    y_state = pd.Series(states, name=LABEL_STATE)
    y_interp = pd.Series(interpretations, name=LABEL_INTERP)
    return X, y_state, y_interp


def features_from_buffer(buffer: list[dict], window_size: int = DEFAULT_WINDOW) -> np.ndarray | None:
    """Canlı simülasyon için buffer'dan özellik vektörü üretir."""
    if len(buffer) < window_size:
        return None
    window = pd.DataFrame(buffer[-window_size:])
    feats = extract_window_features(window)
    return np.array([feats[name] for name in FEATURE_NAMES], dtype=np.float64)
