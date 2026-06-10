"""Sentetik eldiven verisi üreticisi.

Gerçek donanım olmadığı için demo ve sunum amaçlı gerçekçi sensör verisi üretir.
Her seans 6 fazlı egzersiz yapısını (rest -> hand_opening -> hand_closing ->
grip_hold -> release -> fatigue_attempt) içerir. Hasta arketipleri ve seanslar
arası iyileşme (recovery) modeli ile çoklu seans/ilerleme verisi oluşturulabilir.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ANGLE_COLS = ["thumb_angle", "index_angle", "middle_angle", "ring_angle", "pinky_angle"]
ALL_COLS = [
    "time_ms",
    *ANGLE_COLS,
    "grip_force",
    "emg_level",
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "exercise_state",
    "interpretation",
]

GENERATED_DIR = Path(__file__).resolve().parent.parent / "data" / "generated"


@dataclass
class PatientArchetype:
    key: str
    name: str
    condition: str
    base_recovery: float       # 0-1 başlangıç yetkinliği
    improvement_rate: float    # seans başına iyileşme
    fatigue_tendency: float     # 0-1 yorgunluğa yatkınlık


PATIENT_ARCHETYPES: dict[str, PatientArchetype] = {
    "ahmet": PatientArchetype(
        key="ahmet",
        name="Ahmet Y.",
        condition="İnme sonrası (sol el hemiparezi)",
        base_recovery=0.22,
        improvement_rate=0.075,
        fatigue_tendency=0.72,
    ),
    "elif": PatientArchetype(
        key="elif",
        name="Elif K.",
        condition="Ortopedik (el bileği kırığı sonrası)",
        base_recovery=0.45,
        improvement_rate=0.10,
        fatigue_tendency=0.38,
    ),
    "mehmet": PatientArchetype(
        key="mehmet",
        name="Mehmet D.",
        condition="Nörolojik (periferik sinir hasarı)",
        base_recovery=0.33,
        improvement_rate=0.055,
        fatigue_tendency=0.55,
    ),
}


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _interp(start: float, end: float, n: int) -> np.ndarray:
    return np.linspace(start, end, max(n, 1))


def generate_session(
    capability: float,
    fatigue_tendency: float,
    seed: int = 0,
    session_index: int = 0,
    dt_ms: int = 100,
) -> pd.DataFrame:
    """Tek bir rehabilitasyon seansı için gerçekçi sensör verisi üretir.

    capability: 0-1 arası el fonksiyon yetkinliği (yüksek = daha iyi açı/kuvvet).
    fatigue_tendency: 0-1 arası yorgunluğa yatkınlık.
    """
    rng = np.random.default_rng(seed)
    capability = _clip(capability, 0.05, 1.0)
    fatigue_tendency = _clip(fatigue_tendency, 0.0, 1.0)

    max_angle = 45 + 45 * capability
    max_grip = 25 + 70 * capability
    grip_emg = 380 + 150 * capability
    # Seanslar arası görsel fark: tekrar sayısı ve faz süreleri seed + seans no ile değişir
    cycle_jitter = rng.uniform(0.75, 1.25)
    n_cycles = max(2, int(round((3 + capability * 4) * cycle_jitter)))
    phase_scale = rng.uniform(0.8, 1.2)
    rest_scale = rng.uniform(0.7, 1.3)
    fatigue_boost = 1.0 + max(0, fatigue_tendency - 0.5) * rng.uniform(0.0, 0.4)

    records: list[dict] = []

    def add_phase(
        n: int,
        angle_traj: np.ndarray,
        grip_traj: np.ndarray,
        emg_traj: np.ndarray,
        gyro_scale: float,
        gyro_bias_x: float,
        state: str,
        interp,
        angle_jitter: float = 2.6,
    ) -> None:
        finger_offsets = rng.normal(0, 1.5, 5)
        for k in range(n):
            base_angle = float(angle_traj[k])
            angles = base_angle + finger_offsets + rng.normal(0, angle_jitter, 5)
            angles = np.clip(angles, 0.0, 110.0)
            grip = max(0.0, float(grip_traj[k]) + rng.normal(0, 2.0))
            emg = max(0.0, float(emg_traj[k]) + rng.normal(0, 20.0))
            gx = rng.normal(gyro_bias_x, gyro_scale)
            gy = rng.normal(0, gyro_scale * 0.7)
            gz = rng.normal(0, gyro_scale * 0.7)
            ax = rng.normal(0.03, 0.05)
            ay = rng.normal(-0.02, 0.04)
            az = rng.normal(0.97, 0.04)

            if callable(interp):
                interp_label = interp(grip, emg, base_angle)
            else:
                interp_label = interp

            records.append(
                {
                    "thumb_angle": round(angles[0], 1),
                    "index_angle": round(angles[1], 1),
                    "middle_angle": round(angles[2], 1),
                    "ring_angle": round(angles[3], 1),
                    "pinky_angle": round(angles[4], 1),
                    "grip_force": int(round(grip)),
                    "emg_level": int(round(emg)),
                    "acc_x": round(ax, 3),
                    "acc_y": round(ay, 3),
                    "acc_z": round(az, 3),
                    "gyro_x": round(gx, 3),
                    "gyro_y": round(gy, 3),
                    "gyro_z": round(gz, 3),
                    "exercise_state": state,
                    "interpretation": interp_label,
                }
            )

    def grip_interp(grip: float, emg: float, angle: float) -> str:
        return "successful_grip" if grip >= 50 else "movement_phase"

    def fatigue_interp(grip: float, emg: float, angle: float) -> str:
        if emg > 480 and angle < 45:
            return "high_effort_low_motion"
        return "movement_phase"

    # --- Isınma / dinlenme ---
    n_rest = max(20, int(round(30 * rest_scale)))
    add_phase(
        n_rest,
        _interp(5, 5, n_rest),
        _interp(2, 2, n_rest),
        _interp(78, 78, n_rest),
        gyro_scale=0.5,
        gyro_bias_x=0.0,
        state="rest",
        interp="no_exercise",
        angle_jitter=1.8,
    )

    # --- Egzersiz döngüleri ---
    for cycle_i in range(n_cycles):
        cycle_factor = 1.0 + 0.08 * np.sin(cycle_i * 1.7 + session_index)
        n_open = max(10, int(round(14 * phase_scale * cycle_factor)))
        add_phase(
            n_open,
            _interp(max_angle * 0.9, 8, n_open),
            _interp(4, 3, n_open),
            _interp(190, 200, n_open),
            gyro_scale=1.0,
            gyro_bias_x=2.5,
            state="hand_opening",
            interp="movement_phase",
        )

        n_close = max(12, int(round(16 * phase_scale * cycle_factor)))
        add_phase(
            n_close,
            _interp(8, max_angle, n_close),
            _interp(6, max_grip * 0.7, n_close),
            _interp(210, grip_emg, n_close),
            gyro_scale=1.3,
            gyro_bias_x=4.0,
            state="hand_closing",
            interp="movement_phase",
        )

        n_hold = max(10, int(round(14 * phase_scale)))
        add_phase(
            n_hold,
            _interp(max_angle, max_angle, n_hold),
            _interp(max_grip, max_grip, n_hold),
            _interp(grip_emg, grip_emg, n_hold),
            gyro_scale=0.8,
            gyro_bias_x=0.0,
            state="grip_hold",
            interp=grip_interp,
            angle_jitter=2.0,
        )

        n_rel = max(10, int(round(14 * phase_scale * cycle_factor)))
        add_phase(
            n_rel,
            _interp(max_angle, 8, n_rel),
            _interp(max_grip * 0.7, 10, n_rel),
            _interp(grip_emg * 0.8, 150, n_rel),
            gyro_scale=1.3,
            gyro_bias_x=-3.5,
            state="release",
            interp="movement_phase",
        )

    # --- Yorgunluk denemesi ---
    n_fatigue = max(15, int(round((18 + fatigue_tendency * 22) * phase_scale)))
    fatigue_emg = (500 + 250 * fatigue_tendency) * fatigue_boost
    fatigue_angle = max(10.0, (20 + 15 * capability) / fatigue_boost)
    add_phase(
        n_fatigue,
        _interp(fatigue_angle, fatigue_angle, n_fatigue) + rng.normal(0, 4, n_fatigue),
        _interp(max_grip * 0.35, max_grip * 0.35, n_fatigue),
        _interp(fatigue_emg, fatigue_emg, n_fatigue),
        gyro_scale=1.2,
        gyro_bias_x=1.5,
        state="fatigue_attempt",
        interp=fatigue_interp,
        angle_jitter=3.2,
    )

    df = pd.DataFrame(records)
    df.insert(0, "time_ms", np.arange(len(df)) * dt_ms)
    return df[ALL_COLS]


def compute_session_metrics(df: pd.DataFrame) -> dict:
    """Üretilen seanstan özet metrikleri analitik olarak hesaplar."""
    states = df["exercise_state"].values
    transitions = int(
        sum(
            1
            for i in range(len(states))
            if states[i] == "grip_hold" and (i == 0 or states[i - 1] != "grip_hold")
        )
    )
    avg_grip = float(df["grip_force"].mean())
    fatigue_ratio = float((df["interpretation"] == "high_effort_low_motion").mean())

    rep_score = min(transitions / 6.0, 1.0) * 35
    grip_score = min(avg_grip / 60.0, 1.0) * 30
    fatigue_penalty = fatigue_ratio * 25
    progress = _clip(rep_score + grip_score + 25 - fatigue_penalty, 0.0, 100.0)

    return {
        "completed_reps": transitions,
        "avg_grip_force": round(avg_grip, 1),
        "fatigue_ratio_pct": round(fatigue_ratio * 100, 1),
        "progress_score": round(progress, 1),
    }


def generate_patient_sessions(
    archetype: PatientArchetype,
    n_sessions: int = 8,
    base_seed: int = 100,
) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    """Bir hasta için çoklu seans verisi ve ilerleme geçmişi üretir."""
    sessions: list[pd.DataFrame] = []
    history_rows: list[dict] = []
    start_date = datetime.now() - timedelta(days=n_sessions * 3)

    for i in range(n_sessions):
        capability = _clip(archetype.base_recovery + archetype.improvement_rate * i, 0.05, 1.0)
        fatigue = _clip(archetype.fatigue_tendency - 0.025 * i, 0.05, 1.0)
        seed = base_seed + i * 7919 + abs(hash(archetype.key)) % 10000
        df = generate_session(capability, fatigue, seed=seed, session_index=i)
        sessions.append(df)

        metrics = compute_session_metrics(df)
        history_rows.append(
            {
                "session": i + 1,
                "date": (start_date + timedelta(days=i * 3)).strftime("%Y-%m-%d"),
                "recovery_pct": round(capability * 100, 1),
                **metrics,
            }
        )

    history = pd.DataFrame(history_rows)
    return sessions, history


def save_generated_data(n_sessions: int = 8, out_dir: Path = GENERATED_DIR) -> dict:
    """Tüm hasta arketipleri için seans CSV'leri ve geçmiş dosyalarını kaydeder."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, list[str]] = {}

    for key, archetype in PATIENT_ARCHETYPES.items():
        sessions, history = generate_patient_sessions(archetype, n_sessions=n_sessions)
        paths: list[str] = []
        for i, df in enumerate(sessions, start=1):
            path = out_dir / f"{key}_session_{i:02d}.csv"
            df.to_csv(path, index=False)
            paths.append(str(path))
        history_path = out_dir / f"{key}_history.csv"
        history.to_csv(history_path, index=False)
        paths.append(str(history_path))
        summary[key] = paths

    return summary


if __name__ == "__main__":
    result = save_generated_data()
    for patient, files in result.items():
        print(f"{patient}: {len(files)} dosya")
    print(f"Çıktı dizini: {GENERATED_DIR}")
