"""Kural tabanlı öneri motoru ve ilerleme metrikleri."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PatientProfile:
    target_reps: int = 15
    session_number: int = 1
    difficulty: str = "orta"  # kolay, orta, zor


@dataclass
class SessionMetrics:
    completed_reps: int = 0
    avg_grip_force: float = 0.0
    fatigue_ratio: float = 0.0
    progress_score: float = 0.0
    grip_forces: list[float] = field(default_factory=list)
    fatigue_count: int = 0
    total_predictions: int = 0
    states_seen: set[str] = field(default_factory=set)


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

DIFFICULTY_FACTOR = {"kolay": 0.85, "orta": 1.0, "zor": 1.15}


def update_metrics(
    metrics: SessionMetrics,
    row: dict,
    prediction: dict | None,
    prev_state: str | None,
) -> str | None:
    """Metrikleri günceller; grip_hold geçişinde tekrar sayar."""
    current_state = row.get("exercise_state", "")
    metrics.states_seen.add(current_state)

    grip = float(row.get("grip_force", 0))
    metrics.grip_forces.append(grip)

    if prediction:
        metrics.total_predictions += 1
        if prediction["interpretation"] == "high_effort_low_motion":
            metrics.fatigue_count += 1

    if prev_state != "grip_hold" and current_state == "grip_hold":
        metrics.completed_reps += 1

    return current_state


def finalize_metrics(metrics: SessionMetrics) -> SessionMetrics:
    if metrics.grip_forces:
        metrics.avg_grip_force = sum(metrics.grip_forces) / len(metrics.grip_forces)
    if metrics.total_predictions > 0:
        metrics.fatigue_ratio = metrics.fatigue_count / metrics.total_predictions

    rep_score = min(metrics.completed_reps / 3.0, 1.0) * 35
    grip_score = min(metrics.avg_grip_force / 60.0, 1.0) * 30
    fatigue_penalty = metrics.fatigue_ratio * 25
    variety_score = min(len(metrics.states_seen) / 6.0, 1.0) * 10
    metrics.progress_score = max(0.0, min(100.0, rep_score + grip_score + variety_score + 25 - fatigue_penalty))
    return metrics


def get_live_recommendation(
    prediction: dict | None,
    row: dict,
    profile: PatientProfile,
    metrics: SessionMetrics,
) -> dict:
    """Anlık öneri üretir."""
    if prediction is None and not row.get("exercise_state"):
        return {
            "title": "Veri Toplanıyor",
            "message": "Yeterli sensör verisi bekleniyor...",
            "severity": "info",
            "action": "Bekleyin",
        }

    state = row.get("exercise_state") or (prediction["exercise_state"] if prediction else "rest")
    interp = row.get("interpretation") or (prediction["interpretation"] if prediction else "no_exercise")
    grip = float(row.get("grip_force", 0))
    emg = float(row.get("emg_level", 0))
    factor = DIFFICULTY_FACTOR.get(profile.difficulty, 1.0)
    adjusted_target = int(profile.target_reps * factor)

    if interp == "high_effort_low_motion" or state == "fatigue_attempt":
        reduced = max(5, int(adjusted_target * 0.8))
        return {
            "title": "Yorgunluk Tespit Edildi",
            "message": (
                f"EMG ({emg:.0f}) yüksek ancak hareket sınırlı. "
                f"Bugün hedefi {adjusted_target} tekrardan {reduced} tekrara düşürün. "
                "2 dakika dinlenme önerilir."
            ),
            "severity": "warning",
            "action": f"Hedef: {reduced} tekrar",
        }

    if interp == "successful_grip" and grip >= 50:
        increased = int(adjusted_target * 1.12)
        return {
            "title": "Güçlü Kavrama",
            "message": (
                f"Kavrama kuvveti {grip:.0f} N — hedefin üzerinde. "
                f"Yarın tekrar sayısını {adjusted_target}'ten {increased}'e çıkarın (+12%)."
            ),
            "severity": "success",
            "action": f"Sonraki hedef: {increased} tekrar",
        }

    if state in ("hand_opening", "hand_closing") and interp == "movement_phase":
        return {
            "title": "Hareket Devam Ediyor",
            "message": "Parmak açıları ve hareket akıcılığı normal aralıkta. Egzersize devam edin.",
            "severity": "info",
            "action": "Devam",
        }

    if state == "grip_hold":
        return {
            "title": "Kavrama Tutuluyor",
            "message": f"Kavrama fazı aktif (kuvvet: {grip:.0f} N). Pozisyonu 3-5 sn koruyun.",
            "severity": "success",
            "action": "Tut",
        }

    if state == "rest":
        return {
            "title": "Dinlenme Fazı",
            "message": "Hasta dinlenme modunda. Bir sonraki set için hazır olduğunda devam edin.",
            "severity": "info",
            "action": "Dinlen",
        }

    return {
        "title": "İzleme Aktif",
        "message": f"Mevcut faz: {prediction['exercise_state_tr']}. Sistem veriyi analiz ediyor.",
        "severity": "info",
        "action": "İzle",
    }


def get_session_summary(
    metrics: SessionMetrics,
    profile: PatientProfile,
) -> dict:
    """Seans sonu özeti ve sonraki seans planı."""
    metrics = finalize_metrics(metrics)
    factor = DIFFICULTY_FACTOR.get(profile.difficulty, 1.0)
    base_target = int(profile.target_reps * factor)

    if metrics.progress_score >= 75:
        next_reps = int(base_target * 1.12)
        progress_pct = 12
        mode = "Aktif mod"
    elif metrics.fatigue_ratio > 0.3:
        next_reps = max(5, int(base_target * 0.8))
        progress_pct = -8
        mode = "Pasif mod (düşük direnç)"
    else:
        next_reps = base_target
        progress_pct = 5
        mode = "Mevcut mod"

    doctor_note = (
        f"Seans #{profile.session_number} tamamlandı. "
        f"Tamamlanan tekrar: {metrics.completed_reps}. "
        f"Ortalama kavrama kuvveti: {metrics.avg_grip_force:.1f} N. "
        f"Yorgunluk oranı: %{metrics.fatigue_ratio * 100:.0f}. "
        f"İlerleme puanı: {metrics.progress_score:.0f}/100."
    )

    if progress_pct > 0:
        doctor_note += (
            f" Öneri: Bugün {base_target} tekrar yerine {next_reps} tekrar yapın; "
            f"ilerleme %{progress_pct} arttı."
        )
    else:
        doctor_note += (
            f" Öneri: Bugün hedefi {next_reps} tekrara düşürün ve dinlenme süresini artırın."
        )

    return {
        "completed_reps": metrics.completed_reps,
        "avg_grip_force": round(metrics.avg_grip_force, 1),
        "fatigue_ratio_pct": round(metrics.fatigue_ratio * 100, 1),
        "progress_score": round(metrics.progress_score, 1),
        "next_session_reps": next_reps,
        "next_session_mode": mode,
        "progress_pct": progress_pct,
        "doctor_note": doctor_note,
    }
