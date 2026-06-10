"""Plotly grafik yardımcıları (arayüz ve rapor için ortak)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FINGER_COLORS = {
    "thumb_angle": "#ef4444",
    "index_angle": "#f97316",
    "middle_angle": "#eab308",
    "ring_angle": "#22c55e",
    "pinky_angle": "#3b82f6",
}
FINGER_LABELS = {
    "thumb_angle": "Başparmak",
    "index_angle": "İşaret",
    "middle_angle": "Orta",
    "ring_angle": "Yüzük",
    "pinky_angle": "Serçe",
}

BASE_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Segoe UI, Inter, sans-serif", size=13, color="#1e293b"),
    margin=dict(l=30, r=20, t=50, b=30),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


def empty_figure(message: str = "Veri bekleniyor...") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**BASE_LAYOUT, height=420)
    fig.add_annotation(text=message, showarrow=False, font=dict(size=16, color="#94a3b8"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def build_live_figure(history: list[dict] | pd.DataFrame) -> go.Figure:
    """Canlı sensör verisi: parmak açıları, kuvvet/EMG, hareket."""
    df = pd.DataFrame(history) if not isinstance(history, pd.DataFrame) else history
    if df.empty:
        return empty_figure("Sensör verisi bekleniyor...")

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Parmak Açıları (°)", "Kavrama Kuvveti (N) & EMG (µV)", "Hareket Yoğunluğu (Gyro)"),
        vertical_spacing=0.09,
        row_heights=[0.4, 0.32, 0.28],
    )

    for col, color in FINGER_COLORS.items():
        fig.add_trace(
            go.Scatter(
                x=df["time_ms"],
                y=df[col],
                name=FINGER_LABELS[col],
                line=dict(color=color, width=2),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(x=df["time_ms"], y=df["grip_force"], name="Kuvvet", line=dict(color="#8b5cf6", width=2.5)),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["time_ms"],
            y=df["emg_level"],
            name="EMG",
            line=dict(color="#ec4899", width=1.8, dash="dot"),
        ),
        row=2,
        col=1,
    )

    gyro_mag = (df["gyro_x"] ** 2 + df["gyro_y"] ** 2 + df["gyro_z"] ** 2) ** 0.5
    fig.add_trace(
        go.Scatter(
            x=df["time_ms"],
            y=gyro_mag,
            name="Hareket",
            line=dict(color="#14b8a6", width=2),
            fill="tozeroy",
            fillcolor="rgba(20,184,166,0.12)",
        ),
        row=3,
        col=1,
    )

    fig.update_layout(**BASE_LAYOUT, height=560, hovermode="x unified")
    fig.update_xaxes(title_text="Zaman (ms)", row=3, col=1)
    return fig


def build_progress_figure(history_df: pd.DataFrame) -> go.Figure:
    """Seanslar arası ilerleme: puan + kuvvet + yorgunluk."""
    if history_df is None or history_df.empty:
        return empty_figure("Seans geçmişi bekleniyor...")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=history_df["session"],
            y=history_df["progress_score"],
            name="İlerleme Puanı",
            marker_color="rgba(59,130,246,0.75)",
            text=history_df["progress_score"],
            textposition="outside",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=history_df["session"],
            y=history_df["avg_grip_force"],
            name="Ort. Kavrama Kuvveti (N)",
            line=dict(color="#22c55e", width=3),
            mode="lines+markers",
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=history_df["session"],
            y=history_df["fatigue_ratio_pct"],
            name="Yorgunluk Oranı (%)",
            line=dict(color="#f59e0b", width=2, dash="dash"),
            mode="lines+markers",
        ),
        secondary_y=True,
    )

    fig.update_layout(**BASE_LAYOUT, height=440, barmode="group")
    fig.update_xaxes(title_text="Seans", dtick=1)
    fig.update_yaxes(title_text="İlerleme Puanı (0-100)", range=[0, 110], secondary_y=False)
    fig.update_yaxes(title_text="Kuvvet (N) / Yorgunluk (%)", secondary_y=True)
    return fig


def build_phase_distribution_figure(history: list[dict] | pd.DataFrame) -> go.Figure:
    """Seans içinde fazların dağılımı (donut)."""
    df = pd.DataFrame(history) if not isinstance(history, pd.DataFrame) else history
    if df.empty or "exercise_state" not in df:
        return empty_figure("Faz verisi bekleniyor...")

    labels_tr = {
        "rest": "Dinlenme",
        "hand_opening": "El Açma",
        "hand_closing": "El Kapama",
        "grip_hold": "Kavrama",
        "release": "Bırakma",
        "fatigue_attempt": "Yorgunluk",
    }
    counts = df["exercise_state"].value_counts()
    fig = go.Figure(
        go.Pie(
            labels=[labels_tr.get(k, k) for k in counts.index],
            values=counts.values,
            hole=0.55,
            marker=dict(colors=["#94a3b8", "#f97316", "#3b82f6", "#22c55e", "#a855f7", "#ef4444"]),
        )
    )
    fig.update_layout(**BASE_LAYOUT, height=340, showlegend=True)
    return fig
