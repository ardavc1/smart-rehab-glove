"""Akıllı Rehabilitasyon Eldiveni — AI Öneri Sistemi (Streamlit)."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.charts import (
    build_live_figure,
    build_phase_distribution_figure,
    build_progress_figure,
)
from src.data_generator import PATIENT_ARCHETYPES, compute_session_metrics, generate_patient_sessions
from src.predict import Predictor
from src.recommendation import (
    PatientProfile,
    SessionMetrics,
    get_live_recommendation,
    get_session_summary,
    update_metrics,
)
from src.report import build_report_html, fig_to_png_bytes
from src.simulator import SCENARIOS, GloveSimulator, SimulationConfig

MODELS_DIR = Path(__file__).resolve().parent / "models"
SEVERITY_COLORS = {
    "info": "#2563eb",
    "success": "#16a34a",
    "warning": "#d97706",
    "error": "#dc2626",
}

st.set_page_config(
    page_title="Rehab Eldiveni AI",
    page_icon="🧤",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
  .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1400px; }
  #MainMenu, footer { visibility: hidden; }

  .app-header {
    background: linear-gradient(120deg, #1e3a8a 0%, #2563eb 55%, #0ea5e9 100%);
    border-radius: 16px; padding: 20px 26px; color: #fff; margin-bottom: 18px;
    box-shadow: 0 8px 24px rgba(37,99,235,0.25);
  }
  .app-header h1 { margin: 0; font-size: 24px; font-weight: 700; }
  .app-header p { margin: 4px 0 0 0; opacity: 0.9; font-size: 14px; }

  .patient-bar {
    display: flex; flex-wrap: wrap; gap: 14px; align-items: center;
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 14px;
    padding: 14px 20px; margin-bottom: 16px;
  }
  .patient-bar .pname { font-size: 18px; font-weight: 700; color: #0f172a; }
  .patient-bar .pcond { color: #64748b; font-size: 13px; }
  .chip {
    padding: 5px 12px; border-radius: 999px; font-size: 12px; font-weight: 600;
    background: #e0e7ff; color: #3730a3;
  }
  .chip.green { background: #dcfce7; color: #166534; }
  .chip.blue { background: #dbeafe; color: #1e40af; }
  .chip.orange { background: #ffedd5; color: #9a3412; }
  .chip.red { background: #fee2e2; color: #991b1b; }
  .spacer { flex: 1; }

  .kpi-card {
    border: 1px solid #e2e8f0; border-radius: 14px; padding: 16px 18px;
    background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }
  .kpi-card .k-label { font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: .4px; }
  .kpi-card .k-value { font-size: 28px; font-weight: 700; color: #0f172a; margin-top: 4px; }
  .kpi-card .k-sub { font-size: 12px; color: #94a3b8; margin-top: 2px; }

  .rec-card { border-radius: 14px; padding: 16px 18px; border: 1px solid; margin-top: 4px; }
  .rec-card h4 { margin: 0 0 8px 0; font-size: 16px; }
  .rec-card p { margin: 0 0 8px 0; font-size: 14px; color: #334155; }
  .rec-card .action { font-size: 12px; color: #475569; }

  .status-row { display: flex; gap: 10px; flex-wrap: wrap; }
  .status-box {
    flex: 1; min-width: 120px; background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 12px 14px;
  }
  .status-box .s-label { font-size: 11px; color: #64748b; text-transform: uppercase; }
  .status-box .s-value { font-size: 18px; font-weight: 700; color: #0f172a; margin-top: 4px; }
</style>
"""


def init_session_state() -> None:
    defaults = {
        "running": False,
        "finished": False,
        "current_index": 0,
        "buffer": [],
        "history": [],
        "metrics": SessionMetrics(),
        "prev_state": None,
        "last_prediction": None,
        "last_recommendation": None,
        "summary": None,
        "generated": {},
        "active_df": None,
        "active_patient": None,
        "playback_info": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource
def load_predictor() -> Predictor:
    if not (MODELS_DIR / "movement_classifier.joblib").exists():
        from src.train import train_models

        train_models()
    return Predictor(MODELS_DIR)


def reset_session() -> None:
    st.session_state.running = False
    st.session_state.finished = False
    st.session_state.current_index = 0
    st.session_state.buffer = []
    st.session_state.history = []
    st.session_state.metrics = SessionMetrics()
    st.session_state.prev_state = None
    st.session_state.last_prediction = None
    st.session_state.last_recommendation = None
    st.session_state.summary = None


def score_chip_class(score: float) -> str:
    if score >= 75:
        return "green"
    if score >= 50:
        return "blue"
    if score >= 25:
        return "orange"
    return "red"


def kpi_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="k-sub">{sub}</div>' if sub else ""
    return f'<div class="kpi-card"><div class="k-label">{label}</div><div class="k-value">{value}</div>{sub_html}</div>'


def render_recommendation_card(rec: dict) -> None:
    color = SEVERITY_COLORS.get(rec["severity"], "#2563eb")
    bg = {
        "#2563eb": "#eff6ff",
        "#16a34a": "#f0fdf4",
        "#d97706": "#fffbeb",
        "#dc2626": "#fef2f2",
    }.get(color, "#eff6ff")
    st.markdown(
        f"""
        <div class="rec-card" style="border-color:{color}33; background:{bg}; border-left:4px solid {color};">
            <h4 style="color:{color};">{rec['title']}</h4>
            <p>{rec['message']}</p>
            <div class="action"><b>Önerilen aksiyon:</b> {rec['action']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_simulation_step(simulator: GloveSimulator, predictor: Predictor, profile: PatientProfile) -> bool:
    idx = st.session_state.current_index
    row = simulator.get_row(idx)
    if row is None:
        st.session_state.running = False
        st.session_state.finished = True
        st.session_state.summary = get_session_summary(st.session_state.metrics, profile)
        return False

    sensor = simulator.row_to_sensor_dict(row)
    st.session_state.buffer.append(sensor)
    st.session_state.history.append(row)

    prediction = predictor.predict_from_buffer(st.session_state.buffer)
    st.session_state.last_prediction = prediction

    st.session_state.prev_state = update_metrics(
        st.session_state.metrics, row, prediction, st.session_state.prev_state
    )
    st.session_state.last_recommendation = get_live_recommendation(
        prediction, row, profile, st.session_state.metrics
    )
    st.session_state.current_index += 1
    return True


def ensure_generated(patient_key: str, n_sessions: int = 8) -> None:
    if patient_key not in st.session_state.generated:
        archetype = PATIENT_ARCHETYPES[patient_key]
        sessions, history = generate_patient_sessions(archetype, n_sessions=n_sessions)
        st.session_state.generated[patient_key] = {"sessions": sessions, "history": history}


def build_playback_info(
    data_source: str,
    patient_key: str | None,
    session_number: int,
    scenario_key: str,
    active_df: pd.DataFrame | None,
) -> dict:
    scenario = SCENARIOS[scenario_key]
    if active_df is not None:
        if scenario.get("state_filter"):
            filtered = active_df[active_df["exercise_state"].isin(scenario["state_filter"])]
            row_count = len(filtered) if not filtered.empty else len(active_df)
            note = f"Aynı seansın '{scenario['label']}' kesiti ({row_count} ölçüm)"
        else:
            row_count = len(active_df)
            note = f"Tam seans ({row_count} ölçüm)"
        avg_grip = round(active_df["grip_force"].mean(), 1)
        patient_name = PATIENT_ARCHETYPES[patient_key].name if patient_key else "Örnek CSV"
        return {
            "source": data_source,
            "patient": patient_name,
            "session": session_number,
            "scenario": scenario["label"],
            "rows": row_count,
            "avg_grip": avg_grip,
            "note": note,
        }
    return {
        "source": data_source,
        "patient": "Örnek CSV (rehab_glove_sample_300)",
        "session": session_number,
        "scenario": scenario["label"],
        "rows": 300 if scenario_key == "normal" else "değişken",
        "avg_grip": "~32",
        "note": "Sabit örnek dosya — her başlatmada aynı veri",
    }


def render_sidebar(predictor: Predictor) -> tuple[PatientProfile, str, int, str | None, pd.DataFrame | None, dict]:
    with st.sidebar:
        st.markdown("### Veri Kaynağı")
        data_source = st.radio(
            "Kaynak",
            ["Sentetik hasta", "Örnek seans (orijinal CSV)"],
            label_visibility="collapsed",
        )

        patient_key = None
        history_df = None
        active_df = None
        session_number = 1

        if data_source == "Sentetik hasta":
            patient_key = st.selectbox(
                "Hasta",
                options=list(PATIENT_ARCHETYPES.keys()),
                format_func=lambda k: PATIENT_ARCHETYPES[k].name,
            )
            n_sessions = st.slider("Üretilecek seans sayısı", 4, 12, 8)
            if st.button("Demo verisi üret", width="stretch"):
                st.session_state.generated.pop(patient_key, None)
                ensure_generated(patient_key, n_sessions)
                st.toast(f"{PATIENT_ARCHETYPES[patient_key].name} için {n_sessions} seans üretildi.")

            if patient_key in st.session_state.generated:
                gen = st.session_state.generated[patient_key]
                history_df = gen["history"]
                session_number = st.selectbox(
                    "Seans seçimi",
                    options=list(range(1, len(gen["sessions"]) + 1)),
                    index=0,
                    format_func=lambda i: f"Seans #{i}",
                )
                active_df = gen["sessions"][session_number - 1]
                preview = compute_session_metrics(active_df)
                st.caption(
                    f"Seçili: Seans #{session_number} — {len(active_df)} ölçüm, "
                    f"ort. kuvvet {preview['avg_grip_force']} N, "
                    f"ilerleme {preview['progress_score']}/100"
                )
            else:
                st.warning("Başlamak için önce **Demo verisi üret** butonuna basın.")
        else:
            st.caption("Orijinal 300 satırlık örnek seans kullanılacak.")

        st.divider()
        st.markdown("### Hasta Profili")
        target_reps = st.number_input("Hedef günlük tekrar", 5, 50, 15)
        if data_source != "Sentetik hasta":
            session_number = st.number_input("Seans numarası", 1, 100, 1)
        difficulty = st.selectbox("Zorluk seviyesi", ["kolay", "orta", "zor"], index=1)

        st.divider()
        st.markdown("### Simülasyon")
        scenario_key = st.selectbox(
            "Senaryo",
            options=list(SCENARIOS.keys()),
            format_func=lambda k: SCENARIOS[k]["label"],
        )
        speed = st.slider("Hız çarpanı", 1, 50, 12)
        st.info(SCENARIOS[scenario_key]["description"])

        col1, col2 = st.columns(2)
        playback_preview = build_playback_info(
            data_source, patient_key, session_number, scenario_key, active_df
        )
        if data_source == "Sentetik hasta" and active_df is None:
            st.error("Sentetik hasta seçildi ama veri üretilmedi. Önce **Demo verisi üret**'e basın.")
        elif scenario_key != "normal":
            st.caption(f"ℹ️ **{SCENARIOS[scenario_key]['label']}** = aynı seansın bir faz kesiti, yeni seans değil.")

        with col1:
            start_disabled = data_source == "Sentetik hasta" and active_df is None
            if st.button("Seansı Başlat", type="primary", width="stretch", disabled=start_disabled):
                reset_session()
                st.session_state.running = True
                st.session_state.active_df = active_df
                st.session_state.active_patient = patient_key
                st.session_state.playback_info = playback_preview
                st.session_state.simulator = GloveSimulator(
                    SimulationConfig(
                        scenario_key=scenario_key,
                        speed_multiplier=speed,
                        dataframe=active_df,
                    )
                )
                st.rerun()
        with col2:
            if st.button("Sıfırla", width="stretch"):
                reset_session()
                st.rerun()

    profile = PatientProfile(
        target_reps=target_reps,
        session_number=session_number,
        difficulty=difficulty,
    )
    return profile, scenario_key, speed, patient_key, history_df, playback_preview


def render_header(patient_key: str | None) -> None:
    st.markdown(
        """
        <div class="app-header">
            <h1>Akıllı Rehabilitasyon Eldiveni</h1>
            <p>AI destekli kişiselleştirilmiş terapi öneri sistemi — gerçek zamanlı sensör analizi (simülasyon)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if patient_key and patient_key in PATIENT_ARCHETYPES:
        arc = PATIENT_ARCHETYPES[patient_key]
        score = st.session_state.summary["progress_score"] if st.session_state.summary else None
        score_html = (
            f'<span class="chip {score_chip_class(score)}">İlerleme {score}/100</span>'
            if score is not None
            else '<span class="chip">Seans bekleniyor</span>'
        )
        st.markdown(
            f"""
            <div class="patient-bar">
                <div>
                    <div class="pname">{arc.name}</div>
                    <div class="pcond">{arc.condition}</div>
                </div>
                <div class="spacer"></div>
                {score_html}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_playback_banner(info: dict | None) -> None:
    if not info:
        return
    st.info(
        f"**Kaynak:** {info['patient']} · **Seans #{info['session']}** · "
        f"**Senaryo:** {info['scenario']} · **{info['rows']} ölçüm** · "
        f"ort. kuvvet ~{info['avg_grip']} N — _{info['note']}_"
    )


def render_live_tab(simulator: GloveSimulator, profile: PatientProfile) -> None:
    render_playback_banner(st.session_state.get("playback_info"))

    if st.session_state.running:
        st.caption("🔴 Canlı — grafikler ve öneriler seans boyunca güncelleniyor...")
    elif st.session_state.finished:
        st.caption("✅ Seans tamamlandı.")

    col_main, col_side = st.columns([2, 1])

    with col_main:
        if st.session_state.history:
            progress = simulator.progress_pct(st.session_state.current_index - 1)
            st.progress(min(progress / 100, 1.0), text=f"Seans ilerlemesi: %{progress:.0f}")
        st.plotly_chart(
            build_live_figure(st.session_state.history),
            width="stretch",
            key=f"live_chart_{st.session_state.current_index}",
        )

    with col_side:
        st.markdown("#### AI Analizi")
        pred = st.session_state.last_prediction
        if pred:
            st.markdown(
                f"""
                <div class="status-row">
                    <div class="status-box"><div class="s-label">Hareket Fazı</div><div class="s-value">{pred['exercise_state_tr']}</div></div>
                    <div class="status-box"><div class="s-label">Güven</div><div class="s-value">%{pred['state_confidence'] * 100:.0f}</div></div>
                </div>
                <div class="status-row" style="margin-top:10px;">
                    <div class="status-box"><div class="s-label">Klinik Yorum</div><div class="s-value">{pred['interpretation_tr']}</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Seans başlatıldığında AI analizi burada görünecek.")

        st.markdown("#### Anlık Öneri")
        if st.session_state.last_recommendation:
            render_recommendation_card(st.session_state.last_recommendation)
        else:
            st.info("Öneri için seansı başlatın.")

        if st.session_state.history:
            m = st.session_state.metrics
            last_grip = m.grip_forces[-1] if m.grip_forces else 0
            st.markdown(
                f"""
                <div class="status-row" style="margin-top:12px;">
                    <div class="status-box"><div class="s-label">Tamamlanan Tekrar</div><div class="s-value">{m.completed_reps}</div></div>
                    <div class="status-box"><div class="s-label">Son Kuvvet</div><div class="s-value">{last_grip:.0f} N</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if st.session_state.finished and st.session_state.summary:
        st.divider()
        st.markdown("### Seans Özeti")
        summary = st.session_state.summary
        cols = st.columns(4)
        cols[0].markdown(kpi_card("Tamamlanan Tekrar", str(summary["completed_reps"])), unsafe_allow_html=True)
        cols[1].markdown(kpi_card("Ort. Kavrama Kuvveti", f"{summary['avg_grip_force']} N"), unsafe_allow_html=True)
        cols[2].markdown(kpi_card("Yorgunluk Oranı", f"%{summary['fatigue_ratio_pct']}"), unsafe_allow_html=True)
        cols[3].markdown(kpi_card("İlerleme Puanı", f"{summary['progress_score']}/100"), unsafe_allow_html=True)

        st.success(summary["doctor_note"])

        st.markdown("#### Faz Dağılımı")
        st.plotly_chart(
            build_phase_distribution_figure(st.session_state.history),
            width="stretch",
            key="phase_dist",
        )


def render_history_tab(patient_key: str | None, history_df: pd.DataFrame | None) -> None:
    if history_df is None or history_df.empty:
        st.info(
            "Seanslar arası ilerleme grafiği için sol panelden bir sentetik hasta seçip "
            "**'Demo verisi üret'** butonuna basın."
        )
        return

    arc = PATIENT_ARCHETYPES.get(patient_key)
    if arc:
        st.markdown(f"### {arc.name} — Tedavi Süreci ({len(history_df)} seans)")

    latest = history_df.iloc[-1]
    first = history_df.iloc[0]
    cols = st.columns(4)
    cols[0].markdown(
        kpi_card("Güncel İlerleme", f"{latest['progress_score']}/100",
                 f"Başlangıç: {first['progress_score']}"),
        unsafe_allow_html=True,
    )
    cols[1].markdown(
        kpi_card("Kuvvet Artışı", f"+{latest['avg_grip_force'] - first['avg_grip_force']:.1f} N",
                 f"{first['avg_grip_force']} → {latest['avg_grip_force']} N"),
        unsafe_allow_html=True,
    )
    cols[2].markdown(
        kpi_card("Yorgunluk Değişimi", f"%{latest['fatigue_ratio_pct'] - first['fatigue_ratio_pct']:+.1f}",
                 f"Güncel: %{latest['fatigue_ratio_pct']}"),
        unsafe_allow_html=True,
    )
    cols[3].markdown(
        kpi_card("İyileşme Düzeyi", f"%{latest['recovery_pct']:.0f}"),
        unsafe_allow_html=True,
    )

    st.plotly_chart(build_progress_figure(history_df), width="stretch", key="progress_chart")

    st.markdown("#### Seans Kayıtları")
    st.dataframe(
        history_df.rename(
            columns={
                "session": "Seans",
                "date": "Tarih",
                "recovery_pct": "İyileşme %",
                "completed_reps": "Tekrar",
                "avg_grip_force": "Ort. Kuvvet (N)",
                "fatigue_ratio_pct": "Yorgunluk %",
                "progress_score": "İlerleme Puanı",
            }
        ),
        hide_index=True,
        width="stretch",
    )


def render_report_tab(
    profile: PatientProfile,
    patient_key: str | None,
    history_df: pd.DataFrame | None,
) -> None:
    st.markdown("### Klinik Rapor")
    if not st.session_state.history:
        st.info("Rapor oluşturmak için önce bir seans çalıştırın (Canlı İzleme sekmesi).")
        return

    arc = PATIENT_ARCHETYPES.get(patient_key)
    patient_name = arc.name if arc else "Demo Hasta"
    condition = arc.condition if arc else "Örnek seans verisi"

    if st.session_state.summary:
        summary = st.session_state.summary
    else:
        summary = get_session_summary(st.session_state.metrics, profile)

    st.write(
        "Aşağıdaki butonla seans verilerini, AI önerisini ve ilerleme grafiklerini içeren "
        "kendi içinde barındırılan bir HTML rapor indirebilirsiniz. Tarayıcıda açıp "
        "**Ctrl+P → PDF olarak kaydet** ile PDF'e dönüştürebilirsiniz."
    )

    if st.button("Rapor oluştur", type="primary"):
        with st.spinner("Rapor ve grafikler hazırlanıyor..."):
            live_fig = build_live_figure(st.session_state.history)
            progress_fig = build_progress_figure(history_df) if history_df is not None else None
            live_png = fig_to_png_bytes(live_fig)
            progress_png = fig_to_png_bytes(progress_fig) if progress_fig is not None else None

            html = build_report_html(
                patient_name=patient_name,
                condition=condition,
                profile={
                    "session_number": profile.session_number,
                    "target_reps": profile.target_reps,
                    "difficulty": profile.difficulty,
                },
                summary=summary,
                recommendation=st.session_state.last_recommendation,
                live_png=live_png,
                progress_png=progress_png,
                history_df=history_df,
            )
            st.session_state["report_html"] = html
            st.session_state["report_live_png"] = live_png
            st.session_state["report_progress_png"] = progress_png

    if "report_html" in st.session_state:
        st.success("Rapor hazır. Aşağıdan indirin.")
        c1, c2, c3 = st.columns(3)
        c1.download_button(
            "HTML rapor indir",
            data=st.session_state["report_html"],
            file_name=f"rehab_rapor_{patient_name.replace(' ', '_')}.html",
            mime="text/html",
            width="stretch",
        )
        if st.session_state.get("report_live_png"):
            c2.download_button(
                "Sensör grafiği (PNG)",
                data=st.session_state["report_live_png"],
                file_name="sensor_grafigi.png",
                mime="image/png",
                width="stretch",
            )
        if st.session_state.get("report_progress_png"):
            c3.download_button(
                "İlerleme grafiği (PNG)",
                data=st.session_state["report_progress_png"],
                file_name="ilerleme_grafigi.png",
                mime="image/png",
                width="stretch",
            )

        st.download_button(
            "Seans verisi (CSV) indir",
            data=pd.DataFrame(st.session_state.history).to_csv(index=False).encode("utf-8"),
            file_name="seans_verisi.csv",
            mime="text/csv",
        )

        with st.expander("Rapor önizleme"):
            components.html(st.session_state["report_html"], height=600, scrolling=True)


def main() -> None:
    init_session_state()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    predictor = load_predictor()

    profile, scenario_key, speed, patient_key, history_df, playback_preview = render_sidebar(predictor)
    render_header(patient_key)

    if "simulator" not in st.session_state:
        st.session_state.simulator = GloveSimulator(
            SimulationConfig(
                scenario_key=scenario_key,
                speed_multiplier=speed,
                dataframe=st.session_state.active_df,
            )
        )
    simulator: GloveSimulator = st.session_state.simulator

    # Önce adım at, sonra çiz — aksi halde st.rerun() grafikleri hiç göstermeden döngüye girer
    if st.session_state.running:
        run_simulation_step(simulator, predictor, profile)

    tab_live, tab_history, tab_report = st.tabs(["Canlı İzleme", "Seans Geçmişi", "Klinik Rapor"])
    with tab_live:
        render_live_tab(simulator, profile)
    with tab_history:
        render_history_tab(patient_key, history_df)
    with tab_report:
        render_report_tab(profile, patient_key, history_df)

    if st.session_state.running:
        time.sleep(simulator.delay_seconds())
        st.rerun()


if __name__ == "__main__":
    main()
