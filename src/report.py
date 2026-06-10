"""Klinik rapor üreticisi: kendi içinde barındırılan HTML (yazdır -> PDF) ve PNG."""

from __future__ import annotations

import base64
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go


def fig_to_png_bytes(fig: go.Figure, width: int = 1000, height: int = 520, scale: int = 2) -> bytes | None:
    """Plotly figürünü PNG bayt dizisine çevirir (kaleido gerektirir)."""
    try:
        return fig.to_image(format="png", width=width, height=height, scale=scale)
    except Exception:
        return None


def _img_tag(png_bytes: bytes | None, alt: str = "") -> str:
    if not png_bytes:
        return '<p class="muted">[Grafik dışa aktarılamadı — kaleido kurulu değil]</p>'
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f'<img alt="{alt}" src="data:image/png;base64,{b64}" />'


def _history_table(history_df: pd.DataFrame | None) -> str:
    if history_df is None or history_df.empty:
        return '<p class="muted">Seans geçmişi yok.</p>'

    cols = {
        "session": "Seans",
        "date": "Tarih",
        "recovery_pct": "İyileşme (%)",
        "completed_reps": "Tekrar",
        "avg_grip_force": "Ort. Kuvvet (N)",
        "fatigue_ratio_pct": "Yorgunluk (%)",
        "progress_score": "İlerleme Puanı",
    }
    available = [c for c in cols if c in history_df.columns]
    head = "".join(f"<th>{cols[c]}</th>" for c in available)
    rows = ""
    for _, r in history_df.iterrows():
        rows += "<tr>" + "".join(f"<td>{r[c]}</td>" for c in available) + "</tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"


def build_report_html(
    patient_name: str,
    condition: str,
    profile: dict,
    summary: dict,
    recommendation: dict | None = None,
    live_png: bytes | None = None,
    progress_png: bytes | None = None,
    history_df: pd.DataFrame | None = None,
) -> str:
    """Tam klinik rapor HTML'i üretir (tarayıcıda Ctrl+P ile PDF'e çevrilebilir)."""
    now = datetime.now().strftime("%d.%m.%Y %H:%M")
    rec_html = ""
    if recommendation:
        rec_html = f"""
        <div class="rec">
            <h4>{recommendation.get('title', 'Öneri')}</h4>
            <p>{recommendation.get('message', '')}</p>
            <p class="action"><b>Önerilen aksiyon:</b> {recommendation.get('action', '-')}</p>
        </div>
        """

    progress_pct = summary.get("progress_pct", 0)
    progress_sign = f"+{progress_pct}" if progress_pct >= 0 else f"{progress_pct}"

    return f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="utf-8" />
<title>Klinik Rehabilitasyon Raporu - {patient_name}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: "Segoe UI", Inter, Arial, sans-serif;
    color: #1e293b; margin: 0; padding: 32px; background: #fff;
  }}
  .header {{
    display: flex; justify-content: space-between; align-items: flex-start;
    border-bottom: 3px solid #2563eb; padding-bottom: 16px; margin-bottom: 24px;
  }}
  .header h1 {{ margin: 0; font-size: 22px; color: #1e3a8a; }}
  .header .sub {{ color: #64748b; font-size: 13px; margin-top: 4px; }}
  .badge {{
    background: #2563eb; color: #fff; padding: 6px 14px; border-radius: 999px;
    font-size: 13px; font-weight: 600;
  }}
  .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 20px 0; }}
  .kpi {{
    border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; background: #f8fafc;
  }}
  .kpi .label {{ font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: .5px; }}
  .kpi .value {{ font-size: 26px; font-weight: 700; color: #0f172a; margin-top: 6px; }}
  h3 {{ color: #1e3a8a; border-left: 4px solid #2563eb; padding-left: 10px; margin-top: 28px; }}
  .rec {{
    background: #eff6ff; border: 1px solid #bfdbfe; border-left: 4px solid #2563eb;
    border-radius: 10px; padding: 14px 18px; margin: 12px 0;
  }}
  .rec h4 {{ margin: 0 0 6px 0; color: #1d4ed8; }}
  .rec .action {{ font-size: 13px; color: #475569; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 10px; }}
  th, td {{ border: 1px solid #e2e8f0; padding: 8px 10px; text-align: center; }}
  th {{ background: #f1f5f9; color: #334155; }}
  img {{ width: 100%; border: 1px solid #e2e8f0; border-radius: 10px; margin-top: 10px; }}
  .muted {{ color: #94a3b8; font-style: italic; }}
  .note {{ background: #f8fafc; border-radius: 10px; padding: 14px 18px; margin-top: 10px; line-height: 1.6; }}
  .footer {{
    margin-top: 36px; padding-top: 14px; border-top: 1px solid #e2e8f0;
    font-size: 11px; color: #94a3b8;
  }}
  .profile {{ font-size: 13px; color: #475569; }}
  .profile span {{ display: inline-block; margin-right: 18px; }}
</style>
</head>
<body>
  <div class="header">
    <div>
      <h1>Akıllı Rehabilitasyon Eldiveni — Klinik Rapor</h1>
      <div class="sub">{patient_name} · {condition}</div>
      <div class="profile">
        <span><b>Seans:</b> #{profile.get('session_number', '-')}</span>
        <span><b>Hedef tekrar:</b> {profile.get('target_reps', '-')}</span>
        <span><b>Zorluk:</b> {profile.get('difficulty', '-')}</span>
      </div>
    </div>
    <div class="badge">İlerleme {summary.get('progress_score', 0)}/100</div>
  </div>

  <h3>Seans Özeti</h3>
  <div class="grid">
    <div class="kpi"><div class="label">Tamamlanan Tekrar</div><div class="value">{summary.get('completed_reps', 0)}</div></div>
    <div class="kpi"><div class="label">Ort. Kavrama Kuvveti</div><div class="value">{summary.get('avg_grip_force', 0)} N</div></div>
    <div class="kpi"><div class="label">Yorgunluk Oranı</div><div class="value">%{summary.get('fatigue_ratio_pct', 0)}</div></div>
    <div class="kpi"><div class="label">İlerleme Puanı</div><div class="value">{summary.get('progress_score', 0)}</div></div>
  </div>

  <h3>AI Önerisi</h3>
  {rec_html if rec_html else ''}
  <div class="note">{summary.get('doctor_note', '')}</div>

  <h3>Sensör Verisi (Seans)</h3>
  {_img_tag(live_png, "Sensör grafikleri")}

  <h3>Seanslar Arası İlerleme</h3>
  {_img_tag(progress_png, "İlerleme grafiği")}
  {_history_table(history_df)}

  <h3>Sonraki Seans Planı</h3>
  <div class="note">
    <b>Hedef tekrar:</b> {summary.get('next_session_reps', '-')} &nbsp;·&nbsp;
    <b>Egzersiz modu:</b> {summary.get('next_session_mode', '-')} &nbsp;·&nbsp;
    <b>Beklenen ilerleme:</b> %{progress_sign}
  </div>

  <div class="footer">
    Bu rapor {now} tarihinde Akıllı Rehabilitasyon Eldiveni AI Öneri Sistemi tarafından
    otomatik oluşturulmuştur. Veriler demo/simülasyon amaçlıdır ve klinik karar yerine geçmez.
  </div>
</body>
</html>"""
