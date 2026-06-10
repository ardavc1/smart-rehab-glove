"""CSV/bellek içi replay simülatörü ve senaryo segmentleri."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.features import DEFAULT_DATA_PATH, SENSOR_COLS, load_raw_data

SCENARIOS = {
    "normal": {
        "label": "Normal Seans",
        "state_filter": None,
        "description": "Tam rehabilitasyon seansı (tüm fazlar)",
    },
    "rest": {
        "label": "Dinlenme",
        "state_filter": ["rest"],
        "description": "Dinlenme fazı — egzersiz yok",
    },
    "grip": {
        "label": "Başarılı Kavrama",
        "state_filter": ["grip_hold"],
        "description": "Kavrama tutma ve güçlü kavrama segmenti",
    },
    "fatigue": {
        "label": "Yorgunluk",
        "state_filter": ["fatigue_attempt"],
        "description": "Yorgunluk denemesi ve yüksek efor segmenti",
    },
}


@dataclass
class SimulationConfig:
    scenario_key: str = "normal"
    speed_multiplier: float = 10.0
    data_path: Path = DEFAULT_DATA_PATH
    dataframe: pd.DataFrame | None = field(default=None, repr=False)


class GloveSimulator:
    def __init__(self, config: SimulationConfig | None = None):
        self.config = config or SimulationConfig()
        if self.config.dataframe is not None:
            self.full_df = self.config.dataframe.reset_index(drop=True)
        else:
            self.full_df = load_raw_data(self.config.data_path)
        self._load_scenario()

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        scenario_key: str = "normal",
        speed_multiplier: float = 10.0,
    ) -> "GloveSimulator":
        return cls(
            SimulationConfig(
                scenario_key=scenario_key,
                speed_multiplier=speed_multiplier,
                dataframe=df,
            )
        )

    def _load_scenario(self) -> None:
        scenario = SCENARIOS[self.config.scenario_key]
        state_filter = scenario.get("state_filter")
        if state_filter:
            mask = self.full_df["exercise_state"].isin(state_filter)
            self.df = self.full_df[mask].reset_index(drop=True)
            if self.df.empty:
                self.df = self.full_df.reset_index(drop=True)
        else:
            self.df = self.full_df.reset_index(drop=True)
        self.scenario_label = scenario["label"]
        self.scenario_description = scenario["description"]

    def set_scenario(self, scenario_key: str) -> None:
        self.config.scenario_key = scenario_key
        self._load_scenario()

    def set_speed(self, speed_multiplier: float) -> None:
        self.config.speed_multiplier = speed_multiplier

    def row_count(self) -> int:
        return len(self.df)

    def get_row(self, index: int) -> dict | None:
        if index < 0 or index >= len(self.df):
            return None
        return self.df.iloc[index].to_dict()

    def row_to_sensor_dict(self, row: dict) -> dict:
        """Buffer için sensör sözlüğü."""
        return {col: row[col] for col in SENSOR_COLS if col in row}

    def delay_seconds(self) -> float:
        """Satırlar arası gecikme (100 ms baz)."""
        return 0.1 / max(self.config.speed_multiplier, 0.1)

    def progress_pct(self, index: int) -> float:
        if len(self.df) == 0:
            return 0.0
        return (index + 1) / len(self.df) * 100
