from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from data.schema import TrafficDataset


@dataclass(frozen=True)
class DebugOptions:
    enabled: bool = False
    start_date: pd.Timestamp | None = None
    end_date: pd.Timestamp | None = None
    max_days: int | None = None
    max_series: int | None = None
    series_selection: str = "first"


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8-sig")
    try:
        import yaml

        loaded = yaml.safe_load(text)
    except ModuleNotFoundError:
        loaded = json.loads(text)

    if not isinstance(loaded, dict):
        raise ValueError(f"config must be a mapping: {config_path}")
    return loaded


def parse_date(value: Any) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    return pd.Timestamp(value)


class BaseDatasetAdapter(ABC):
    """Base class for raw dataset adapters."""

    adapter_name: str = "base"

    def __init__(self, config: Mapping[str, Any]):
        self.config = dict(config)
        self.dataset_config = dict(self.config.get("dataset", {}))
        self.debug_config = dict(self.config.get("debug", {}))
        if not self.dataset_config:
            raise ValueError("config must contain a dataset section")

    @abstractmethod
    def load(self, debug: bool | None = None) -> TrafficDataset:
        """Load raw data into the unified TrafficDataset schema."""

    @property
    def dataset_name(self) -> str:
        return str(self.dataset_config.get("name", self.adapter_name))

    @property
    def raw_dir(self) -> Path:
        value = self.dataset_config.get("raw_dir")
        if not value:
            raise ValueError("dataset.raw_dir is required")
        return Path(str(value))

    def debug_options(self, debug: bool | None = None) -> DebugOptions:
        enabled = bool(self.debug_config.get("enabled", False))
        if debug is not None:
            enabled = bool(debug)

        if not enabled:
            return DebugOptions(enabled=False)

        return DebugOptions(
            enabled=True,
            start_date=parse_date(self.debug_config.get("start_date")),
            end_date=parse_date(self.debug_config.get("end_date")),
            max_days=self._optional_int(self.debug_config.get("max_days")),
            max_series=self._optional_int(self.debug_config.get("max_series")),
            series_selection=str(self.debug_config.get("series_selection", "first")),
        )

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value in (None, ""):
            return None
        parsed = int(value)
        if parsed <= 0:
            raise ValueError("debug limits must be positive")
        return parsed

    def require_raw_dir(self) -> Path:
        path = self.raw_dir
        if not path.exists():
            raise FileNotFoundError(f"raw data directory does not exist: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"raw data path is not a directory: {path}")
        return path
