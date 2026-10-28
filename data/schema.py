from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd


def _sort_series_ids(values: Sequence[Any]) -> list[str]:
    ids = [str(value) for value in values]
    try:
        return sorted(ids, key=lambda item: (0, int(item)))
    except ValueError:
        return sorted(ids)


@dataclass(frozen=True)
class TimeSeriesPanel:
    """Dense panel with shape [time, series, feature]."""

    values: np.ndarray
    timestamps: pd.DatetimeIndex
    series_ids: list[str]
    columns: list[str]


@dataclass
class TrafficDataset:
    """Unified traffic forecasting dataset.

    The canonical table is long-form:
    timestamp | series_id | target/feature columns...
    """

    name: str
    frame: pd.DataFrame
    freq: str = "5min"
    target_column: str = "flow"
    feature_columns: Sequence[str] = field(default_factory=lambda: ("flow",))
    static_frame: pd.DataFrame | None = None
    adjacency: np.ndarray | None = None
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    REQUIRED_COLUMNS: ClassVar[tuple[str, str]] = ("timestamp", "series_id")

    def validate(self) -> "TrafficDataset":
        if not isinstance(self.frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        if self.frame.empty:
            raise ValueError("dataset frame is empty")

        required = set(self.REQUIRED_COLUMNS)
        required.add(self.target_column)
        required.update(self.feature_columns)
        missing = sorted(required.difference(self.frame.columns))
        if missing:
            raise ValueError(f"dataset is missing required columns: {missing}")

        if not pd.api.types.is_datetime64_any_dtype(self.frame["timestamp"]):
            self.frame = self.frame.copy()
            self.frame["timestamp"] = pd.to_datetime(self.frame["timestamp"])

        null_timestamp_count = int(self.frame["timestamp"].isna().sum())
        null_series_count = int(self.frame["series_id"].isna().sum())
        if null_timestamp_count or null_series_count:
            raise ValueError(
                "timestamp and series_id must be non-null "
                f"(timestamp nulls={null_timestamp_count}, series_id nulls={null_series_count})"
            )

        duplicate_mask = self.frame.duplicated(["timestamp", "series_id"])
        if bool(duplicate_mask.any()):
            duplicate_count = int(duplicate_mask.sum())
            raise ValueError(f"dataset has {duplicate_count} duplicated timestamp/series rows")

        self.frame = self.frame.sort_values(["timestamp", "series_id"]).reset_index(drop=True)
        self.frame["series_id"] = self.frame["series_id"].astype(str)
        return self

    @property
    def timestamps(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(sorted(self.frame["timestamp"].unique()))

    @property
    def series_ids(self) -> list[str]:
        return _sort_series_ids(self.frame["series_id"].unique())

    @property
    def num_timesteps(self) -> int:
        return len(self.timestamps)

    @property
    def num_series(self) -> int:
        return len(self.series_ids)

    @property
    def time_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        timestamps = self.timestamps
        return timestamps[0], timestamps[-1]

    def subset_by_time(
        self,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        *,
        include_end: bool = True,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "TrafficDataset":
        frame = self.frame
        mask = pd.Series(True, index=frame.index)
        if start is not None:
            mask &= frame["timestamp"] >= pd.Timestamp(start)
        if end is not None:
            end_ts = pd.Timestamp(end)
            mask &= frame["timestamp"] <= end_ts if include_end else frame["timestamp"] < end_ts

        next_metadata = dict(self.metadata)
        if metadata:
            next_metadata.update(metadata)
        return TrafficDataset(
            name=name or self.name,
            frame=frame.loc[mask].copy(),
            freq=self.freq,
            target_column=self.target_column,
            feature_columns=tuple(self.feature_columns),
            static_frame=None if self.static_frame is None else self.static_frame.copy(),
            adjacency=None if self.adjacency is None else self.adjacency.copy(),
            metadata=next_metadata,
        ).validate()

    def subset_by_timestamps(
        self,
        timestamps: Sequence[pd.Timestamp],
        *,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "TrafficDataset":
        timestamp_index = pd.DatetimeIndex(timestamps)
        frame = self.frame[self.frame["timestamp"].isin(timestamp_index)].copy()
        next_metadata = dict(self.metadata)
        if metadata:
            next_metadata.update(metadata)
        return TrafficDataset(
            name=name or self.name,
            frame=frame,
            freq=self.freq,
            target_column=self.target_column,
            feature_columns=tuple(self.feature_columns),
            static_frame=None if self.static_frame is None else self.static_frame.copy(),
            adjacency=None if self.adjacency is None else self.adjacency.copy(),
            metadata=next_metadata,
        ).validate()

    def to_wide(
        self,
        value_column: str | None = None,
        *,
        fill_method: str | None = "ffill_zero",
        fill_value: float = 0.0,
    ) -> pd.DataFrame:
        column = value_column or self.target_column
        if column not in self.frame.columns:
            raise ValueError(f"column {column!r} is not available in dataset")

        wide = self.frame.pivot(index="timestamp", columns="series_id", values=column)
        wide = wide.sort_index().reindex(columns=self.series_ids)

        if fill_method is None or fill_method == "none":
            return wide
        if fill_method == "zero":
            return wide.fillna(fill_value)
        if fill_method == "ffill":
            return wide.ffill().bfill()
        if fill_method == "ffill_zero":
            return wide.ffill().fillna(fill_value)
        raise ValueError(f"unsupported fill method: {fill_method}")

    def to_panel(
        self,
        columns: Sequence[str],
        *,
        fill_method: str | None = "ffill_zero",
        fill_value: float = 0.0,
    ) -> TimeSeriesPanel:
        if not columns:
            raise ValueError("columns must contain at least one feature")

        wide_frames = [
            self.to_wide(column, fill_method=fill_method, fill_value=fill_value)
            for column in columns
        ]
        timestamps = pd.DatetimeIndex(wide_frames[0].index)
        series_ids = list(wide_frames[0].columns.astype(str))
        values = np.stack([wide.to_numpy(dtype=np.float32) for wide in wide_frames], axis=-1)
        return TimeSeriesPanel(
            values=values,
            timestamps=timestamps,
            series_ids=series_ids,
            columns=list(columns),
        )

    def summary(self) -> dict[str, Any]:
        start, end = self.time_range
        target = self.frame[self.target_column]
        return {
            "name": self.name,
            "rows": int(len(self.frame)),
            "timesteps": int(self.num_timesteps),
            "series": int(self.num_series),
            "freq": self.freq,
            "start": str(start),
            "end": str(end),
            "target": self.target_column,
            "target_missing_ratio": float(target.isna().mean()),
        }


@dataclass
class TrafficTask:
    task_id: int
    name: str
    train: TrafficDataset
    val: TrafficDataset
    test: TrafficDataset
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "train_timesteps": self.train.num_timesteps,
            "val_timesteps": self.val.num_timesteps,
            "test_timesteps": self.test.num_timesteps,
            "train_rows": len(self.train.frame),
            "val_rows": len(self.val.frame),
            "test_rows": len(self.test.frame),
            "series": self.train.num_series,
        }


@dataclass(frozen=True)
class WindowBatch:
    x: np.ndarray
    y: np.ndarray
    x_timestamps: pd.DatetimeIndex
    y_timestamps: pd.DatetimeIndex
    series_ids: list[str]
    input_columns: list[str]
    target_column: str
    adjacency: np.ndarray | None = None
    y_mask: np.ndarray | None = None
