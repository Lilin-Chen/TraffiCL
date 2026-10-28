from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd

from data.schema import TrafficDataset, WindowBatch


@dataclass(frozen=True)
class WindowConfig:
    input_length: int = 12
    horizon: int = 12
    stride: int = 1
    input_columns: tuple[str, ...] = ("flow",)
    target_column: str = "flow"
    fill_method: str = "ffill_zero"
    fill_value: float = 0.0


class SlidingWindowGenerator:
    def __init__(self, config: WindowConfig):
        if config.input_length <= 0:
            raise ValueError("input_length must be positive")
        if config.horizon <= 0:
            raise ValueError("horizon must be positive")
        if config.stride <= 0:
            raise ValueError("stride must be positive")
        self.config = config

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None = None) -> "SlidingWindowGenerator":
        config = dict(config or {})
        input_columns = config.get("input_columns", config.get("feature_columns", ["flow"]))
        return cls(
            WindowConfig(
                input_length=int(config.get("input_length", 12)),
                horizon=int(config.get("horizon", 12)),
                stride=int(config.get("stride", 1)),
                input_columns=tuple(input_columns),
                target_column=str(config.get("target_column", "flow")),
                fill_method=str(config.get("fill_method", "ffill_zero")),
                fill_value=float(config.get("fill_value", 0.0)),
            )
        )

    def count_windows(self, dataset: TrafficDataset) -> int:
        timesteps = dataset.num_timesteps
        total_length = self.config.input_length + self.config.horizon
        if timesteps < total_length:
            return 0
        return ((timesteps - total_length) // self.config.stride) + 1

    def make_windows(
        self,
        dataset: TrafficDataset,
        *,
        max_windows: int | None = None,
    ) -> WindowBatch:
        starts = list(self._window_starts(dataset))
        if max_windows is not None:
            starts = starts[:max_windows]
        if not starts:
            raise ValueError("dataset does not contain enough timesteps for one window")

        return self._make_window_batch(dataset, starts)

    def iter_windows(self, dataset: TrafficDataset) -> Iterator[WindowBatch]:
        for start in self._window_starts(dataset):
            yield self._make_window_batch(dataset, [start])

    def _make_window_batch(self, dataset: TrafficDataset, starts: list[int]) -> WindowBatch:
        input_panel = dataset.to_panel(
            self.config.input_columns,
            fill_method=self.config.fill_method,
            fill_value=self.config.fill_value,
        )
        target_panel = dataset.to_panel(
            [self.config.target_column],
            fill_method=self.config.fill_method,
            fill_value=self.config.fill_value,
        )
        target_values = target_panel.values[..., 0]

        x = []
        y = []
        for start in starts:
            input_end = start + self.config.input_length
            target_end = input_end + self.config.horizon
            x.append(input_panel.values[start:input_end])
            y.append(target_values[input_end:target_end])

        first_start = starts[0]
        first_input_end = first_start + self.config.input_length
        first_target_end = first_input_end + self.config.horizon
        return WindowBatch(
            x=np.stack(x, axis=0),
            y=np.stack(y, axis=0),
            x_timestamps=pd.DatetimeIndex(input_panel.timestamps[first_start:first_input_end]),
            y_timestamps=pd.DatetimeIndex(input_panel.timestamps[first_input_end:first_target_end]),
            series_ids=input_panel.series_ids,
            input_columns=list(self.config.input_columns),
            target_column=self.config.target_column,
            adjacency=None if dataset.adjacency is None else dataset.adjacency.copy(),
        )

    def _window_starts(self, dataset: TrafficDataset) -> range:
        total_length = self.config.input_length + self.config.horizon
        stop = dataset.num_timesteps - total_length + 1
        if stop <= 0:
            return range(0)
        return range(0, stop, self.config.stride)
