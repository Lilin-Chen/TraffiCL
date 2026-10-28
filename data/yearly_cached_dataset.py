from __future__ import annotations

import bisect
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from data.cached_dataset import normalize_ratios
from data.schema import WindowBatch
from data.window_generator import WindowConfig


@dataclass(frozen=True)
class YearlyShard:
    date: str
    path: Path
    timesteps: int
    series_count: int
    kind: str = "daily"
    split: str | None = None
    samples: int | None = None
    windows: int | None = None
    target_bounds_ns: tuple[int, int] | None = None


@dataclass(frozen=True)
class YearlyPeriod:
    year: int
    anchor_date: str
    series_ids: list[str]
    shards: list[YearlyShard]
    dataset_name: str | None = None
    task_name: str | None = None
    adjacency_path: Path | None = None
    station_set_strategy: str | None = None
    first_seen: list[str] | None = None

    @property
    def num_series(self) -> int:
        return len(self.series_ids)

    @property
    def name(self) -> str:
        if self.task_name:
            return self.task_name
        if self.dataset_name:
            return f"{self.dataset_name}_{self.year}"
        return str(self.year)


class UnusableYearlyPeriodError(ValueError):
    """A cached period cannot provide leakage-free train/val/test datasets."""


def validate_audited_window_arrays(x, y, mask, xtime, ytime, bounds) -> None:
    """Fail closed when a v2 window or its claimed temporal interval is malformed."""
    x, y, mask, xtime, ytime = map(np.asarray, (x, y, mask, xtime, ytime))
    if x.ndim != 4 or y.ndim != 3 or mask.shape != y.shape or x.shape[0] != y.shape[0] or x.shape[2] != y.shape[2]:
        raise ValueError("Audited cache arrays do not align with samples, nodes, and horizon")
    if xtime.shape != x.shape[:2] or ytime.shape != y.shape[:2] or not x.shape[1] or not y.shape[1]:
        raise ValueError("Target/input timestamps do not align with the forecasting horizon")
    if not np.isin(mask, [0, 1]).all():
        raise ValueError("Audited cache masks must be binary")
    for times in (xtime, ytime):
        if not np.issubdtype(times.dtype, np.integer) or np.any(times == np.iinfo(np.int64).min):
            raise ValueError("Audited cache timestamps must be valid integer nanoseconds")
        if (np.diff(times, axis=1) <= 0).any():
            raise ValueError("Temporal leakage: window timestamps must be strictly increasing")
    lower, upper = map(int, bounds)
    if lower >= upper:
        raise ValueError("Audited target interval is empty or reversed")
    valid = mask.astype(bool)
    escaped = valid & ((ytime < lower) | (ytime >= upper))[..., None]
    if escaped.any() or (xtime[:, -1] >= ytime[:, 0]).any():
        raise ValueError("Temporal leakage detected in audited cache")
    if ((ytime[:, 0] < lower) | (ytime[:, 0] >= upper)).any():
        raise ValueError("Forecast origin lies outside its assigned target interval")
    if not np.isfinite(y[valid]).all():
        raise ValueError("Audited cache contains non-finite supervised labels")
    if np.any(y[~valid] != 0):
        raise ValueError("Masked future labels must be zeroed in the safe cache")


class YearlyTrafficStore:
    """Yearly cache where each period has its own station set and shards."""

    def __init__(self, metadata_path: str | Path):
        self.metadata_path = Path(metadata_path)
        self.metadata = json.loads(
            self.metadata_path.read_text(
                encoding="utf-8-sig"
            )
        )
        original_cache = str(self.metadata["cache_dir"]).replace("\\", "/").rstrip("/")
        self.cache_dir = self.metadata_path.resolve().parent
        self.input_columns = list(self.metadata["input_columns"])
        self.target_column = str(self.metadata["target_column"])
        def relocated(value: Path | None) -> Path | None:
            if value is None:
                return None
            text = str(value).replace("\\", "/")
            if text.lower().startswith(original_cache.lower() + "/"):
                candidate = self.cache_dir / text[len(original_cache) + 1:]
            elif not value.is_absolute() and not (len(text) > 1 and text[1] == ":"):
                candidate = self.cache_dir / text
            else:
                return value
            return candidate
        self.periods = []
        for item in self.metadata["periods"]:
            period = self._parse_period(item)
            self.periods.append(replace(
                period, adjacency_path=relocated(period.adjacency_path),
                shards=[replace(shard, path=relocated(shard.path)) for shard in period.shards],
            ))
        if not self.periods:
            raise ValueError(f"yearly cache contains no periods: {metadata_path}")
        if self.metadata.get("split_protocol") == "target_disjoint_daily_v2":
            self._validate_daily_target_intervals()

    def _validate_daily_target_intervals(self) -> None:
        names = set()
        dates = set()
        for period in self.periods:
            day = pd.Timestamp(period.anchor_date).normalize()
            if period.name in names or day.value in dates:
                raise ValueError("Audited daily cache contains duplicate task names or dates")
            names.add(period.name)
            dates.add(day.value)
            by_split = {shard.split: shard for shard in period.shards}
            if len(period.shards) != 3 or set(by_split) != {"train", "val", "test"}:
                raise ValueError("Audited daily tasks require one train/val/test shard each")
            previous = day.value
            for split in ("train", "val", "test"):
                shard = by_split[split]
                if shard.kind != "windows" or shard.target_bounds_ns is None or len(shard.target_bounds_ns) != 2:
                    raise ValueError("Audited daily cache is missing target interval metadata")
                lower, upper = shard.target_bounds_ns
                if lower != previous or upper <= lower or pd.Timestamp(shard.date).normalize() != day:
                    raise ValueError("Audited target intervals must partition their prediction day")
                previous = upper
            if previous != (day + pd.Timedelta(days=1)).value:
                raise ValueError("Audited target intervals must end at the next day boundary")

    @property
    def name(self) -> str:
        return str(self.metadata.get("dataset_name", self.metadata_path.parent.name))

    @property
    def input_dim(self) -> int:
        return len(self.input_columns)

    def period_by_year(self, year: int) -> YearlyPeriod:
        for period in self.periods:
            if period.year == year:
                return period
        raise KeyError(f"year not found in cache: {year}")

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "periods": len(self.periods),
            "years": [period.year for period in self.periods],
            "task_names": [period.name for period in self.periods],
            "series_by_year": {str(period.year): period.num_series for period in self.periods},
            "days_by_year": {str(period.year): len(period.shards) for period in self.periods},
            "series_by_task": {period.name: period.num_series for period in self.periods},
            "days_by_task": {period.name: len(period.shards) for period in self.periods},
            "input_dim": self.input_dim,
            "station_transitions": self._station_transitions(),
            "metadata": str(self.metadata_path),
        }

    def _station_transitions(self) -> list[dict[str, Any]]:
        transitions = []
        for previous, current in zip(self.periods, self.periods[1:]):
            if previous.dataset_name != current.dataset_name:
                continue
            previous_ids = set(previous.series_ids)
            current_ids = set(current.series_ids)
            transitions.append(
                {
                    "dataset": current.dataset_name,
                    "previous_year": previous.year,
                    "current_year": current.year,
                    "retained": len(previous_ids & current_ids),
                    "new": len(current_ids - previous_ids),
                    "disappeared": len(previous_ids - current_ids),
                }
            )
        return transitions

    @staticmethod
    def _parse_period(item: Mapping[str, Any]) -> YearlyPeriod:
        adjacency_path = item.get("adjacency_path")
        return YearlyPeriod(
            year=int(item["year"]),
            anchor_date=str(item["anchor_date"]),
            series_ids=[str(value) for value in item["series_ids"]],
            dataset_name=str(item["dataset_name"]) if item.get("dataset_name") else None,
            task_name=str(item["task_name"]) if item.get("task_name") else None,
            adjacency_path=Path(str(adjacency_path)) if adjacency_path else None,
            station_set_strategy=(
                str(item["station_set_strategy"])
                if item.get("station_set_strategy")
                else None
            ),
            first_seen=[str(value) for value in item.get("first_seen", [])] or None,
            shards=[
                YearlyShard(
                    date=str(shard["date"]),
                    path=Path(str(shard["path"])),
                    timesteps=int(shard["timesteps"]),
                    series_count=int(shard["series_count"]),
                    kind=str(shard.get("kind", "daily")),
                    split=str(shard["split"]) if shard.get("split") else None,
                    samples=int(shard["samples"]) if shard.get("samples") is not None else None,
                    windows=int(shard["windows"]) if shard.get("windows") is not None else None,
                    target_bounds_ns=tuple(map(int, shard["target_bounds_ns"])) if shard.get("target_bounds_ns") else None,
                )
                for shard in item["shards"]
            ],
        )


class YearlyCachedWindowDataset(Dataset):
    def __init__(
        self,
        period: YearlyPeriod,
        shard_indices: Sequence[int],
        window_config: WindowConfig,
        *,
        input_columns: Sequence[str],
        target_column: str,
        max_windows: int | None = None,
    ):
        self.period = period
        self.shard_indices = list(shard_indices)
        self.window_config = window_config
        self.input_columns = list(input_columns)
        self.target_column = target_column
        self._loaded_shard_index: int | None = None
        self._loaded_values: np.ndarray | None = None
        self._loaded_target: np.ndarray | None = None
        self._loaded_active_mask: np.ndarray | None = None
        self._loaded_observed_mask: np.ndarray | None = None
        self._loaded_window_starts: np.ndarray | None = None
        self.adjacency = self._load_adjacency()

        total_length = window_config.input_length + window_config.horizon
        counts = []
        for shard_index in self.shard_indices:
            shard = period.shards[shard_index]
            if shard.kind == "windows":
                counts.append(int(shard.samples if shard.samples is not None else shard.timesteps))
            elif shard.windows is not None:
                counts.append(int(shard.windows))
            else:
                counts.append(max(0, ((shard.timesteps - total_length) // window_config.stride) + 1))
        self._counts = counts
        self._cumulative = np.cumsum(counts).tolist()
        self._length = int(sum(counts))
        if max_windows is not None:
            self._length = min(self._length, int(max_windows))

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_position = bisect.bisect_right(self._cumulative, index)
        previous = 0 if shard_position == 0 else self._cumulative[shard_position - 1]
        local_window = index - previous
        shard_index = self.shard_indices[shard_position]
        shard = self.period.shards[shard_index]
        if shard.kind == "windows":
            values, target, _, observed_mask, _ = self._load_shard(shard_index)
            return (
                values[local_window].astype(np.float32),
                target[local_window].astype(np.float32),
                observed_mask[local_window].astype(np.float32),
            )

        values, target, active_mask, observed_mask, window_starts = self._load_shard(shard_index)

        start = int(window_starts[local_window])
        input_end = start + self.window_config.input_length
        target_end = input_end + self.window_config.horizon
        target_mask = observed_mask[input_end:target_end] & active_mask[input_end - 1][None, :]
        return (
            values[start:input_end].astype(np.float32),
            target[input_end:target_end].astype(np.float32),
            target_mask.astype(np.float32),
        )

    def sample_batch(self, max_windows: int = 8) -> WindowBatch:
        count = min(len(self), max_windows)
        if count <= 0:
            raise ValueError("yearly cached dataset has no windows")
        xs = []
        ys = []
        masks = []
        for index in range(count):
            x, y, mask = self[index]
            xs.append(x)
            ys.append(y)
            masks.append(mask)
        return WindowBatch(
            x=np.stack(xs, axis=0),
            y=np.stack(ys, axis=0),
            x_timestamps=pd.DatetimeIndex([]),
            y_timestamps=pd.DatetimeIndex([]),
            series_ids=self.node_keys,
            input_columns=list(self.input_columns),
            target_column=self.target_column,
            adjacency=None if self.adjacency is None else self.adjacency.copy(),
            y_mask=np.stack(masks, axis=0),
        )

    @property
    def node_keys(self) -> list[str]:
        prefix = self.period.dataset_name or self.period.name
        return [f"{prefix}:{series_id}" for series_id in self.period.series_ids]

    def _load_shard(
        self,
        shard_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._loaded_shard_index == shard_index:
            assert self._loaded_values is not None
            assert self._loaded_target is not None
            assert self._loaded_active_mask is not None
            assert self._loaded_observed_mask is not None
            assert self._loaded_window_starts is not None
            return (
                self._loaded_values,
                self._loaded_target,
                self._loaded_active_mask,
                self._loaded_observed_mask,
                self._loaded_window_starts,
            )

        shard = self.period.shards[shard_index]
        with np.load(shard.path) as data:
            if shard.kind == "windows":
                self._loaded_values = data["x"].astype(np.float32)
                self._loaded_target = data["y"].astype(np.float32)
                self._loaded_active_mask = np.ones_like(self._loaded_target, dtype=bool)
                self._loaded_observed_mask = (
                    data["y_mask"].astype(bool)
                    if "y_mask" in data.files
                    else np.ones_like(self._loaded_target, dtype=bool)
                )
                if shard.target_bounds_ns is not None:
                    if "ytime" not in data.files or "xtime" not in data.files or "y_mask" not in data.files:
                        raise ValueError(f"Audited cache is missing timestamps: {shard.path}")
                    validate_audited_window_arrays(self._loaded_values, self._loaded_target,
                                                   data["y_mask"], data["xtime"], data["ytime"],
                                                   shard.target_bounds_ns)
                self._loaded_window_starts = np.arange(len(self._loaded_target), dtype=np.int32)
            else:
                self._loaded_values = data["values"].astype(np.float32)
                self._loaded_target = data["target"].astype(np.float32)
                self._loaded_active_mask = (
                    data["active_mask"].astype(bool)
                    if "active_mask" in data.files
                    else np.ones_like(self._loaded_target, dtype=bool)
                )
                self._loaded_observed_mask = (
                    data["observed_mask"].astype(bool)
                    if "observed_mask" in data.files
                    else np.ones_like(self._loaded_target, dtype=bool)
                )
                if "window_starts" in data.files:
                    self._loaded_window_starts = data["window_starts"].astype(np.int32)
                else:
                    total_length = self.window_config.input_length + self.window_config.horizon
                    self._loaded_window_starts = np.arange(
                        0,
                        len(self._loaded_target) - total_length + 1,
                        self.window_config.stride,
                        dtype=np.int32,
                    )
        self._loaded_shard_index = shard_index
        return (
            self._loaded_values,
            self._loaded_target,
            self._loaded_active_mask,
            self._loaded_observed_mask,
            self._loaded_window_starts,
        )

    def _load_adjacency(self) -> np.ndarray | None:
        if self.period.adjacency_path is None:
            return None
        with np.load(self.period.adjacency_path) as data:
            key = "x" if "x" in data else next(iter(data.files))
            return data[key].astype(np.float32)


def yearly_shard_splits(
    period: YearlyPeriod,
    config: Mapping[str, Any],
) -> dict[str, list[int]]:
    task_config = dict(config.get("task_stream", {}))
    split_ratio = dict(task_config.get("split_ratio", {"train": 0.6, "val": 0.2, "test": 0.2}))
    ratios = normalize_ratios(split_ratio)
    indices = [
        index
        for index, shard in enumerate(period.shards)
        if known_shard_windows(shard) != 0
    ]
    if any(shard.split for shard in period.shards):
        split = {
            name: [index for index in indices if period.shards[index].split == name]
            for name in ("train", "val", "test")
        }
        if not all(split.values()):
            raise UnusableYearlyPeriodError(
                f"task {period.name} predefined splits must include non-empty train/val/test"
            )
        return split

    n = len(indices)
    if n < 3:
        raise UnusableYearlyPeriodError(
            f"task {period.name} needs at least 3 usable shards for train/val/test; "
            f"found {n} of {len(period.shards)}"
        )
    train_end = max(1, int(n * ratios["train"]))
    val_end = max(train_end + 1, train_end + int(n * ratios["val"]))
    val_end = min(val_end, n - 1)
    return {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }


def known_shard_windows(shard: YearlyShard) -> int | None:
    if shard.kind == "windows" and shard.samples is not None:
        return int(shard.samples)
    if shard.windows is not None:
        return int(shard.windows)
    if shard.samples is not None:
        return int(shard.samples)
    return None
