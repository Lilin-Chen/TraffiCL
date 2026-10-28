from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from data.schema import WindowBatch
from data.window_generator import WindowConfig


@dataclass(frozen=True)
class CachedShard:
    date: str
    path: Path
    timesteps: int
    series_count: int
    windows: int | None = None


class CachedTrafficStore:
    def __init__(self, metadata_path: str | Path):
        self.metadata_path = Path(metadata_path)
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.cache_dir = Path(self.metadata["cache_dir"])
        self.series_ids = [str(value) for value in self.metadata["series_ids"]]
        self.input_columns = list(self.metadata["input_columns"])
        self.target_column = str(self.metadata["target_column"])
        self.shards = [
            CachedShard(
                date=str(item["date"]),
                path=Path(item["path"]),
                timesteps=int(item["timesteps"]),
                series_count=int(item["series_count"]),
                windows=int(item["windows"]) if item.get("windows") is not None else None,
            )
            for item in self.metadata["shards"]
        ]
        if not self.shards:
            raise ValueError(f"cache contains no shards: {metadata_path}")

    @property
    def name(self) -> str:
        return str(self.metadata.get("dataset_name", self.metadata_path.parent.name))

    @property
    def num_series(self) -> int:
        return len(self.series_ids)

    @property
    def input_dim(self) -> int:
        return len(self.input_columns)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "days": len(self.shards),
            "series": self.num_series,
            "input_dim": self.input_dim,
            "start": self.shards[0].date,
            "end": self.shards[-1].date,
            "metadata": str(self.metadata_path),
        }


class CachedWindowDataset(Dataset):
    def __init__(
        self,
        store: CachedTrafficStore,
        shard_indices: Sequence[int],
        window_config: WindowConfig,
        *,
        max_windows: int | None = None,
    ):
        self.store = store
        self.shard_indices = list(shard_indices)
        self.window_config = window_config
        self._loaded_shard_index: int | None = None
        self._loaded_values: np.ndarray | None = None
        self._loaded_target: np.ndarray | None = None
        self._loaded_active_mask: np.ndarray | None = None
        self._loaded_observed_mask: np.ndarray | None = None
        self._loaded_window_starts: np.ndarray | None = None

        counts = []
        total_length = window_config.input_length + window_config.horizon
        for shard_index in self.shard_indices:
            shard = store.shards[shard_index]
            count = (
                int(shard.windows)
                if shard.windows is not None
                else max(0, ((shard.timesteps - total_length) // window_config.stride) + 1)
            )
            counts.append(count)
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
        values, target, active_mask, observed_mask, window_starts = self._load_shard(shard_index)

        start = int(window_starts[local_window])
        input_end = start + self.window_config.input_length
        target_end = input_end + self.window_config.horizon
        x = values[start:input_end]
        y = target[input_end:target_end]
        y_mask = observed_mask[input_end:target_end] & active_mask[input_end - 1][None, :]
        return x.astype(np.float32), y.astype(np.float32), y_mask.astype(np.float32)

    def sample_batch(self, max_windows: int = 8) -> WindowBatch:
        count = min(len(self), max_windows)
        if count <= 0:
            raise ValueError("cached dataset has no windows")
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
            input_columns=list(self.store.input_columns),
            target_column=self.store.target_column,
            adjacency=None,
            y_mask=np.stack(masks, axis=0),
        )

    @property
    def node_keys(self) -> list[str]:
        return [f"{self.store.name}:{series_id}" for series_id in self.store.series_ids]

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

        shard = self.store.shards[shard_index]
        with np.load(shard.path) as data:
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


def cached_time_block_splits(
    store: CachedTrafficStore,
    config: Mapping[str, Any],
) -> list[dict[str, list[int]]]:
    task_config = dict(config.get("task_stream", {}))
    num_tasks = int(task_config.get("num_tasks", 3))
    split_ratio = dict(task_config.get("split_ratio", {"train": 0.6, "val": 0.2, "test": 0.2}))
    ratios = normalize_ratios(split_ratio)
    shard_blocks = [block.tolist() for block in np.array_split(np.arange(len(store.shards)), num_tasks)]
    tasks = []
    for block in shard_blocks:
        n = len(block)
        if n < 3:
            raise ValueError("each cached task needs at least 3 day shards")
        train_end = max(1, int(n * ratios["train"]))
        val_end = max(train_end + 1, train_end + int(n * ratios["val"]))
        val_end = min(val_end, n - 1)
        tasks.append(
            {
                "train": block[:train_end],
                "val": block[train_end:val_end],
                "test": block[val_end:],
            }
        )
    return tasks


def normalize_ratios(split_ratio: Mapping[str, float]) -> dict[str, float]:
    keys = ("train", "val", "test")
    values = {key: float(split_ratio.get(key, 0.0)) for key in keys}
    total = sum(values.values())
    if total <= 0:
        raise ValueError("split ratios must sum to a positive value")
    return {key: value / total for key, value in values.items()}
