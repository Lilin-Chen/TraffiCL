"""Cache helpers used by the TraffiCL global runner."""
from __future__ import annotations
from typing import Any
from data.yearly_cached_dataset import (
    YearlyPeriod, YearlyCachedWindowDataset, yearly_shard_splits, UnusableYearlyPeriodError,
)

def make_period_bundle(
    period: YearlyPeriod,
    config: dict,
    window_config,
    *,
    input_columns: list[str],
    target_column: str,
    max_windows: int | None,
) -> dict[str, Any]:
    split = yearly_shard_splits(period, config)
    dataset_kwargs = {
        "input_columns": input_columns,
        "target_column": target_column,
        "max_windows": max_windows,
    }
    bundle = {
        "period": period,
        "train": YearlyCachedWindowDataset(period, split["train"], window_config, **dataset_kwargs),
        "val": YearlyCachedWindowDataset(period, split["val"], window_config, **dataset_kwargs),
        "test": YearlyCachedWindowDataset(period, split["test"], window_config, **dataset_kwargs),
    }
    lengths = {name: len(bundle[name]) for name in ("train", "val", "test")}
    if not all(lengths.values()):
        raise UnusableYearlyPeriodError(
            f"task {period.name} has an empty data split: "
            + ", ".join(f"{name}={length}" for name, length in lengths.items())
        )
    return bundle

def safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
