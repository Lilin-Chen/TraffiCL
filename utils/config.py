from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from data.base_adapter import load_config


def deep_update(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_many(paths: list[str | Path]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in paths:
        merged = deep_update(merged, load_config(path))
    return merged

