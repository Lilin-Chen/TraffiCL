"""Fail closed when a run would mix checkpoints with different data or settings."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence


DAILY_PROTOCOL = "target_disjoint_daily_v2"
DAILY_DATASETS = {
    "caltrans_bicycle", "caltrans_pedestrian", "dc_nova", "pvcgn_hangzhou",
    "pvcgn_shanghai", "shanghai_metroflow",
}


def validate_daily_cache(metadata: Mapping[str, Any], dataset: str) -> None:
    """Validate interval declarations before any formal daily cache is loaded.

    Array-level timestamp and mask checks remain in YearlyCachedWindowDataset.
    """
    if dataset not in DAILY_DATASETS and metadata.get("task_granularity") != "day":
        return
    if metadata.get("split_protocol") != DAILY_PROTOCOL:
        raise ValueError(
            f"{dataset}: formal daily-task metadata requires split_protocol={DAILY_PROTOCOL!r}; "
            "legacy caches can contain overlapping targets. Rebuild with "
            "scripts/build_daily_task_cache.py into nonpems_daily_task_v2."
        )
    if metadata.get("task_granularity") != "day":
        raise ValueError(f"{dataset}: v2 daily cache must declare task_granularity='day'")
    if metadata.get("timestamp_basis") not in {"source_timestamps", "synthetic_sample_order"}:
        raise ValueError(f"{dataset}: v2 cache must declare its timestamp_basis")
    periods = metadata.get("periods", [])
    if not periods:
        raise ValueError(f"{dataset}: v2 daily cache has no tasks")
    intervals = []
    names = set()
    for period in periods:
        name = period.get("task_name")
        if not name or name in names:
            raise ValueError(f"{dataset}: missing or duplicate daily task name: {name!r}")
        names.add(name)
        shards = period.get("shards", [])
        by_split = {item.get("split"): item for item in shards}
        if len(shards) != 3 or set(by_split) != {"train", "val", "test"}:
            raise ValueError(f"{name}: v2 requires exactly one train/val/test shard each")
        date = datetime.fromisoformat(str(period["anchor_date"])).replace(tzinfo=timezone.utc)
        day_start = int(date.timestamp()) * 1_000_000_000
        day_end = day_start + 86_400_000_000_000
        previous_upper = day_start
        for split in ("train", "val", "test"):
            shard = by_split[split]
            bounds = shard.get("target_bounds_ns")
            if (shard.get("kind") != "windows" or not isinstance(bounds, (list, tuple))
                    or len(bounds) != 2 or any(type(x) is not int for x in bounds)):
                raise ValueError(f"{name}/{split}: v2 requires window shards with integer target_bounds_ns")
            lower, upper = bounds
            if not (day_start <= lower < upper <= day_end) or lower < previous_upper:
                raise ValueError(f"{name}/{split}: target bounds overlap or cross the prediction day")
            previous_upper = upper
            intervals.append((lower, upper, name, split))
    intervals.sort()
    for left, right in zip(intervals, intervals[1:]):
        if right[0] < left[1]:
            raise ValueError(f"{dataset}: target bounds overlap across tasks: {left[2:]} and {right[2:]}")


def _json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _portable(value: Any, *, cache_root: str = "") -> Any:
    if isinstance(value, Mapping):
        return {str(k): _portable(v, cache_root=cache_root) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(v, cache_root=cache_root) for v in value]
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        text = value.replace("\\", "/")
        root = cache_root.replace("\\", "/").rstrip("/")
        if root and (text.lower() == root.lower() or text.lower().startswith(root.lower() + "/")):
            return "$DATASET_CACHE" + text[len(root):]
        for marker in ("/data/cache/", "/configs/"):
            position = text.lower().find(marker)
            if position >= 0:
                return "$PROJECT" + text[position:]
        if Path(text).is_absolute() or PureWindowsPath(text).is_absolute():
            return "$EXTERNAL/" + text.rsplit("/", 1)[-1]
    return value


def _resolve_shard_path(value: str, metadata_path: Path, original_root: str) -> Path:
    text = value.replace("\\", "/")
    root = original_root.replace("\\", "/").rstrip("/")
    if root and text.lower().startswith(root.lower() + "/"):
        return metadata_path.parent / text[len(root) + 1:]
    if not Path(text).is_absolute() and not PureWindowsPath(text).is_absolute():
        return metadata_path.parent / text
    return Path(value)


def build_run_provenance(*, metadata_by_dataset: Mapping[str, str | Path],
                         config_by_dataset: Mapping[str, Any], task_names: Sequence[str],
                         protocol: Any) -> dict[str, Any]:
    """Hash settings, relocated metadata, and actual shard bytes once at startup."""
    caches = {}
    for dataset, metadata_value in metadata_by_dataset.items():
        metadata_path = Path(metadata_value).resolve()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        original_root = str(metadata.get("cache_dir", metadata_path.parent))
        files = {}
        for period in metadata.get("periods", []):
            for shard in period.get("shards", []):
                value = str(shard["path"])
                path = _resolve_shard_path(value, metadata_path, original_root)
                key = str(_portable(value, cache_root=original_root))
                if key not in files:
                    if not path.is_file():
                        raise FileNotFoundError(f"Cannot establish cache provenance: missing shard {path}")
                    files[key] = _file_digest(path)
            if period.get("adjacency_path"):
                value = str(period["adjacency_path"])
                path = _resolve_shard_path(value, metadata_path, original_root)
                key = "adjacency:" + str(_portable(value, cache_root=original_root))
                if key not in files:
                    # The dataset loader permits missing optional adjacency files.
                    files[key] = _file_digest(path) if path.is_file() else "absent"
        caches[dataset] = {"metadata": _portable(metadata, cache_root=original_root), "shards_sha256": files}
    root = Path(__file__).resolve().parents[1]
    sources = sorted({*root.glob("models/**/*.py"), *root.glob("trainers/**/*.py"),
                      *root.glob("evaluation/**/*.py"), root / "utils/checkpointing.py",
                      root / "data/yearly_cached_dataset.py", root / "data/window_generator.py"})
    payload = {"schema": 1, "protocol": protocol, "task_names": list(task_names),
               "configs": _portable(config_by_dataset), "caches": caches,
               "implementation_sha256": {str(p.relative_to(root)).replace("\\", "/"): _file_digest(p)
                                         for p in sources}}
    return {"schema": 1, "sha256": _json_digest(payload), "payload": payload}


def validate_checkpoint_provenance(state: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    saved = state.get("run_provenance")
    if not isinstance(saved, Mapping) or saved.get("schema") != expected.get("schema"):
        raise RuntimeError("Checkpoint has no compatible run provenance. Start in a new checkpoint directory; "
                           "legacy checkpoints cannot be resumed against the repaired cache.")
    if saved.get("sha256") != expected.get("sha256"):
        raise RuntimeError("Checkpoint provenance mismatch: cache contents, task stream, configuration or "
                           "implementation changed. Restore the original run inputs or use a new checkpoint directory.")


def prepare_checkpoint_directory(checkpoint_dir: str | Path, provenance: Mapping[str, Any], *, resume: bool) -> None:
    """Guard also first-task epoch checkpoints, which trainers auto-load internally."""
    directory = Path(checkpoint_dir)
    manifest = directory / "run_provenance.json"
    checkpoints = [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in {".pt", ".pth", ".ckpt"}]
    if checkpoints:
        if not manifest.is_file():
            raise RuntimeError("Checkpoint directory contains legacy/unverified checkpoints without run_provenance.json: "
                               f"{directory}. Use a new empty checkpoint directory.")
        try:
            saved = json.loads(manifest.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise RuntimeError(f"Unreadable checkpoint provenance manifest: {manifest}") from exc
        validate_checkpoint_provenance({"run_provenance": saved}, provenance)
        if not resume:
            raise RuntimeError(f"Checkpoint directory already contains a run: {directory}. "
                               "Use --resume to continue it, or choose a new empty checkpoint directory.")
    directory.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(manifest)
