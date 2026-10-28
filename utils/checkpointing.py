from __future__ import annotations

import os
import uuid
import random
import numpy as np
from pathlib import Path
from typing import Any

import torch


def capture_rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def atomic_torch_save(state: Any, path: str | Path) -> None:
    """Write and validate a PyTorch checkpoint before replacing the prior file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        torch.save(state, temporary)
        torch.load(temporary, map_location="cpu", weights_only=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def try_load_torch(path: str | Path, *, map_location: Any) -> Any | None:
    checkpoint = Path(path)
    if not checkpoint.exists():
        return None
    try:
        return torch.load(checkpoint, map_location=map_location, weights_only=False)
    except (OSError, RuntimeError, EOFError, ValueError) as exc:
        print(f"IGNORE unreadable checkpoint={checkpoint}: {exc}", flush=True)
        return None
