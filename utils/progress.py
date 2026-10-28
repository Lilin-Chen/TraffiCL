from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LogProgressBar:
    label: str
    total: int
    width: int = 24
    min_interval_seconds: float = 20.0
    percent_step: float = 5.0
    started_at: float = field(default_factory=time.monotonic)
    _last_printed_at: float = field(init=False)
    _last_percent_bucket: int = field(default=-1, init=False)

    def __post_init__(self) -> None:
        self.total = max(1, int(self.total))
        self._last_printed_at = self.started_at
        self._print(0, {})

    def update(self, completed: int, **metrics: Any) -> None:
        completed = min(max(0, int(completed)), self.total)
        now = time.monotonic()
        percent = 100.0 * completed / self.total
        bucket = int(percent // max(self.percent_step, 0.1))
        should_print = (
            completed >= self.total
            or bucket > self._last_percent_bucket
            or now - self._last_printed_at >= self.min_interval_seconds
        )
        if not should_print:
            return
        self._print(completed, metrics, now=now)
        self._last_percent_bucket = bucket
        self._last_printed_at = now

    def _print(
        self,
        completed: int,
        metrics: dict[str, Any],
        *,
        now: float | None = None,
    ) -> None:
        now = time.monotonic() if now is None else now
        elapsed = max(0.0, now - self.started_at)
        ratio = completed / self.total
        filled = min(self.width, int(self.width * ratio))
        bar = "#" * filled + "." * (self.width - filled)
        eta = (
            elapsed * (self.total - completed) / completed
            if completed > 0
            else math.inf
        )
        metric_text = " ".join(
            f"{name}={float(value):.6f}"
            for name, value in metrics.items()
            if value is not None and math.isfinite(float(value))
        )
        suffix = f" {metric_text}" if metric_text else ""
        print(
            f"PROGRESS {self.label} [{bar}] {100.0 * ratio:6.2f}% "
            f"batch={completed}/{self.total}{suffix} "
            f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}",
            flush=True,
        )


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "--:--:--"
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
