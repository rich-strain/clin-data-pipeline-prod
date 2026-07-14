"""Crash-safe caching + retry + call stats for the paid API-calling stages.

- `IncrementalCache` persists to disk on every write, so a run interrupted
  partway through (crash, Ctrl-C, rate-limit giveup) keeps every response it
  already paid for — a re-run resumes instead of re-paying.
- `call_with_retry` wraps an SDK call with bounded exponential backoff on
  transient errors (the SDK already retries 429/5xx twice; this adds a small
  outer margin and records the attempt count).
- `CallStats` records per-call latency + retry counts so a sequential-vs-batch
  comparison isn't opaque about whether a slow run hit a backoff.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TRANSIENT_HINTS = ("rate_limit", "overloaded", "timeout", "connection", "500", "502", "503", "529")


class IncrementalCache(MutableMapping):
    """Dict-like JSON cache that flushes to disk on every mutation."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._data: dict[str, Any] = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text())

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Write atomically so a crash mid-write can't corrupt the committed cache.
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, sort_keys=True))
        tmp.replace(self.path)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value
        self._flush()

    def __delitem__(self, key: str) -> None:
        del self._data[key]
        self._flush()

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


@dataclass
class CallStats:
    """Records literal API calls: count, total wall-time, retries."""

    calls: int = 0
    total_seconds: float = 0.0
    retries: int = 0
    latencies: list[float] = field(default_factory=list)

    def record(self, seconds: float, retries: int) -> None:
        self.calls += 1
        self.total_seconds += seconds
        self.retries += retries
        self.latencies.append(round(seconds, 3))

    def summary(self) -> dict:
        return {
            "calls": self.calls,
            "total_seconds": round(self.total_seconds, 2),
            "retries": self.retries,
            "avg_seconds": round(self.total_seconds / self.calls, 3) if self.calls else 0.0,
        }


def _is_transient(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(hint in text for hint in TRANSIENT_HINTS)


def call_with_retry(
    fn,
    *args,
    stats: CallStats | None = None,
    max_retries: int = 4,
    base_delay: float = 1.0,
    sleep=time.sleep,
    **kwargs,
):
    """Call `fn`, retrying transient errors with exponential backoff. Records
    one CallStats entry (latency + retry count) for the successful call."""
    start = time.monotonic()
    attempt = 0
    while True:
        try:
            result = fn(*args, **kwargs)
            if stats is not None:
                stats.record(time.monotonic() - start, attempt)
            return result
        except Exception as exc:  # noqa: BLE001 — classify, then re-raise if fatal
            if attempt >= max_retries or not _is_transient(exc):
                raise
            sleep(min(base_delay * (2**attempt), 30.0))
            attempt += 1
