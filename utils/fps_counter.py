"""
NeonVision AI - FPS Counter
===========================

Frame-rate instrumentation for the render loop.

:class:`FPSCounter` reports two numbers that answer different questions:

* ``fps`` - a rolling average over a short window. This is the number worth
  putting on the HUD, because instantaneous ``1 / dt`` jitters far too much to
  read.
* ``instant_fps`` - the reciprocal of the last frame time, useful when hunting
  individual stalls.

It also exposes ``frame_time_ms`` and per-stage timings so bottlenecks can be
attributed to inference or rendering rather than guessed at.

Author: ChamathDilshanC
Project: NeonVision AI
License: MIT
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, Optional


class FPSCounter:
    """Rolling-average frame-rate meter with optional stage profiling.

    Args:
        window: Number of recent frames averaged for :attr:`fps`.
    """

    def __init__(self, window: int = 30) -> None:
        self.window = max(1, int(window))
        self._frame_times: Deque[float] = deque(maxlen=self.window)
        self._last_timestamp: Optional[float] = None
        self._start_timestamp: float = time.perf_counter()
        self._frame_count: int = 0
        self._stage_marks: Dict[str, float] = {}
        self._stage_times: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Per-frame updates
    # ------------------------------------------------------------------ #
    def update(self) -> float:
        """Record one rendered frame.

        Call this exactly once per iteration of the main loop.

        Returns:
            The current smoothed FPS value.
        """
        now = time.perf_counter()
        if self._last_timestamp is not None:
            delta = now - self._last_timestamp
            if delta > 0:
                self._frame_times.append(delta)
        self._last_timestamp = now
        self._frame_count += 1
        return self.fps

    def reset(self) -> None:
        """Clear all history and restart the session timer."""
        self._frame_times.clear()
        self._last_timestamp = None
        self._start_timestamp = time.perf_counter()
        self._frame_count = 0
        self._stage_marks.clear()
        self._stage_times.clear()

    # ------------------------------------------------------------------ #
    # Stage profiling
    # ------------------------------------------------------------------ #
    def stage_start(self, name: str) -> None:
        """Begin timing a named pipeline stage, e.g. ``"inference"``."""
        self._stage_marks[name] = time.perf_counter()

    def stage_end(self, name: str) -> float:
        """Finish timing a named stage.

        Args:
            name: The name passed to :meth:`stage_start`.

        Returns:
            Duration of the stage in milliseconds (``0.0`` if never started).
        """
        started = self._stage_marks.pop(name, None)
        if started is None:
            return 0.0
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        # Exponential smoothing keeps the readout legible frame to frame.
        previous = self._stage_times.get(name)
        self._stage_times[name] = (
            elapsed_ms if previous is None else previous * 0.9 + elapsed_ms * 0.1
        )
        return elapsed_ms

    def stage_ms(self, name: str) -> float:
        """Smoothed duration of a named stage, in milliseconds."""
        return self._stage_times.get(name, 0.0)

    # ------------------------------------------------------------------ #
    # Readouts
    # ------------------------------------------------------------------ #
    @property
    def fps(self) -> float:
        """Smoothed frames per second over the rolling window."""
        if not self._frame_times:
            return 0.0
        average = sum(self._frame_times) / len(self._frame_times)
        return 1.0 / average if average > 0 else 0.0

    @property
    def instant_fps(self) -> float:
        """Frames per second derived from the most recent frame only."""
        if not self._frame_times:
            return 0.0
        last = self._frame_times[-1]
        return 1.0 / last if last > 0 else 0.0

    @property
    def frame_time_ms(self) -> float:
        """Smoothed frame time in milliseconds."""
        current = self.fps
        return 1000.0 / current if current > 0 else 0.0

    @property
    def frame_count(self) -> int:
        """Total frames recorded since construction or the last reset."""
        return self._frame_count

    @property
    def elapsed(self) -> float:
        """Seconds elapsed since construction or the last reset."""
        return time.perf_counter() - self._start_timestamp

    @property
    def average_fps(self) -> float:
        """Mean FPS across the whole session, not just the window."""
        elapsed = self.elapsed
        return self._frame_count / elapsed if elapsed > 0 else 0.0

    def label(self) -> str:
        """Compact HUD string, e.g. ``"FPS 58 | 17.2 ms"``."""
        return f"FPS {self.fps:>4.0f} | {self.frame_time_ms:>4.1f} ms"

    def summary(self) -> str:
        """Multi-field session summary for shutdown logs."""
        return (
            f"frames={self._frame_count} "
            f"elapsed={self.elapsed:.1f}s "
            f"avg_fps={self.average_fps:.1f} "
            f"window_fps={self.fps:.1f}"
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<FPSCounter {self.fps:.1f} fps window={self.window}>"
