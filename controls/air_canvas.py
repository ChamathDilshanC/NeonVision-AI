"""
NeonVision AI - Air Drawing Canvas
==================================

Turns the index fingertip into a neon brush.

The canvas stores *geometry*, never pixels: strokes are lists of points, which
the renderer draws through the same glow pipeline as the face mesh. Keeping
vectors instead of an image buffer means undo is trivial, strokes survive a
resolution change, and the paint picks up the current palette colour for free.

Pen state is read from the hand pose rather than a button:

* index finger alone extended -> **pen down**
* index and middle extended (Peace) -> **pen up**, move without drawing
* fist or open palm -> **pen up**

Author: ChamathDilshanC
Project: NeonVision AI
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, List, Optional, Tuple

import numpy as np

from core.hand_tracker import HandLandmarkIndex, HandResult

Color = Tuple[int, int, int]


@dataclass(slots=True)
class Stroke:
    """One continuous pen-down trace."""

    color: Color
    points: List[Tuple[int, int]] = field(default_factory=list)

    def as_polyline(self) -> np.ndarray:
        """Return the stroke shaped for ``cv2.polylines``: ``(1, N, 2)``."""
        return np.asarray(self.points, dtype=np.int32).reshape(1, -1, 2)

    @property
    def length(self) -> int:
        """Number of points in the stroke."""
        return len(self.points)


class AirCanvas:
    """Collects fingertip strokes while the pen-down pose is held.

    Args:
        min_step: Minimum pixel distance between stored points. Filters out
            sensor jitter and keeps stroke geometry compact.
        smoothing: Exponential smoothing of the fingertip in ``[0, 1)``.
            Higher is smoother but lags the finger.
        max_points: Global cap on stored points; the oldest stroke is dropped
            when exceeded, so a long session cannot grow without bound.
    """

    #: Poses that keep the pen up without ending the session.
    HOVER_POSES: Final[Tuple[str, ...]] = ("Peace", "Open Palm", "Fist")

    def __init__(
        self,
        min_step: float = 4.0,
        smoothing: float = 0.45,
        max_points: int = 12000,
    ) -> None:
        self.min_step = float(min_step)
        self.smoothing = float(min(0.95, max(0.0, smoothing)))
        self.max_points = int(max_points)

        self.strokes: List[Stroke] = []
        self._active: Optional[Stroke] = None
        self._tip: Optional[Tuple[float, float]] = None
        self._point_count = 0

    # ------------------------------------------------------------------ #
    # Frame update
    # ------------------------------------------------------------------ #
    def update(
        self, hand: Optional[HandResult], color: Color
    ) -> Optional[Tuple[int, int]]:
        """Advance the canvas by one frame.

        Args:
            hand: The drawing hand, or ``None`` when none is tracked.
            color: Colour for any new stroke, from the active palette.

        Returns:
            The smoothed fingertip position, or ``None`` if there is no hand.
        """
        if hand is None:
            self.lift()
            self._tip = None
            return None

        raw = hand.points[HandLandmarkIndex.INDEX_TIP]
        target = (float(raw[0]), float(raw[1]))
        if self._tip is None:
            self._tip = target
        else:
            alpha = 1.0 - self.smoothing
            self._tip = (
                self._tip[0] + (target[0] - self._tip[0]) * alpha,
                self._tip[1] + (target[1] - self._tip[1]) * alpha,
            )
        position = (int(self._tip[0]), int(self._tip[1]))

        if self.is_pen_down(hand):
            self._extend(position, color)
        else:
            self.lift()
        return position

    @staticmethod
    def is_pen_down(hand: HandResult) -> bool:
        """``True`` when the pose is "index finger only" (thumb ignored)."""
        _, index, middle, ring, pinky = hand.fingers_up
        return bool(index and not middle and not ring and not pinky)

    def _extend(self, position: Tuple[int, int], color: Color) -> None:
        """Append a point to the active stroke, starting one if needed."""
        if self._active is None:
            self._active = Stroke(color=color)
            self.strokes.append(self._active)

        points = self._active.points
        if points:
            previous = points[-1]
            step = (position[0] - previous[0]) ** 2 + (
                position[1] - previous[1]
            ) ** 2
            if step < self.min_step * self.min_step:
                return
        points.append(position)
        self._point_count += 1
        self._trim()

    def _trim(self) -> None:
        """Drop the oldest strokes once the point budget is exceeded."""
        while self._point_count > self.max_points and len(self.strokes) > 1:
            oldest = self.strokes.pop(0)
            self._point_count -= oldest.length
            if oldest is self._active:
                self._active = None

    def lift(self) -> None:
        """End the current stroke, discarding it if it is a single dot."""
        if self._active is None:
            return
        if self._active.length < 2:
            try:
                self.strokes.remove(self._active)
            except ValueError:  # pragma: no cover - already trimmed away
                pass
            else:
                self._point_count -= self._active.length
        self._active = None

    # ------------------------------------------------------------------ #
    # Editing
    # ------------------------------------------------------------------ #
    def clear(self) -> int:
        """Erase everything.

        Returns:
            The number of strokes removed.
        """
        removed = len(self.strokes)
        self.strokes.clear()
        self._active = None
        self._point_count = 0
        return removed

    def undo(self) -> bool:
        """Remove the most recent stroke.

        Returns:
            ``True`` if a stroke was removed.
        """
        self.lift()
        if not self.strokes:
            return False
        removed = self.strokes.pop()
        self._point_count -= removed.length
        return True

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def polylines(self) -> List[Tuple[np.ndarray, Color]]:
        """Drawable strokes as ``(polyline, colour)`` pairs.

        Single-point strokes are skipped: a polyline needs two points.
        """
        return [
            (stroke.as_polyline(), stroke.color)
            for stroke in self.strokes
            if stroke.length >= 2
        ]

    @property
    def is_drawing(self) -> bool:
        """``True`` while a stroke is being extended."""
        return self._active is not None

    @property
    def stroke_count(self) -> int:
        """Number of stored strokes."""
        return len(self.strokes)

    @property
    def point_count(self) -> int:
        """Total stored points across all strokes."""
        return self._point_count

    def status(self) -> str:
        """Compact HUD summary."""
        state = "DRAWING" if self.is_drawing else "PEN UP"
        return f"{state}   strokes {self.stroke_count}   points {self._point_count}"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<AirCanvas strokes={self.stroke_count} "
            f"points={self._point_count} drawing={self.is_drawing}>"
        )
