"""
NeonVision AI - Hand Tracker & Gesture Engine
=============================================

Wrapper around the MediaPipe Hands solution plus the gesture-recognition
layer.

Finger-extension detection is deliberately **rotation invariant**: instead of
the common ``tip.y < pip.y`` shortcut (which breaks the moment the hand is
tilted or upside down), each finger is classified by comparing its
fingertip-to-wrist distance against its knuckle-to-wrist distance, normalised
by palm size. The thumb is measured against the pinky knuckle, because its
range of motion is lateral rather than radial.

A short majority-vote history per hand removes the single-frame flicker that
makes raw per-frame counting look unstable on screen.

The MediaPipe API differences are absorbed by :mod:`core.mp_runtime`, so this
module works unchanged on both the classic ``solutions`` graphs and the newer
``tasks`` landmarkers.

Author: ChamathDilshanC
Project: NeonVision AI
License: MIT
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Dict, Final, List, Optional, Tuple

import cv2
import numpy as np

from core.mp_runtime import HandBackend, hand_connection_table

Frame = np.ndarray
FingerStates = Tuple[bool, bool, bool, bool, bool]


class HandLandmarkIndex:
    """Named indices into the 21-point MediaPipe hand skeleton."""

    WRIST: Final[int] = 0
    THUMB_CMC: Final[int] = 1
    THUMB_MCP: Final[int] = 2
    THUMB_IP: Final[int] = 3
    THUMB_TIP: Final[int] = 4
    INDEX_MCP: Final[int] = 5
    INDEX_PIP: Final[int] = 6
    INDEX_DIP: Final[int] = 7
    INDEX_TIP: Final[int] = 8
    MIDDLE_MCP: Final[int] = 9
    MIDDLE_PIP: Final[int] = 10
    MIDDLE_DIP: Final[int] = 11
    MIDDLE_TIP: Final[int] = 12
    RING_MCP: Final[int] = 13
    RING_PIP: Final[int] = 14
    RING_DIP: Final[int] = 15
    RING_TIP: Final[int] = 16
    PINKY_MCP: Final[int] = 17
    PINKY_PIP: Final[int] = 18
    PINKY_DIP: Final[int] = 19
    PINKY_TIP: Final[int] = 20

    #: ``(tip, pip)`` pairs for index, middle, ring and pinky.
    FINGER_JOINTS: Final[Tuple[Tuple[int, int], ...]] = (
        (INDEX_TIP, INDEX_PIP),
        (MIDDLE_TIP, MIDDLE_PIP),
        (RING_TIP, RING_PIP),
        (PINKY_TIP, PINKY_PIP),
    )

    FINGER_NAMES: Final[Tuple[str, ...]] = (
        "Thumb",
        "Index",
        "Middle",
        "Ring",
        "Pinky",
    )


#: Finger-state patterns mapped to human-readable gesture names.
#: Order is ``(thumb, index, middle, ring, pinky)``.
_GESTURE_TABLE: Final[Dict[FingerStates, str]] = {
    (False, False, False, False, False): "Fist",
    (True, True, True, True, True): "Open Palm",
    (False, True, True, False, False): "Peace",
    (True, False, False, False, False): "Thumbs Up",
    (False, True, False, False, False): "Pointing",
    (True, True, False, False, False): "Gun",
    (False, True, False, False, True): "Rock On",
    (True, False, False, False, True): "Call Me",
    (False, True, True, True, False): "Three Up",
    (False, True, True, True, True): "Four Up",
    (True, True, True, False, False): "Trio",
    (False, False, False, False, True): "Pinky Out",
}

#: Landmark connections of the hand skeleton, as an ``(N, 2)`` index array.
HAND_CONNECTIONS: Final[np.ndarray] = hand_connection_table()

#: Fingertip landmarks, drawn as accent nodes by the renderer.
FINGERTIPS: Final[np.ndarray] = np.asarray(
    [
        HandLandmarkIndex.THUMB_TIP,
        HandLandmarkIndex.INDEX_TIP,
        HandLandmarkIndex.MIDDLE_TIP,
        HandLandmarkIndex.RING_TIP,
        HandLandmarkIndex.PINKY_TIP,
    ],
    dtype=np.int32,
)


@dataclass(slots=True)
class HandResult:
    """Landmarks and gesture state for a single detected hand.

    Attributes:
        label: ``"Left"`` or ``"Right"`` as reported by MediaPipe (already
            correct for a mirrored preview frame).
        score: Handedness classification confidence in ``[0, 1]``.
        points: Pixel coordinates, ``int32`` array of shape ``(21, 2)``.
        normalized: Raw MediaPipe landmarks, ``float32`` ``(21, 3)``.
        frame_size: ``(width, height)`` of the source frame.
        fingers_up: Per-finger extension flags in thumb-to-pinky order.
        gesture: Recognised gesture name, e.g. ``"Peace"``.
    """

    label: str
    score: float
    points: np.ndarray
    normalized: np.ndarray
    frame_size: Tuple[int, int]
    fingers_up: FingerStates = (False,) * 5
    gesture: str = "Unknown"

    @property
    def finger_count(self) -> int:
        """How many fingers are currently extended (0-5)."""
        return int(sum(self.fingers_up))

    @property
    def extended_finger_names(self) -> List[str]:
        """Names of the extended fingers, thumb first."""
        return [
            name
            for name, up in zip(HandLandmarkIndex.FINGER_NAMES, self.fingers_up)
            if up
        ]

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Axis-aligned bounding box as ``(x, y, width, height)``."""
        x_min, y_min = self.points.min(axis=0)
        x_max, y_max = self.points.max(axis=0)
        return int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)

    @property
    def palm_size(self) -> float:
        """Wrist-to-middle-knuckle distance in pixels (a scale reference)."""
        return float(
            np.linalg.norm(
                self.points[HandLandmarkIndex.MIDDLE_MCP]
                - self.points[HandLandmarkIndex.WRIST]
            )
        )

    def point(self, index: int) -> Tuple[int, int]:
        """Return a single landmark as an ``(x, y)`` pixel tuple."""
        x, y = self.points[index]
        return int(x), int(y)

    def segments(self, connections: np.ndarray = HAND_CONNECTIONS) -> np.ndarray:
        """Polyline batch for the hand skeleton, shape ``(N, 2, 2)``."""
        return self.points[connections]

    def scaled_segments(
        self, scale: float, connections: np.ndarray = HAND_CONNECTIONS
    ) -> np.ndarray:
        """Skeleton segments in a down-scaled space, for the glow pass."""
        scaled = self.normalized[:, :2] * (
            self.frame_size[0] * scale,
            self.frame_size[1] * scale,
        )
        return scaled.astype(np.int32)[connections]


class HandTracker:
    """MediaPipe Hands front-end with finger counting and gesture naming.

    Args:
        max_hands: Maximum number of hands to track per frame.
        detection_confidence: Minimum confidence for the palm detector.
        tracking_confidence: Minimum confidence for landmark tracking before
            detection is re-run.
        model_complexity: ``0`` (fast) or ``1`` (accurate). ``0`` buys roughly
            4-6 ms per frame on a mid-range CPU with a small accuracy cost.
        static_image_mode: Treat every frame as unrelated; leave ``False`` for
            video.
        smoothing_window: Number of frames used for the majority vote that
            stabilises the reported gesture. ``1`` disables smoothing.
    """

    #: Extension thresholds, expressed as tip/joint distance ratios.
    FINGER_RATIO: Final[float] = 1.12
    THUMB_RATIO: Final[float] = 1.10
    #: Fingertip proximity (as a fraction of palm size) for the OK sign.
    OK_PINCH_RATIO: Final[float] = 0.42

    def __init__(
        self,
        max_hands: int = 2,
        detection_confidence: float = 0.6,
        tracking_confidence: float = 0.5,
        model_complexity: int = 1,
        static_image_mode: bool = False,
        smoothing_window: int = 5,
    ) -> None:
        self.max_hands = int(max_hands)
        self.detection_confidence = float(detection_confidence)
        self.tracking_confidence = float(tracking_confidence)
        self.model_complexity = int(model_complexity)
        self.static_image_mode = bool(static_image_mode)
        self.smoothing_window = max(1, int(smoothing_window))

        self._backend = HandBackend(
            max_hands=self.max_hands,
            detection_confidence=self.detection_confidence,
            tracking_confidence=self.tracking_confidence,
            model_complexity=self.model_complexity,
            static_image_mode=self.static_image_mode,
        )
        self._closed = False
        self._last_results: List[HandResult] = []
        self._history: Dict[str, Deque[FingerStates]] = {}

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def process(
        self,
        frame_bgr: Frame,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> List[HandResult]:
        """Run hand inference and gesture classification on one BGR frame.

        Args:
            frame_bgr: Frame straight from the camera (BGR, ``uint8``).
            output_size: ``(width, height)`` the pixel landmarks should be
                mapped into. Pass the *display* frame size when inference runs
                on a down-scaled copy.

        Returns:
            One :class:`HandResult` per detected hand; empty when no hand is
            visible.
        """
        if self._closed:
            raise RuntimeError("HandTracker has already been closed")

        if output_size is not None:
            width, height = int(output_size[0]), int(output_size[1])
        else:
            height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results: List[HandResult] = []
        for slot, (label, score, normalized) in enumerate(
            self._backend.process(rgb)
        ):
            hand = HandResult(
                label=label,
                score=score,
                points=_to_pixels(normalized, width, height),
                normalized=normalized,
                frame_size=(width, height),
            )
            raw_states = self._classify_fingers(hand)
            # Key the vote history by handedness *and* slot, so two hands
            # carrying the same label cannot share a buffer.
            hand.fingers_up = self._smooth(f"{label}:{slot}", raw_states)
            hand.gesture = self._name_gesture(hand)
            results.append(hand)

        if not results:
            self._history.clear()
        self._last_results = results
        return results

    # ------------------------------------------------------------------ #
    # Gesture logic
    # ------------------------------------------------------------------ #
    def _classify_fingers(self, hand: HandResult) -> FingerStates:
        """Decide which fingers are extended, independent of hand rotation.

        Args:
            hand: Hand whose landmark array should be classified.

        Returns:
            Extension flags in ``(thumb, index, middle, ring, pinky)`` order.
        """
        points = hand.normalized[:, :2].astype(np.float64)
        wrist = points[HandLandmarkIndex.WRIST]
        palm = float(np.linalg.norm(points[HandLandmarkIndex.MIDDLE_MCP] - wrist))
        if palm < 1e-6:  # degenerate detection
            return (False,) * 5

        # Thumb: lateral motion, so measure against the opposite knuckle.
        pinky_mcp = points[HandLandmarkIndex.PINKY_MCP]
        tip_span = float(
            np.linalg.norm(points[HandLandmarkIndex.THUMB_TIP] - pinky_mcp)
        )
        base_span = float(
            np.linalg.norm(points[HandLandmarkIndex.THUMB_MCP] - pinky_mcp)
        )
        thumb_up = base_span > 1e-6 and (tip_span / base_span) > self.THUMB_RATIO

        states: List[bool] = [thumb_up]
        for tip_index, pip_index in HandLandmarkIndex.FINGER_JOINTS:
            tip_distance = float(np.linalg.norm(points[tip_index] - wrist))
            pip_distance = float(np.linalg.norm(points[pip_index] - wrist))
            states.append(
                pip_distance > 1e-6
                and (tip_distance / pip_distance) > self.FINGER_RATIO
            )

        thumb, index, middle, ring, pinky = states
        return (thumb, index, middle, ring, pinky)

    def _smooth(self, key: str, states: FingerStates) -> FingerStates:
        """Majority-vote the last ``smoothing_window`` classifications."""
        if self.smoothing_window == 1:
            return states
        history = self._history.setdefault(
            key, deque(maxlen=self.smoothing_window)
        )
        history.append(states)
        most_common = Counter(history).most_common(1)[0][0]
        return most_common

    def _name_gesture(self, hand: HandResult) -> str:
        """Map finger states (plus a pinch test) to a gesture name."""
        states = hand.fingers_up
        if self._is_ok_sign(hand):
            return "OK"
        named = _GESTURE_TABLE.get(states)
        if named is not None:
            return named
        count = int(sum(states))
        return f"{count} Finger" if count == 1 else f"{count} Fingers"

    def _is_ok_sign(self, hand: HandResult) -> bool:
        """Detect the OK sign: thumb and index pinched, other fingers up."""
        _, _, middle, ring, pinky = hand.fingers_up
        if not (middle and ring and pinky):
            return False
        palm = hand.palm_size
        if palm < 1e-6:
            return False
        pinch = float(
            np.linalg.norm(
                hand.points[HandLandmarkIndex.THUMB_TIP]
                - hand.points[HandLandmarkIndex.INDEX_TIP]
            )
        )
        return (pinch / palm) < self.OK_PINCH_RATIO

    # ------------------------------------------------------------------ #
    # Introspection / lifecycle
    # ------------------------------------------------------------------ #
    @property
    def last_results(self) -> List[HandResult]:
        """Results from the most recent :meth:`process` call."""
        return self._last_results

    @property
    def hand_count(self) -> int:
        """Number of hands found in the most recent frame."""
        return len(self._last_results)

    @property
    def total_fingers(self) -> int:
        """Combined extended-finger count across every tracked hand."""
        return sum(hand.finger_count for hand in self._last_results)

    def gesture_summary(self) -> str:
        """One-line description of the current gesture state, for the HUD."""
        if not self._last_results:
            return "No Hand Detected"
        parts = [
            f"{hand.label[0]}: {hand.gesture}" for hand in self._last_results
        ]
        return "  |  ".join(parts)

    @property
    def backend(self) -> str:
        """Identifier of the MediaPipe API in use."""
        return self._backend.backend

    def close(self) -> None:
        """Release the MediaPipe graph."""
        if not self._closed:
            self._backend.close()
            self._closed = True

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<HandTracker max_hands={self.max_hands} "
            f"complexity={self.model_complexity} backend={self.backend}>"
        )


def _to_pixels(normalized: np.ndarray, width: int, height: int) -> np.ndarray:
    """Project normalized landmarks onto a pixel grid.

    Args:
        normalized: ``(N, 3)`` float32 array of normalized landmarks.
        width: Target width in pixels.
        height: Target height in pixels.

    Returns:
        ``int32`` array of shape ``(N, 2)``.
    """
    points = np.empty((normalized.shape[0], 2), dtype=np.int32)
    np.multiply(normalized[:, 0], width, out=points[:, 0], casting="unsafe")
    np.multiply(normalized[:, 1], height, out=points[:, 1], casting="unsafe")
    return points
