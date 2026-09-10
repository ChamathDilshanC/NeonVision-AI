"""
NeonVision AI - Unified Vision Tracker
======================================

Single front-end over the face-mesh and hand pipelines.

:class:`VisionTracker` owns the two detectors plus the two scheduling tricks
that keep the loop above 30 FPS, so the application layer never has to think
about either:

* **Inference down-scaling.** MediaPipe reports normalized landmarks, so
  inference can run on a smaller copy of the frame while the overlay is still
  drawn at full resolution.
* **Idle-frame sampling for hands.** MediaPipe only pays for the palm
  detector while nothing is being tracked; once a hand is locked on, the much
  cheaper landmark path takes over. Sampling the idle case every n-th frame
  removes that cost floor without adding latency when a hand is present.

Every call returns one immutable :class:`TrackingFrame`, which is the only
thing the renderer and the control layer need.

Author: ChamathDilshanC
Project: NeonVision AI
License: MIT
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.face_mesh import FaceMeshDetector, FaceMeshResult
from core.hand_tracker import HandResult, HandTracker
from core.mp_runtime import backend_description

Frame = np.ndarray


@dataclass(slots=True)
class TrackingFrame:
    """Everything detected in one frame.

    Attributes:
        faces: Detected faces, most confident first.
        hands: Detected hands, most confident first.
        frame_size: ``(width, height)`` the landmarks are mapped into.
        inference_ms: Wall-clock cost of the inference stage.
        hands_sampled: ``False`` when hand inference was skipped this frame
            by the idle-stride scheduler.
    """

    faces: List[FaceMeshResult] = field(default_factory=list)
    hands: List[HandResult] = field(default_factory=list)
    frame_size: Tuple[int, int] = (0, 0)
    inference_ms: float = 0.0
    hands_sampled: bool = True

    @property
    def face(self) -> Optional[FaceMeshResult]:
        """The primary face, or ``None``."""
        return self.faces[0] if self.faces else None

    @property
    def hand(self) -> Optional[HandResult]:
        """The primary hand, or ``None``."""
        return self.hands[0] if self.hands else None

    @property
    def has_subject(self) -> bool:
        """``True`` when at least one face or hand is present."""
        return bool(self.faces or self.hands)

    @property
    def finger_count(self) -> int:
        """Extended fingers on the primary hand, or ``0``."""
        return self.hands[0].finger_count if self.hands else 0

    @property
    def total_fingers(self) -> int:
        """Extended fingers across every tracked hand."""
        return sum(hand.finger_count for hand in self.hands)

    @property
    def eye_aspect_ratio(self) -> Optional[float]:
        """Primary face's EAR, or ``None`` when no face is tracked."""
        face = self.face
        return face.eye_aspect_ratio() if face is not None else None

    def gesture_label(self) -> str:
        """Primary gesture readout, e.g. ``"Gesture: 3 Fingers"``."""
        if not self.hands:
            return "Gesture: --"
        count = self.hands[0].finger_count
        noun = "Finger" if count == 1 else "Fingers"
        return f"Gesture: {count} {noun}"

    def pose_summary(self) -> str:
        """Per-hand pose names, e.g. ``"R:Peace  L:Fist"``."""
        if not self.hands:
            return "no hand"
        return "  ".join(
            f"{hand.label[:1]}:{hand.gesture}" for hand in self.hands
        )


class VisionTracker:
    """Runs face-mesh and hand inference on a frame.

    Args:
        max_faces: Maximum faces to track.
        max_hands: Maximum hands to track.
        refine_landmarks: Request iris refinement (478 landmarks).
        face_confidence: Face detection/tracking threshold.
        hand_confidence: Hand detection/tracking threshold.
        model_complexity: Hand model accuracy dial (``0`` or ``1``).
        gesture_smoothing: Frames in the gesture majority vote.
        infer_scale: Inference input scale in ``(0, 1]``.
        hand_idle_stride: While no hand is tracked, run hand inference on one
            of every N frames. ``1`` runs it every frame.
        faces_enabled: Start with face tracking on.
        hands_enabled: Start with hand tracking on.
    """

    def __init__(
        self,
        max_faces: int = 1,
        max_hands: int = 2,
        refine_landmarks: bool = True,
        face_confidence: float = 0.5,
        hand_confidence: float = 0.6,
        model_complexity: int = 1,
        gesture_smoothing: int = 5,
        infer_scale: float = 1.0,
        hand_idle_stride: int = 2,
        faces_enabled: bool = True,
        hands_enabled: bool = True,
    ) -> None:
        self.infer_scale = float(min(1.0, max(0.25, infer_scale)))
        self.hand_idle_stride = max(1, int(hand_idle_stride))
        self.faces_enabled = bool(faces_enabled)
        self.hands_enabled = bool(hands_enabled)

        self.face_detector = FaceMeshDetector(
            max_faces=max_faces,
            refine_landmarks=refine_landmarks,
            detection_confidence=face_confidence,
            tracking_confidence=face_confidence,
        )
        self.hand_tracker = HandTracker(
            max_hands=max_hands,
            detection_confidence=hand_confidence,
            tracking_confidence=hand_confidence,
            model_complexity=model_complexity,
            smoothing_window=gesture_smoothing,
        )

        self._frame_index = 0
        self._tracked_hands: List[HandResult] = []
        self._closed = False
        self.last = TrackingFrame()

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def process(self, frame_bgr: Frame) -> TrackingFrame:
        """Detect faces and hands in one BGR frame.

        Args:
            frame_bgr: Frame straight from the camera.

        Returns:
            A :class:`TrackingFrame` with landmarks in *display* pixels.

        Raises:
            RuntimeError: The tracker has been closed.
        """
        if self._closed:
            raise RuntimeError("VisionTracker has already been closed")

        height, width = frame_bgr.shape[:2]
        display_size = (width, height)
        self._frame_index += 1
        started = time.perf_counter()

        infer_frame = self._downscale(frame_bgr)
        faces = (
            self.face_detector.process(infer_frame, output_size=display_size)
            if self.faces_enabled
            else []
        )
        hands, sampled = self._infer_hands(infer_frame, display_size)

        self.last = TrackingFrame(
            faces=faces,
            hands=hands,
            frame_size=display_size,
            inference_ms=(time.perf_counter() - started) * 1000.0,
            hands_sampled=sampled,
        )
        return self.last

    def _downscale(self, frame: Frame) -> Frame:
        """Return the frame the detectors should see."""
        if self.infer_scale >= 0.999:
            return frame
        height, width = frame.shape[:2]
        return cv2.resize(
            frame,
            (
                max(64, int(width * self.infer_scale)),
                max(64, int(height * self.infer_scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

    def _infer_hands(
        self, infer_frame: Frame, display_size: Tuple[int, int]
    ) -> Tuple[List[HandResult], bool]:
        """Run hand inference, skipping idle frames per the stride.

        Returns:
            ``(hands, sampled)`` -- ``sampled`` is ``False`` on a skipped
            frame. Only an *empty* result is ever reused, so the overlay
            never shows stale landmarks.
        """
        if not self.hands_enabled:
            self._tracked_hands = []
            return [], True

        idle = not self._tracked_hands
        stride = self.hand_idle_stride if idle else 1
        if stride > 1 and self._frame_index % stride:
            return [], False

        self._tracked_hands = self.hand_tracker.process(
            infer_frame, output_size=display_size
        )
        return self._tracked_hands, True

    # ------------------------------------------------------------------ #
    # Toggles / introspection
    # ------------------------------------------------------------------ #
    def toggle_faces(self) -> bool:
        """Flip face tracking. Returns the new state."""
        self.faces_enabled = not self.faces_enabled
        return self.faces_enabled

    def toggle_hands(self) -> bool:
        """Flip hand tracking. Returns the new state."""
        self.hands_enabled = not self.hands_enabled
        if not self.hands_enabled:
            self._tracked_hands = []
        return self.hands_enabled

    @property
    def backend(self) -> str:
        """Description of the active MediaPipe backend."""
        return backend_description()

    def close(self) -> None:
        """Release both MediaPipe graphs."""
        if not self._closed:
            self.face_detector.close()
            self.hand_tracker.close()
            self._closed = True

    def __enter__(self) -> "VisionTracker":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<VisionTracker faces={self.faces_enabled} "
            f"hands={self.hands_enabled} scale={self.infer_scale}>"
        )
