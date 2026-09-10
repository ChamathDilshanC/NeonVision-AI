"""
NeonVision AI - Face Mesh Tracker
=================================

High-density facial landmark extraction (468 points, or 478 with iris
refinement).

Two design decisions carry the performance of this module:

* MediaPipe's normalized landmarks are converted into a pixel-space NumPy
  array **once** per frame, so the renderer can slice them with vectorised
  fancy indexing instead of looping in Python.
* The mesh topology (tesselation, contours, iris rings) is flattened into
  ``(N, 2)`` int32 index arrays a single time at import and reused for every
  frame. That is what makes drawing 2 500+ mesh edges affordable at 30+ FPS.

The MediaPipe API differences are absorbed by :mod:`core.mp_runtime`, so this
module works unchanged on both the classic ``solutions`` graphs and the newer
``tasks`` landmarkers.

Author: ChamathDilshanC
Project: NeonVision AI
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Final, List, Optional, Tuple

import cv2
import numpy as np

from core.mp_runtime import FaceBackend, face_connection_tables

Frame = np.ndarray

_TABLES: Final[Dict[str, np.ndarray]] = face_connection_tables()


class FaceTopology:
    """Pre-computed landmark connection tables for the 468/478-point mesh.

    Every table is an ``int32`` array of shape ``(N, 2)`` holding landmark
    index pairs, ready to be turned into polyline batches.
    """

    TESSELATION: Final[np.ndarray] = _TABLES["tesselation"]
    CONTOURS: Final[np.ndarray] = _TABLES["contours"]
    FACE_OVAL: Final[np.ndarray] = _TABLES["face_oval"]
    LIPS: Final[np.ndarray] = _TABLES["lips"]
    LEFT_EYE: Final[np.ndarray] = _TABLES["left_eye"]
    RIGHT_EYE: Final[np.ndarray] = _TABLES["right_eye"]
    LEFT_EYEBROW: Final[np.ndarray] = _TABLES["left_eyebrow"]
    RIGHT_EYEBROW: Final[np.ndarray] = _TABLES["right_eyebrow"]
    IRISES: Final[np.ndarray] = _TABLES["irises"]

    #: Iris centre landmarks; present only in the 478-landmark output.
    RIGHT_IRIS_CENTER: Final[int] = 468
    LEFT_IRIS_CENTER: Final[int] = 473

    #: ``(top, bottom, outer, inner)`` lid landmarks, per eye.
    LEFT_EYE_LIDS: Final[Tuple[int, int, int, int]] = (386, 374, 263, 362)
    RIGHT_EYE_LIDS: Final[Tuple[int, int, int, int]] = (159, 145, 133, 33)

    #: Landmark count that indicates iris refinement was applied.
    REFINED_LANDMARK_COUNT: Final[int] = 478


@dataclass(slots=True)
class FaceMeshResult:
    """Landmarks for a single detected face.

    Attributes:
        points: Pixel coordinates, ``int32`` array of shape ``(N, 2)``.
        normalized: Raw MediaPipe output, ``float32`` array of shape
            ``(N, 3)``: x/y in ``[0, 1]`` and z as relative depth.
        frame_size: ``(width, height)`` the pixel coordinates map into.
    """

    points: np.ndarray
    normalized: np.ndarray
    frame_size: Tuple[int, int]

    @property
    def landmark_count(self) -> int:
        """Number of landmarks (468 without iris refinement, 478 with)."""
        return int(self.points.shape[0])

    @property
    def has_irises(self) -> bool:
        """``True`` when iris landmarks are available."""
        return self.landmark_count >= FaceTopology.REFINED_LANDMARK_COUNT

    def segments(self, connections: np.ndarray) -> np.ndarray:
        """Build a polyline batch for the given connection table.

        Args:
            connections: ``(N, 2)`` index array from :class:`FaceTopology`.

        Returns:
            ``int32`` array of shape ``(N, 2, 2)`` -- one two-point segment per
            connection, ready for a single ``cv2.polylines`` call.
        """
        return self.points[connections]

    def scaled_segments(self, connections: np.ndarray, scale: float) -> np.ndarray:
        """Same as :meth:`segments`, but in a down-scaled coordinate space.

        The renderer uses this to stamp the glow pass into a smaller buffer.

        Args:
            connections: ``(N, 2)`` index array from :class:`FaceTopology`.
            scale: Size factor of the target buffer, e.g. ``0.5``.

        Returns:
            ``int32`` array of shape ``(N, 2, 2)``.
        """
        scaled = self.normalized[:, :2] * (
            self.frame_size[0] * scale,
            self.frame_size[1] * scale,
        )
        return scaled.astype(np.int32)[connections]

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Axis-aligned bounding box as ``(x, y, width, height)``."""
        x_min, y_min = self.points.min(axis=0)
        x_max, y_max = self.points.max(axis=0)
        return int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)

    @property
    def center(self) -> Tuple[int, int]:
        """Centroid of all landmarks, in pixels."""
        cx, cy = self.points.mean(axis=0)
        return int(cx), int(cy)

    def point(self, index: int) -> Tuple[int, int]:
        """Return a single landmark as an ``(x, y)`` pixel tuple."""
        x, y = self.points[index]
        return int(x), int(y)

    def eye_aspect_ratio(self) -> float:
        """Mean eye-aspect ratio across both eyes.

        Values below roughly ``0.18`` indicate closed eyelids, which the HUD
        reports as a blink.

        Returns:
            The averaged vertical/horizontal lid ratio, or ``0.0`` when the
            landmarks are unusable.
        """
        ratios: List[float] = []
        for top, bottom, outer, inner in (
            FaceTopology.LEFT_EYE_LIDS,
            FaceTopology.RIGHT_EYE_LIDS,
        ):
            vertical = float(np.linalg.norm(self.points[top] - self.points[bottom]))
            horizontal = float(
                np.linalg.norm(self.points[outer] - self.points[inner])
            )
            if horizontal > 1e-6:
                ratios.append(vertical / horizontal)
        return float(np.mean(ratios)) if ratios else 0.0


class FaceMeshDetector:
    """Face-mesh front-end producing :class:`FaceMeshResult` objects.

    Args:
        max_faces: Maximum number of faces to track per frame.
        refine_landmarks: Request the iris/lip refinement model (478 landmarks
            instead of 468). Honoured by the ``solutions`` backend; the
            ``tasks`` backend always returns the refined 478.
        detection_confidence: Minimum confidence for the face detector.
        tracking_confidence: Minimum confidence for landmark tracking before
            detection is re-run.
        static_image_mode: Treat every frame as unrelated. Leave ``False`` for
            video -- tracking mode is both faster and temporally smoother.
    """

    def __init__(
        self,
        max_faces: int = 1,
        refine_landmarks: bool = True,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
    ) -> None:
        self.max_faces = int(max_faces)
        self.refine_landmarks = bool(refine_landmarks)
        self.detection_confidence = float(detection_confidence)
        self.tracking_confidence = float(tracking_confidence)
        self.static_image_mode = bool(static_image_mode)

        self._backend = FaceBackend(
            max_faces=self.max_faces,
            refine_landmarks=self.refine_landmarks,
            detection_confidence=self.detection_confidence,
            tracking_confidence=self.tracking_confidence,
            static_image_mode=self.static_image_mode,
        )
        self._closed = False
        self._last_results: List[FaceMeshResult] = []

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def process(
        self,
        frame_bgr: Frame,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> List[FaceMeshResult]:
        """Run face-mesh inference on one BGR frame.

        Args:
            frame_bgr: Frame straight from the camera (BGR, ``uint8``).
            output_size: ``(width, height)`` the pixel landmarks should be
                mapped into. Pass the *display* frame size when inference runs
                on a down-scaled copy -- MediaPipe returns normalized
                coordinates, so the landmarks stay valid at any target size.

        Returns:
            One :class:`FaceMeshResult` per detected face; empty when no face
            is visible.

        Raises:
            RuntimeError: The detector has been closed.
        """
        if self._closed:
            raise RuntimeError("FaceMeshDetector has already been closed")

        if output_size is not None:
            width, height = int(output_size[0]), int(output_size[1])
        else:
            height, width = frame_bgr.shape[:2]

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = [
            FaceMeshResult(
                points=_to_pixels(normalized, width, height),
                normalized=normalized,
                frame_size=(width, height),
            )
            for normalized in self._backend.process(rgb)
        ]
        self._last_results = results
        return results

    # ------------------------------------------------------------------ #
    # Introspection / lifecycle
    # ------------------------------------------------------------------ #
    @property
    def last_results(self) -> List[FaceMeshResult]:
        """Results from the most recent :meth:`process` call."""
        return self._last_results

    @property
    def face_count(self) -> int:
        """Number of faces found in the most recent frame."""
        return len(self._last_results)

    @property
    def backend(self) -> str:
        """Identifier of the MediaPipe API in use."""
        return self._backend.backend

    def close(self) -> None:
        """Release the MediaPipe graph."""
        if not self._closed:
            self._backend.close()
            self._closed = True

    def __enter__(self) -> "FaceMeshDetector":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<FaceMeshDetector max_faces={self.max_faces} "
            f"refine={self.refine_landmarks} backend={self.backend}>"
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
