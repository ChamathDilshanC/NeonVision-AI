"""
NeonVision AI - MediaPipe Runtime Abstraction
=============================================

Backend layer that lets the rest of NeonVision AI stay unaware of *which*
MediaPipe API is installed.

MediaPipe ships two generations of Python API and which one is available
depends entirely on the wheel:

* **solutions** - the classic ``mediapipe.solutions.face_mesh`` /
  ``.hands`` graphs. Bundled with the models, zero setup, but the last wheels
  carrying it are built for Python 3.9-3.12.
* **tasks** - the current ``mediapipe.tasks.python.vision`` landmarkers. This
  is the only API present in recent wheels (including every Python 3.13
  build). The models are *not* bundled: each landmarker loads a ``.task``
  bundle, which this module resolves from ``models/`` and fetches once on
  first run if it is missing.

Both backends are normalised to the same tiny contract -- hand back plain
``(N, 3)`` float32 arrays of normalized landmarks -- so ``face_mesh.py`` and
``hand_tracker.py`` contain one code path regardless of the installed wheel.

Author: ChamathDilshanC
Project: NeonVision AI
License: MIT
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Final, List, Sequence, Set, Tuple

import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "MediaPipe is required by NeonVision AI. Install it with:\n"
        "    pip install -r requirements.txt"
    ) from exc

#: Backend identifiers.
BACKEND_SOLUTIONS: Final[str] = "solutions"
BACKEND_TASKS: Final[str] = "tasks"

#: Where ``.task`` bundles are cached. Override with ``NEONVISION_MODEL_DIR``.
MODEL_DIR: Final[Path] = Path(
    os.environ.get("NEONVISION_MODEL_DIR")
    or Path(__file__).resolve().parent.parent / "models"
)

_MODEL_ROOT: Final[str] = "https://storage.googleapis.com/mediapipe-models"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A downloadable MediaPipe Tasks model bundle."""

    key: str
    filename: str
    url: str
    approx_mb: float


MODEL_SPECS: Final[Dict[str, ModelSpec]] = {
    "face_landmarker": ModelSpec(
        key="face_landmarker",
        filename="face_landmarker.task",
        url=(
            f"{_MODEL_ROOT}/face_landmarker/face_landmarker/float16/1/"
            "face_landmarker.task"
        ),
        approx_mb=3.8,
    ),
    "hand_landmarker": ModelSpec(
        key="hand_landmarker",
        filename="hand_landmarker.task",
        url=(
            f"{_MODEL_ROOT}/hand_landmarker/hand_landmarker/float16/1/"
            "hand_landmarker.task"
        ),
        approx_mb=7.5,
    ),
}


def detect_backend() -> str:
    """Report which MediaPipe API this interpreter has.

    Returns:
        :data:`BACKEND_SOLUTIONS` when the classic graphs are importable,
        otherwise :data:`BACKEND_TASKS`.

    Raises:
        ImportError: Neither API is present, i.e. the wheel is unusable.
    """
    if hasattr(mp, "solutions"):
        return BACKEND_SOLUTIONS
    try:
        from mediapipe.tasks.python import vision  # noqa: F401
    except ImportError as exc:  # pragma: no cover - broken installation
        raise ImportError(
            "The installed MediaPipe build exposes neither 'mediapipe."
            "solutions' nor 'mediapipe.tasks'. Reinstall it with:\n"
            "    pip install -r requirements.txt"
        ) from exc
    return BACKEND_TASKS


#: Resolved once at import; every backend class below branches on this.
BACKEND: Final[str] = detect_backend()


def backend_description() -> str:
    """One-line backend summary for the startup banner."""
    api = "solutions graphs" if BACKEND == BACKEND_SOLUTIONS else "tasks landmarkers"
    return f"MediaPipe {mp.__version__} ({api})"


# ---------------------------------------------------------------------- #
# Model resolution
# ---------------------------------------------------------------------- #
def ensure_model(key: str, allow_download: bool = True) -> Path:
    """Return a local path to a Tasks model bundle, fetching it if needed.

    Args:
        key: A key of :data:`MODEL_SPECS`.
        allow_download: When ``False``, raise instead of hitting the network.

    Returns:
        Path to the cached ``.task`` file.

    Raises:
        KeyError: Unknown model key.
        FileNotFoundError: The bundle is absent and downloading is disabled.
        RuntimeError: The download failed.
    """
    spec = MODEL_SPECS[key]
    target = MODEL_DIR / spec.filename
    if target.is_file() and target.stat().st_size > 0:
        return target

    if not allow_download:
        raise FileNotFoundError(
            f"Model bundle '{spec.filename}' not found in {MODEL_DIR}. "
            f"Download it from {spec.url} and place it there."
        )
    return _download_model(spec, target)


def _download_model(spec: ModelSpec, target: Path) -> Path:
    """Fetch one model bundle to ``target`` atomically."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"[NeonVision AI] Fetching model '{spec.filename}' "
        f"(~{spec.approx_mb:.1f} MB, one-time) ...",
        flush=True,
    )
    handle, temporary = tempfile.mkstemp(dir=str(MODEL_DIR), suffix=".part")
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        request = urllib.request.Request(
            spec.url, headers={"User-Agent": "NeonVisionAI/1.0"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            with temporary_path.open("wb") as sink:
                shutil.copyfileobj(response, sink, length=1 << 16)
        if temporary_path.stat().st_size == 0:
            raise RuntimeError("downloaded file is empty")
        temporary_path.replace(target)
    except Exception as error:  # noqa: BLE001 - reported with full context
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download '{spec.filename}' from {spec.url}: {error}\n"
            f"Download it manually and place it in {MODEL_DIR}."
        ) from error
    print(f"[NeonVision AI] Model ready: {target}", flush=True)
    return target


# ---------------------------------------------------------------------- #
# Connection tables
# ---------------------------------------------------------------------- #
def _pairs_to_array(pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Normalise, de-duplicate and sort landmark index pairs."""
    unique = sorted({tuple(sorted(pair)) for pair in pairs})
    return np.asarray(unique, dtype=np.int32).reshape(-1, 2)


def face_connection_tables() -> Dict[str, np.ndarray]:
    """Face-mesh topology as ``(N, 2)`` index arrays, keyed by region name.

    Keys: ``tesselation``, ``contours``, ``face_oval``, ``lips``,
    ``left_eye``, ``right_eye``, ``left_eyebrow``, ``right_eyebrow``,
    ``irises``.
    """
    if BACKEND == BACKEND_SOLUTIONS:
        source = mp.solutions.face_mesh
        raw: Dict[str, Sequence[Tuple[int, int]]] = {
            "tesselation": list(source.FACEMESH_TESSELATION),
            "contours": list(source.FACEMESH_CONTOURS),
            "face_oval": list(source.FACEMESH_FACE_OVAL),
            "lips": list(source.FACEMESH_LIPS),
            "left_eye": list(source.FACEMESH_LEFT_EYE),
            "right_eye": list(source.FACEMESH_RIGHT_EYE),
            "left_eyebrow": list(source.FACEMESH_LEFT_EYEBROW),
            "right_eyebrow": list(source.FACEMESH_RIGHT_EYEBROW),
            "irises": list(getattr(source, "FACEMESH_IRISES", ())),
        }
    else:
        from mediapipe.tasks.python.vision import FaceLandmarksConnections as fc

        def as_pairs(connections: Sequence[object]) -> List[Tuple[int, int]]:
            return [(c.start, c.end) for c in connections]  # type: ignore[attr-defined]

        raw = {
            "tesselation": as_pairs(fc.FACE_LANDMARKS_TESSELATION),
            "contours": as_pairs(fc.FACE_LANDMARKS_CONTOURS),
            "face_oval": as_pairs(fc.FACE_LANDMARKS_FACE_OVAL),
            "lips": as_pairs(fc.FACE_LANDMARKS_LIPS),
            "left_eye": as_pairs(fc.FACE_LANDMARKS_LEFT_EYE),
            "right_eye": as_pairs(fc.FACE_LANDMARKS_RIGHT_EYE),
            "left_eyebrow": as_pairs(fc.FACE_LANDMARKS_LEFT_EYEBROW),
            "right_eyebrow": as_pairs(fc.FACE_LANDMARKS_RIGHT_EYEBROW),
            "irises": (
                as_pairs(fc.FACE_LANDMARKS_LEFT_IRIS)
                + as_pairs(fc.FACE_LANDMARKS_RIGHT_IRIS)
            ),
        }
    return {name: _pairs_to_array(pairs) for name, pairs in raw.items()}


def hand_connection_table() -> np.ndarray:
    """Hand skeleton topology as an ``(N, 2)`` index array."""
    if BACKEND == BACKEND_SOLUTIONS:
        return _pairs_to_array(list(mp.solutions.hands.HAND_CONNECTIONS))
    from mediapipe.tasks.python.vision import HandLandmarksConnections as hc

    return _pairs_to_array([(c.start, c.end) for c in hc.HAND_CONNECTIONS])


# ---------------------------------------------------------------------- #
# Inference backends
# ---------------------------------------------------------------------- #
class _VideoClock:
    """Strictly increasing millisecond clock for Tasks video mode.

    The Tasks landmarkers reject a timestamp that is not greater than the
    previous one, which is easy to hit on a fast loop where two frames land in
    the same millisecond.
    """

    def __init__(self) -> None:
        self._last: int = -1

    def next(self) -> int:
        """Return the next timestamp, in milliseconds."""
        now = int(time.perf_counter() * 1000.0)
        self._last = max(now, self._last + 1)
        return self._last


class FaceBackend:
    """Face landmark inference, on whichever MediaPipe API is installed.

    Args:
        max_faces: Maximum faces per frame.
        refine_landmarks: Request iris/lip refinement. Only meaningful on the
            solutions backend; the Tasks bundle always returns 478 landmarks.
        detection_confidence: Detector confidence threshold.
        tracking_confidence: Tracking confidence threshold.
        static_image_mode: Disable temporal tracking.
    """

    def __init__(
        self,
        max_faces: int = 1,
        refine_landmarks: bool = True,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
    ) -> None:
        self.backend = BACKEND
        self.static_image_mode = static_image_mode
        self._clock = _VideoClock()
        self._closed = False

        if self.backend == BACKEND_SOLUTIONS:
            self._impl = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=static_image_mode,
                max_num_faces=max_faces,
                refine_landmarks=refine_landmarks,
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
            )
        else:
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision

            mode = (
                vision.RunningMode.IMAGE
                if static_image_mode
                else vision.RunningMode.VIDEO
            )
            options = vision.FaceLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(
                    model_asset_path=str(ensure_model("face_landmarker"))
                ),
                running_mode=mode,
                num_faces=max_faces,
                min_face_detection_confidence=detection_confidence,
                min_face_presence_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._impl = vision.FaceLandmarker.create_from_options(options)

    def process(self, rgb: np.ndarray) -> List[np.ndarray]:
        """Detect faces in an RGB frame.

        Args:
            rgb: Contiguous ``uint8`` RGB image.

        Returns:
            One ``(N, 3)`` float32 array of normalized landmarks per face.
        """
        if self._closed:
            raise RuntimeError("FaceBackend has already been closed")

        if self.backend == BACKEND_SOLUTIONS:
            raw = self._impl.process(rgb)
            faces = raw.multi_face_landmarks or []
            return [_landmarks_to_array(face.landmark) for face in faces]

        image = _as_mp_image(rgb)
        result = (
            self._impl.detect(image)
            if self.static_image_mode
            else self._impl.detect_for_video(image, self._clock.next())
        )
        return [_landmarks_to_array(face) for face in result.face_landmarks]

    def close(self) -> None:
        """Release the underlying graph."""
        if not self._closed:
            self._impl.close()
            self._closed = True


class HandBackend:
    """Hand landmark inference, on whichever MediaPipe API is installed.

    Args:
        max_hands: Maximum hands per frame.
        detection_confidence: Palm-detector confidence threshold.
        tracking_confidence: Tracking confidence threshold.
        model_complexity: Solutions-only accuracy/speed dial (``0`` or ``1``).
        static_image_mode: Disable temporal tracking.
    """

    def __init__(
        self,
        max_hands: int = 2,
        detection_confidence: float = 0.6,
        tracking_confidence: float = 0.5,
        model_complexity: int = 1,
        static_image_mode: bool = False,
    ) -> None:
        self.backend = BACKEND
        self.static_image_mode = static_image_mode
        self._clock = _VideoClock()
        self._closed = False

        if self.backend == BACKEND_SOLUTIONS:
            self._impl = mp.solutions.hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=max_hands,
                model_complexity=model_complexity,
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
            )
        else:
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision

            mode = (
                vision.RunningMode.IMAGE
                if static_image_mode
                else vision.RunningMode.VIDEO
            )
            options = vision.HandLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(
                    model_asset_path=str(ensure_model("hand_landmarker"))
                ),
                running_mode=mode,
                num_hands=max_hands,
                min_hand_detection_confidence=detection_confidence,
                min_hand_presence_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
            )
            self._impl = vision.HandLandmarker.create_from_options(options)

    def process(self, rgb: np.ndarray) -> List[Tuple[str, float, np.ndarray]]:
        """Detect hands in an RGB frame.

        Args:
            rgb: Contiguous ``uint8`` RGB image.

        Returns:
            One ``(label, score, landmarks)`` triple per hand, where
            ``landmarks`` is a ``(21, 3)`` float32 array of normalized
            coordinates.
        """
        if self._closed:
            raise RuntimeError("HandBackend has already been closed")

        if self.backend == BACKEND_SOLUTIONS:
            raw = self._impl.process(rgb)
            hands = raw.multi_hand_landmarks or []
            handedness = raw.multi_hand_landmarks and (raw.multi_handedness or [])
            output: List[Tuple[str, float, np.ndarray]] = []
            for slot, hand in enumerate(hands):
                label, score = "Unknown", 0.0
                if handedness and slot < len(handedness):
                    classification = handedness[slot].classification[0]
                    label, score = classification.label, float(classification.score)
                output.append((label, score, _landmarks_to_array(hand.landmark)))
            return output

        image = _as_mp_image(rgb)
        result = (
            self._impl.detect(image)
            if self.static_image_mode
            else self._impl.detect_for_video(image, self._clock.next())
        )
        output = []
        for slot, landmarks in enumerate(result.hand_landmarks):
            label, score = "Unknown", 0.0
            if slot < len(result.handedness) and result.handedness[slot]:
                category = result.handedness[slot][0]
                label = category.category_name or "Unknown"
                score = float(category.score)
            output.append((label, score, _landmarks_to_array(landmarks)))
        return output

    def close(self) -> None:
        """Release the underlying graph."""
        if not self._closed:
            self._impl.close()
            self._closed = True


# ---------------------------------------------------------------------- #
# Shared helpers
# ---------------------------------------------------------------------- #
def _landmarks_to_array(landmarks: Sequence[object]) -> np.ndarray:
    """Convert a MediaPipe landmark list into an ``(N, 3)`` float32 array."""
    return np.array(
        [(lm.x, lm.y, lm.z) for lm in landmarks],  # type: ignore[attr-defined]
        dtype=np.float32,
    )


def _as_mp_image(rgb: np.ndarray) -> "mp.Image":
    """Wrap an RGB NumPy array in an ``mp.Image`` without copying it.

    The Tasks API requires a C-contiguous buffer; frames sliced or resized by
    OpenCV usually already are, so the copy is only paid when necessary.
    """
    if not rgb.flags["C_CONTIGUOUS"]:
        rgb = np.ascontiguousarray(rgb)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def warn_once(message: str) -> None:  # pragma: no cover - diagnostics only
    """Print a one-off diagnostic to stderr."""
    if message not in _WARNED:
        _WARNED.add(message)
        print(f"[NeonVision AI] {message}", file=sys.stderr)


_WARNED: Set[str] = set()
