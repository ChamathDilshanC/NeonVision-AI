"""
NeonVision AI - Camera Interface
================================

Thread-safe webcam acquisition layer.

The :class:`Camera` class owns the ``cv2.VideoCapture`` handle and hides every
platform quirk behind a small API. It can run in two modes:

* **Synchronous** - ``read()`` grabs a frame on the calling thread.
* **Threaded** (default) - a daemon worker continuously drains the driver
  buffer so the main loop never blocks on I/O. Stale frames are dropped
  instead of queued, which keeps end-to-end latency flat and is usually worth
  10-20 FPS on Windows capture backends.

Author: ChamathDilshanC
Project: NeonVision AI
License: MIT
"""

from __future__ import annotations

import platform
import threading
import time
from typing import Dict, Final, Optional, Tuple

import cv2
import numpy as np

Frame = np.ndarray

#: Capture backends tried in order, per platform. ``cv2.CAP_ANY`` is always
#: the last resort so the class still works on exotic setups.
_BACKENDS: Final[Dict[str, Tuple[int, ...]]] = {
    "Windows": (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY),
    "Darwin": (cv2.CAP_AVFOUNDATION, cv2.CAP_ANY),
    "Linux": (cv2.CAP_V4L2, cv2.CAP_ANY),
}


class CameraError(RuntimeError):
    """Raised when the capture device cannot be opened or has died."""


class Camera:
    """Webcam wrapper with optional background frame grabbing.

    Args:
        source: Device index (``0`` = default webcam) or a video file path.
        width: Requested capture width in pixels.
        height: Requested capture height in pixels.
        fps: Requested capture frame rate.
        flip: Mirror the frame horizontally for a natural "selfie" view.
        threaded: Read frames on a background daemon thread.
        mjpg: Ask the driver for the MJPG codec. Most USB webcams only deliver
            720p/1080p at 30 FPS in MJPG; the default YUY2 stream often caps
            at 5-10 FPS at those resolutions.
        warmup: Seconds to let auto-exposure settle before the first read.
    """

    #: Consecutive failed reads in the worker thread before the device is
    #: declared dead (at ~5 ms of back-off each, roughly one second).
    DEAD_DEVICE_READS: Final[int] = 200

    def __init__(
        self,
        source: int | str = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        flip: bool = True,
        threaded: bool = True,
        mjpg: bool = True,
        warmup: float = 0.35,
    ) -> None:
        self.source = source
        self.requested_width = int(width)
        self.requested_height = int(height)
        self.requested_fps = int(fps)
        self.flip = bool(flip)
        self.threaded = bool(threaded)
        self.mjpg = bool(mjpg)
        self.warmup = float(warmup)

        self._capture: Optional[cv2.VideoCapture] = None
        self._backend_name: str = "unopened"

        # Threaded-mode state.
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._latest_frame: Optional[Frame] = None
        self._frame_id: int = 0
        self._last_delivered_id: int = -1
        self._exhausted: bool = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def open(self) -> "Camera":
        """Open the device and apply the requested capture settings.

        Returns:
            This instance, so calls can be chained.

        Raises:
            CameraError: No backend could deliver a frame from ``source``.
        """
        if self.is_open:
            return self

        backends = _BACKENDS.get(platform.system(), (cv2.CAP_ANY,))
        if isinstance(self.source, str):
            # Video files: let OpenCV pick the demuxer itself, and always read
            # synchronously. Threaded mode deliberately drops frames to stay
            # on the newest one, which is right for a live camera but would
            # skip most of a file during playback.
            backends = (cv2.CAP_ANY,)
            self.threaded = False

        last_error = "no backend attempted"
        for backend in backends:
            capture = cv2.VideoCapture(self.source, backend)
            if not capture.isOpened():
                capture.release()
                last_error = f"backend {backend} refused to open the source"
                continue

            self._configure(capture)
            ok, _ = capture.read()
            if not ok:
                capture.release()
                last_error = f"backend {backend} opened but returned no frame"
                continue

            self._capture = capture
            self._backend_name = capture.getBackendName()
            break
        else:
            raise CameraError(
                f"Unable to open capture source {self.source!r}: {last_error}. "
                "Check that the camera is connected, enabled in privacy "
                "settings, and not held by another application."
            )

        self._exhausted = False
        if self.warmup > 0:
            time.sleep(self.warmup)
        if self.threaded:
            self._start_thread()
        return self

    def _configure(self, capture: cv2.VideoCapture) -> None:
        """Push resolution, frame-rate and codec hints to the driver."""
        if self.mjpg and not isinstance(self.source, str):
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
        capture.set(cv2.CAP_PROP_FPS, self.requested_fps)
        # A one-frame buffer keeps latency low; ignored by some backends.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _start_thread(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="NeonVisionCameraReader",
            daemon=True,
        )
        self._thread.start()

    def _reader_loop(self) -> None:
        """Continuously replace the cached frame with the newest one."""
        capture = self._capture
        if capture is None:
            return
        misses = 0
        while not self._stop_event.is_set():
            ok, frame = capture.read()
            if not ok:
                # A single failed read is usually a transient driver hiccup,
                # so back off and retry. A long run of them means the device
                # is gone (unplugged, or claimed by another process), which
                # has to be reported rather than retried forever.
                misses += 1
                if misses >= self.DEAD_DEVICE_READS:
                    self._exhausted = True
                    return
                time.sleep(0.005)
                continue
            misses = 0
            with self._lock:
                self._latest_frame = frame
                self._frame_id += 1

    def release(self) -> None:
        """Stop the worker thread and release the device."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        with self._lock:
            self._latest_frame = None
        self._backend_name = "closed"

    # ------------------------------------------------------------------ #
    # Frame access
    # ------------------------------------------------------------------ #
    def read(self) -> Tuple[bool, Optional[Frame]]:
        """Return the newest available frame.

        Returns:
            ``(True, frame)`` on success, or ``(False, None)`` when the stream
            is exhausted or no fresh frame has arrived yet.
        """
        if self._capture is None:
            return False, None

        if self.threaded:
            with self._lock:
                frame = self._latest_frame
                frame_id = self._frame_id
            if frame is None or frame_id == self._last_delivered_id:
                return False, None
            self._last_delivered_id = frame_id
        else:
            ok, frame = self._capture.read()
            if not ok or frame is None:
                # Synchronous mode has no retry loop, so a failed read here is
                # the end of the stream (file) or a dead device.
                self._exhausted = True
                return False, None

        if self.flip:
            frame = cv2.flip(frame, 1)
        return True, frame

    def read_blocking(self, timeout: float = 2.0) -> Tuple[bool, Optional[Frame]]:
        """Like :meth:`read`, but wait up to ``timeout`` seconds for a frame."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            ok, frame = self.read()
            if ok:
                return True, frame
            if self._exhausted:
                break
            time.sleep(0.001)
        return False, None

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def is_open(self) -> bool:
        """``True`` while the underlying device is usable."""
        return self._capture is not None and self._capture.isOpened()

    @property
    def is_exhausted(self) -> bool:
        """``True`` once the stream has ended or the device stopped responding.

        Distinguishes "no frame ready yet" -- normal in threaded mode -- from
        "no frame will ever arrive", so the main loop can exit promptly.
        """
        return self._exhausted

    @property
    def resolution(self) -> Tuple[int, int]:
        """Actual ``(width, height)`` negotiated with the driver."""
        if self._capture is None:
            return 0, 0
        return (
            int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    @property
    def backend(self) -> str:
        """Name of the capture backend actually in use."""
        return self._backend_name

    def describe(self) -> str:
        """Human-readable one-line summary, handy for startup logs."""
        width, height = self.resolution
        mode = "threaded" if self.threaded else "sync"
        return (
            f"source={self.source} {width}x{height} "
            f"backend={self.backend} mode={mode}"
        )

    # ------------------------------------------------------------------ #
    # Context manager
    # ------------------------------------------------------------------ #
    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.release()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Camera {self.describe()}>"
