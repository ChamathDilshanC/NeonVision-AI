"""
NeonVision AI - Video Input / Output
====================================

Capture-source resolution, MP4 recording and snapshot writing.

The recorder deliberately sizes itself from the **first frame handed to it**
rather than from the camera's reported resolution. Two reasons: a driver can
negotiate a different size than it advertises, and the composited frame
NeonVision AI displays is wider than the camera frame because of the
holographic side panel. ``cv2.VideoWriter`` silently discards frames whose
dimensions do not match, producing a valid-looking file with missing content,
so a mismatch is treated as an error worth reporting.

Author: ChamathDilshanC
Project: NeonVision AI
License: MIT
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Final, List, Optional, Tuple, Union

import cv2
import numpy as np

Frame = np.ndarray
Source = Union[int, str]

#: Container/codec pairs tried in order when opening a recorder. ``mp4v`` is
#: the most portable MP4 encoder shipped with opencv-python wheels; ``avc1``
#: gives smaller files where the platform provides an H.264 encoder.
_CODECS: Final[Tuple[Tuple[str, str], ...]] = (
    ("mp4v", ".mp4"),
    ("avc1", ".mp4"),
    ("XVID", ".avi"),
)

#: File extensions treated as video files rather than device indices.
VIDEO_SUFFIXES: Final[Tuple[str, ...]] = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".m4v",
    ".webm",
    ".wmv",
    ".mpg",
    ".mpeg",
)


def resolve_source(value: str) -> Source:
    """Interpret a ``--camera`` argument.

    Args:
        value: Raw command-line string: a device index or a file path.

    Returns:
        An ``int`` device index, or the path as a string.

    Raises:
        FileNotFoundError: The value looks like a video file but is missing.
    """
    text = str(value).strip().strip('"')
    if text.lstrip("-").isdigit():
        return int(text)

    path = Path(text).expanduser()
    if not path.exists():
        # Only insist on existence for things that look like media files;
        # anything else (a stream URL, say) is passed straight to OpenCV.
        if path.suffix.lower() in VIDEO_SUFFIXES:
            raise FileNotFoundError(f"Video file not found: {path}")
        return text
    return str(path)


def describe_source(source: Source) -> str:
    """Human-readable label for a capture source."""
    if isinstance(source, int):
        return f"webcam #{source}"
    return f"file {Path(source).name}"


def probe_video(path: str) -> Tuple[int, int, float, int]:
    """Read a video file's properties without decoding it fully.

    Args:
        path: Path to the video file.

    Returns:
        ``(width, height, fps, frame_count)``; zeros where unavailable.
    """
    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            return 0, 0, 0.0, 0
        return (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            float(capture.get(cv2.CAP_PROP_FPS)),
            int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        capture.release()


def timestamp() -> str:
    """Filename-safe local timestamp, to the second."""
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


class VideoRecorder:
    """Writes composited frames to a video file.

    The recorder is created idle; :meth:`start` opens the encoder using the
    dimensions of the frame it is given, and :meth:`write` refuses frames
    that do not match.

    Args:
        output_dir: Directory for recordings; created on demand.
        fps: Frame rate stamped into the container. Pass the measured loop
            rate so playback speed matches reality.
        prefix: Filename prefix.
    """

    def __init__(
        self,
        output_dir: Path,
        fps: float = 30.0,
        prefix: str = "neonvision",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.fps = float(fps)
        self.prefix = str(prefix)

        self._writer: Optional[cv2.VideoWriter] = None
        self._path: Optional[Path] = None
        self._size: Tuple[int, int] = (0, 0)
        self._frames = 0
        self.last_error: str = ""

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self, frame: Frame, fps: Optional[float] = None) -> Optional[Path]:
        """Open an encoder sized to ``frame``.

        Args:
            frame: The first frame to be written; sets the recording size.
            fps: Overrides the frame rate given at construction.

        Returns:
            The output path, or ``None`` if no encoder could be opened.
        """
        if self.is_recording:
            return self._path
        if frame is None or frame.ndim != 3:
            self.last_error = "no frame available to size the recording"
            return None

        height, width = frame.shape[:2]
        rate = float(fps or self.fps)
        if rate < 1.0 or rate > 240.0:
            rate = 30.0
        self.output_dir.mkdir(parents=True, exist_ok=True)

        attempts: List[str] = []
        for fourcc_name, suffix in _CODECS:
            path = self.output_dir / f"{self.prefix}_{timestamp()}{suffix}"
            fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
            writer = cv2.VideoWriter(str(path), fourcc, rate, (width, height))
            if writer.isOpened():
                self._writer = writer
                self._path = path
                self._size = (width, height)
                self._frames = 0
                self.last_error = ""
                return path
            writer.release()
            path.unlink(missing_ok=True)
            attempts.append(fourcc_name)

        self.last_error = f"no usable encoder (tried {', '.join(attempts)})"
        return None

    def write(self, frame: Frame) -> bool:
        """Append one frame.

        Args:
            frame: Frame matching the recording's dimensions.

        Returns:
            ``True`` when the frame was written. A size mismatch stops the
            recording rather than let the encoder drop frames silently.
        """
        if self._writer is None:
            return False
        height, width = frame.shape[:2]
        if (width, height) != self._size:
            self.last_error = (
                f"frame size changed {self._size} -> {(width, height)}"
            )
            self.stop()
            return False
        self._writer.write(frame)
        self._frames += 1
        return True

    def stop(self) -> Optional[Path]:
        """Close the encoder.

        Returns:
            The finished file's path, or ``None`` if nothing was recording.
        """
        if self._writer is None:
            return None
        self._writer.release()
        self._writer = None
        path = self._path
        self._size = (0, 0)
        return path

    def toggle(
        self, frame: Frame, fps: Optional[float] = None
    ) -> Tuple[bool, str]:
        """Start recording if idle, stop it if active.

        Args:
            frame: Current composited frame.
            fps: Measured loop rate, used when starting.

        Returns:
            ``(recording_now, message)`` -- a message suitable for the HUD.
        """
        if self.is_recording:
            frames = self._frames
            path = self.stop()
            name = path.name if path else "recording"
            return False, f"Recording saved: {name} ({frames} frames)"

        path = self.start(frame, fps=fps)
        if path is None:
            return False, f"Recording failed: {self.last_error}"
        width, height = self._size
        return True, f"Recording to {path.name} ({width}x{height})"

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def is_recording(self) -> bool:
        """``True`` while an encoder is open."""
        return self._writer is not None

    @property
    def path(self) -> Optional[Path]:
        """Path of the current or most recent recording."""
        return self._path

    @property
    def frames_written(self) -> int:
        """Frames written to the current recording."""
        return self._frames

    @property
    def size(self) -> Tuple[int, int]:
        """``(width, height)`` of the current recording."""
        return self._size

    def status(self) -> str:
        """HUD label, e.g. ``"REC 00:04 (128f)"``."""
        if not self.is_recording:
            return ""
        seconds = self._frames / self.fps if self.fps else 0.0
        return f"REC {int(seconds // 60):02d}:{int(seconds % 60):02d} ({self._frames}f)"

    def __enter__(self) -> "VideoRecorder":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<VideoRecorder recording={self.is_recording} "
            f"size={self._size} frames={self._frames}>"
        )


def save_snapshot(
    frame: Frame, output_dir: Path, prefix: str = "neonvision"
) -> Optional[Path]:
    """Write a single PNG snapshot.

    Args:
        frame: Frame to save.
        output_dir: Destination directory; created on demand.
        prefix: Filename prefix.

    Returns:
        The written path, or ``None`` on failure.
    """
    if frame is None:
        return None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{prefix}_{timestamp()}.png"
    try:
        if cv2.imwrite(str(path), frame):
            return path
    except cv2.error:  # pragma: no cover - unwritable path or bad frame
        return None
    return None
