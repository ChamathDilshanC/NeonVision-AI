"""
NeonVision AI - Utils Package
=============================

Presentation and I/O helpers: the neon rendering engine, the holographic
preview panel, the real-time FPS counter and video input/output.

Author: ChamathDilshanC
Project: NeonVision AI
License: MIT
"""

from utils.fps_counter import FPSCounter
from utils.renderer import (
    FINGER_PALETTE,
    PALETTE_NAMES,
    HeadPose,
    HoloPreview,
    MeshStyle,
    NeonRenderer,
    NeonTheme,
    RenderStats,
    estimate_head_pose,
    theme_for_finger_count,
)
from utils.video_io import (
    VideoRecorder,
    describe_source,
    probe_video,
    resolve_source,
    save_snapshot,
)

__all__ = [
    "FINGER_PALETTE",
    "FPSCounter",
    "HeadPose",
    "HoloPreview",
    "MeshStyle",
    "NeonRenderer",
    "NeonTheme",
    "PALETTE_NAMES",
    "RenderStats",
    "VideoRecorder",
    "describe_source",
    "estimate_head_pose",
    "probe_video",
    "resolve_source",
    "save_snapshot",
    "theme_for_finger_count",
]
