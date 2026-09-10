"""
NeonVision AI - Core Package
============================

Core computer-vision building blocks of the NeonVision AI pipeline:
camera acquisition, face mesh inference and hand tracking / gesture logic.

Author: ChamathDilshanC
Project: NeonVision AI
License: MIT
"""

from core.camera import Camera, CameraError
from core.face_mesh import FaceMeshDetector, FaceMeshResult, FaceTopology
from core.hand_tracker import HandLandmarkIndex, HandResult, HandTracker
from core.mp_runtime import BACKEND, backend_description
from core.tracker import TrackingFrame, VisionTracker

__all__ = [
    "BACKEND",
    "Camera",
    "CameraError",
    "FaceMeshDetector",
    "FaceMeshResult",
    "FaceTopology",
    "HandLandmarkIndex",
    "HandResult",
    "HandTracker",
    "TrackingFrame",
    "VisionTracker",
    "backend_description",
]
