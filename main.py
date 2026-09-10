"""
NeonVision AI - Application Entry Point
=======================================

Real-time face mesh and hand gesture recognition with a neon/holographic
overlay, gesture-driven OS control, air drawing and a drowsiness watchdog.

The frame pipeline, once per iteration:

    capture -> (optional down-scale) -> face mesh + hand inference
            -> gesture palette -> queue neon geometry -> single glow
            -> HUD -> holographic side panel -> display / record
            -> gesture actions (cursor, media, slides, volume)

Run it with::

    python main.py                            # 720p, all features
    python main.py --camera myclip.mp4         # process a recorded video
    python main.py --mode mouse                # start in air-mouse mode
    python main.py --dry-run                   # gestures logged, never sent
    python main.py --infer-scale 0.6 --no-holo # maximum frame rate

Interactive keys are listed in :data:`KEY_HINTS` and printed at startup.

Author: ChamathDilshanC
Project: NeonVision AI
License: MIT
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Final, List, Optional, Sequence

import cv2
import numpy as np

from controls.air_canvas import AirCanvas
from controls.gesture_actions import (
    ActionMode,
    DrowsinessMonitor,
    GestureActionEngine,
)
from core.camera import Camera, CameraError
from core.tracker import TrackingFrame, VisionTracker
from utils.fps_counter import FPSCounter
from utils.renderer import (
    FINGER_PALETTE,
    HoloPreview,
    MeshStyle,
    NeonRenderer,
    NeonTheme,
    theme_for_finger_count,
)
from utils.video_io import (
    VideoRecorder,
    describe_source,
    probe_video,
    resolve_source,
    save_snapshot,
)

Frame = np.ndarray

WINDOW_NAME: Final[str] = "NeonVision AI"
OUTPUT_DIR: Final[Path] = Path(__file__).resolve().parent / "output"
KEY_HINTS: Final[str] = (
    "[Q] quit  [TAB] mode  [A] arm  [M] mesh  [C] theme  [P] palette  "
    "[O] holo  [X] clear  [S] snap  [R] rec"
)
#: Give up after this many consecutive empty reads from a live device.
MAX_EMPTY_READS: Final[int] = 240


class NeonVisionApp:
    """Owns the capture device, tracker, renderer, controls and the loop.

    Args:
        config: Parsed command-line configuration.
    """

    def __init__(self, config: argparse.Namespace) -> None:
        self.config = config

        self.camera = Camera(
            source=config.camera,
            width=config.width,
            height=config.height,
            fps=config.fps,
            flip=not config.no_flip,
            threaded=not config.no_thread,
        )
        self.tracker = VisionTracker(
            max_faces=config.max_faces,
            max_hands=config.max_hands,
            refine_landmarks=not config.no_refine,
            face_confidence=config.face_confidence,
            hand_confidence=config.hand_confidence,
            model_complexity=config.hand_complexity,
            gesture_smoothing=config.gesture_smoothing,
            infer_scale=config.infer_scale,
            hand_idle_stride=config.hand_idle_stride,
        )

        self.themes = NeonTheme.presets()
        self._theme_index = 0
        self.base_theme = self.themes[0]
        self.renderer = NeonRenderer(
            theme=self.base_theme,
            glow_enabled=not config.no_glow,
            node_stride=config.node_stride,
        )
        self.holo = HoloPreview(
            theme=self.base_theme,
            width_ratio=config.holo_width,
            density=config.holo_density,
            breathe_speed=config.breathe_speed,
            breathe_amount=config.breathe_amount,
            glow_enabled=not config.no_glow,
        )
        self.holo_enabled = not config.no_holo

        self.actions = GestureActionEngine(
            mode=config.mode,
            dry_run=config.dry_run,
            armed=not config.no_actions,
        )
        self.canvas = AirCanvas()
        self.drowsiness = DrowsinessMonitor(
            ear_threshold=config.ear_threshold,
            closed_seconds=config.drowsy_seconds,
        )
        self.drowsiness_enabled = not config.no_drowsiness
        if config.no_alarm:
            self.drowsiness.buzzer.enabled = False

        self.recorder = VideoRecorder(OUTPUT_DIR, fps=float(config.fps))
        self.fps = FPSCounter(window=45)

        # Runtime toggles.
        self.mesh_style: str = config.mesh_style
        self.hud_enabled: bool = True
        self.palette_follow: bool = not config.no_palette
        self.fullscreen: bool = False
        self.running: bool = False

        self._active_palette: int = -1
        self._last_frame: Optional[Frame] = None
        self._last_tick: float = time.perf_counter()
        self._toast: str = ""
        self._toast_until: float = 0.0
        self._empty_reads: int = 0
        self._keymap: Dict[int, Callable[[], None]] = self._build_keymap()

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self) -> int:
        """Open the capture source and drive the loop until the user quits.

        Returns:
            A process exit code: ``0`` on a clean shutdown, ``1`` on error.
        """
        try:
            self.camera.open()
        except CameraError as error:
            print(f"[NeonVision AI] Camera error: {error}", file=sys.stderr)
            return 1

        self._print_banner()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        width, height = self.camera.resolution
        if width and height:
            panel = int(width * self.config.holo_width) if self.holo_enabled else 0
            cv2.resizeWindow(WINDOW_NAME, width + panel, height)

        self.running = True
        self._empty_reads = 0
        try:
            while self.running:
                ok, frame = self.camera.read()
                if not ok or frame is None:
                    if self._handle_missing_frame():
                        break
                    continue

                self._empty_reads = 0
                canvas = self._process_frame(frame)
                self._last_frame = canvas
                cv2.imshow(WINDOW_NAME, canvas)

                if self.recorder.is_recording and not self.recorder.write(canvas):
                    self._notify(f"Recording stopped: {self.recorder.last_error}")

                self.fps.update()
                if self._handle_key(cv2.waitKey(1) & 0xFF):
                    break
        except KeyboardInterrupt:
            print("\n[NeonVision AI] Interrupted by user.")
        finally:
            self.shutdown()
        return 0

    def _handle_missing_frame(self) -> bool:
        """Decide what to do when a read produced no frame.

        Returns:
            ``True`` when the loop should stop -- the stream ended, the device
            died, or the user asked to quit while waiting.
        """
        if self.camera.is_exhausted:
            print("[NeonVision AI] Video stream ended.")
            return True
        self._empty_reads += 1
        if self._empty_reads > MAX_EMPTY_READS:
            print(
                "[NeonVision AI] Camera stopped delivering frames.",
                file=sys.stderr,
            )
            return True
        # Yield so the reader thread can refill, and stay responsive to keys.
        return self._handle_key(cv2.waitKey(1) & 0xFF)

    def _print_banner(self) -> None:
        """Report the resolved configuration at startup."""
        print(f"[NeonVision AI] Capture   -> {self.camera.describe()}")
        print(f"[NeonVision AI] Inference -> {self.tracker.backend}")
        print(f"[NeonVision AI] Controls  -> {self.actions.describe_backends()}")
        print(f"[NeonVision AI] Mode      -> {self.actions.status_line()}")
        if isinstance(self.config.camera, str):
            width, height, fps, frames = probe_video(self.config.camera)
            if frames:
                print(
                    f"[NeonVision AI] Source    -> {width}x{height} "
                    f"@ {fps:.1f} fps, {frames} frames"
                )
        print(f"[NeonVision AI] Keys: {KEY_HINTS}")

    def _process_frame(self, frame: Frame) -> Frame:
        """Run inference, draw the overlay and dispatch gesture actions.

        Args:
            frame: Raw BGR frame from the capture source.

        Returns:
            The composed canvas (holographic panel plus camera feed).
        """
        now = time.perf_counter()
        dt = min(0.2, max(1e-3, now - self._last_tick))
        self._last_tick = now

        self.fps.stage_start("inference")
        tracking = self.tracker.process(frame)
        self.fps.stage_end("inference")

        self._apply_palette(tracking)

        # Interaction runs before rendering so the HUD shows this frame's
        # state rather than lagging one frame behind.
        self.fps.stage_start("controls")
        self.actions.update(tracking.hands, tracking.frame_size, now)
        pen = self._update_canvas(tracking)
        drowsy = (
            self.drowsiness.update(tracking.eye_aspect_ratio, now)
            if self.drowsiness_enabled
            else None
        )
        self.fps.stage_end("controls")

        self.fps.stage_start("render")
        feed = self._draw_overlay(frame, tracking, pen, drowsy, now)
        canvas = (
            self.holo.compose(
                feed,
                tracking.face,
                dt=dt,
                status=self._holo_status(tracking),
            )
            if self.holo_enabled
            else feed
        )
        self.fps.stage_end("render")
        return canvas

    # ------------------------------------------------------------------ #
    # Drawing
    # ------------------------------------------------------------------ #
    def _draw_overlay(
        self,
        frame: Frame,
        tracking: TrackingFrame,
        pen: Optional[tuple],
        drowsy: object,
        now: float,
    ) -> Frame:
        """Composite the neon overlay and HUD onto the camera frame."""
        self.renderer.begin_frame(frame)
        for face in tracking.faces:
            self.renderer.add_face(face, self.mesh_style)
        for hand in tracking.hands:
            self.renderer.add_hand(hand)
        self.renderer.add_paint(self.canvas.polylines())
        feed = self.renderer.render()

        for face in tracking.faces:
            self.renderer.draw_face_bracket(feed, face)
        for hand in tracking.hands:
            self.renderer.draw_hand_badge(feed, hand)
        if pen is not None and self.actions.mode == ActionMode.DRAW:
            self.renderer.draw_pen_cursor(
                feed, pen, self.canvas.is_drawing, self.renderer.theme.mesh
            )

        if self.hud_enabled:
            self.renderer.draw_hud(
                feed,
                fps_label=self.fps.label(),
                gesture=tracking.gesture_label(),
                status_lines=self._status_lines(tracking),
                hints=KEY_HINTS,
            )
            self.renderer.draw_action_log(feed, self.actions.recent)
        if not tracking.has_subject:
            self.renderer.draw_message(feed, "NO SUBJECT DETECTED")

        if drowsy is not None and getattr(drowsy, "alarm", False):
            # Pulse at ~2 Hz so the band reads as an active alarm.
            pulse = 0.5 + 0.5 * float(np.sin(now * 12.0))
            self.renderer.draw_alert_banner(
                feed,
                "DROWSINESS DETECTED",
                f"eyes closed {drowsy.closed_for:.1f}s - EAR {drowsy.ear:.3f}",
                pulse=pulse,
            )
        self._draw_toast(feed)
        return feed

    def _apply_palette(self, tracking: TrackingFrame) -> None:
        """Recolour the overlay from the held-up finger count."""
        if not self.palette_follow:
            if self._active_palette != 0:
                self._active_palette = 0
                self.renderer.set_theme(self.base_theme)
                self.holo.set_theme(self.base_theme)
            return

        count = tracking.finger_count if tracking.hands else 0
        selector = count if count in FINGER_PALETTE else 0
        if selector == self._active_palette:
            return
        self._active_palette = selector
        theme = theme_for_finger_count(self.base_theme, selector)
        self.renderer.set_theme(theme)
        self.holo.set_theme(theme)

    def _update_canvas(self, tracking: TrackingFrame) -> Optional[tuple]:
        """Advance the air-drawing canvas when in drawing mode."""
        if self.actions.mode != ActionMode.DRAW:
            self.canvas.lift()
            return None
        return self.canvas.update(tracking.hand, self.renderer.theme.mesh)

    # ------------------------------------------------------------------ #
    # HUD content
    # ------------------------------------------------------------------ #
    def _status_lines(self, tracking: TrackingFrame) -> List[str]:
        """Assemble the secondary HUD detail lines."""
        stats = self.renderer.stats
        lines = [
            self.actions.status_line(),
            f"Faces {len(tracking.faces)}   Hands {len(tracking.hands)}"
            f"   Edges {stats.edges}",
            f"Mesh {self.mesh_style.upper()}   Theme {self.renderer.theme.name}",
            f"Infer {self.fps.stage_ms('inference'):5.1f} ms"
            f"   Render {self.fps.stage_ms('render'):5.1f} ms",
        ]
        if tracking.hands:
            lines.append(f"Pose {tracking.pose_summary()}")
        if self.actions.mode == ActionMode.MOUSE:
            lines.append(f"Pinch {self.actions.pinch_ratio:.2f}")
        elif self.actions.mode == ActionMode.MEDIA:
            level = self.actions.volume.display_level
            lines.append(
                f"Volume {level * 100:3.0f}%  ({self.actions.volume.backend})"
            )
        elif self.actions.mode == ActionMode.DRAW:
            lines.append(self.canvas.status())
        if self.drowsiness_enabled:
            state = self.drowsiness.state
            label = "CLOSED" if state.eyes_closed else "OPEN"
            lines.append(f"Eyes {label} (EAR {state.ear:.2f})")
        if self.recorder.is_recording:
            lines.append(self.recorder.status())
        return lines

    def _holo_status(self, tracking: TrackingFrame) -> List[str]:
        """Lines shown inside the holographic panel."""
        face = tracking.face
        lines = [f"LANDMARKS {face.landmark_count if face else 0}"]
        lines.append(f"DOTS {self.holo.point_count}  (x{self.holo.density}/edge)")
        if self.holo.breathing_enabled:
            # A simple bar makes the pulse legible at a glance.
            filled = int(round((self.holo.breath + 1.0) * 4))
            lines.append(f"BREATHE [{'|' * filled}{'.' * (8 - filled)}]")
        else:
            lines.append("BREATHE OFF")
        lines.append("RENDER DOT CLOUD")
        if self._active_palette:
            lines.append(f"PALETTE {self._active_palette}F")
        return lines

    def _draw_toast(self, frame: Frame) -> None:
        """Show a short-lived confirmation message, bottom-left."""
        if not self._toast or time.perf_counter() > self._toast_until:
            return
        height = frame.shape[0]
        for thickness, color in ((4, (0, 0, 0)), (1, self.renderer.theme.hud_accent)):
            cv2.putText(
                frame,
                self._toast,
                (24, height - 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                thickness,
                cv2.LINE_AA,
            )

    def _notify(self, message: str, seconds: float = 1.9) -> None:
        """Queue a toast message and echo it to the console."""
        self._toast = message
        self._toast_until = time.perf_counter() + seconds
        print(f"[NeonVision AI] {message}")

    # ------------------------------------------------------------------ #
    # Input handling
    # ------------------------------------------------------------------ #
    def _build_keymap(self) -> Dict[int, Callable[[], None]]:
        """Map key codes to handlers.

        Built once and shared by both letter cases, which keeps the dispatch
        flat instead of a long ``elif`` ladder.

        Returns:
            Key code -> zero-argument handler.
        """
        bindings: Dict[str, Callable[[], None]] = {
            "a": self._toggle_armed,
            "m": self._cycle_mesh_style,
            "v": self._toggle_faces,
            "h": self._toggle_hands,
            "g": self._toggle_glow,
            "c": self._cycle_theme,
            "p": self._toggle_palette,
            "o": self._toggle_holo,
            "l": self._toggle_breathing,
            "k": self._cycle_holo_density,
            "d": self._toggle_drowsiness,
            "x": self._clear_canvas,
            "z": self._undo_stroke,
            "b": self._toggle_hud,
            "s": self._save_snapshot,
            "r": self._toggle_recording,
            "f": self._toggle_fullscreen,
        }
        keymap: Dict[int, Callable[[], None]] = {9: self._cycle_mode}  # Tab
        for letter, handler in bindings.items():
            keymap[ord(letter)] = handler
            keymap[ord(letter.upper())] = handler
        return keymap

    def _handle_key(self, key: int) -> bool:
        """Apply a keypress.

        Args:
            key: Value from ``cv2.waitKey``, masked to 8 bits.

        Returns:
            ``True`` when the application should quit.
        """
        if key in (255, -1):
            return False
        if key in (ord("q"), ord("Q"), 27):
            return True
        handler = self._keymap.get(key)
        if handler is not None:
            handler()
        return False

    # ------------------------------------------------------------------ #
    # Key handlers
    # ------------------------------------------------------------------ #
    def _cycle_mode(self) -> None:
        """Advance the interaction mode."""
        self.canvas.lift()
        self._notify(f"Mode: {self.actions.cycle_mode().upper()}")

    def _toggle_armed(self) -> None:
        """Arm or disarm all OS actions."""
        armed = self.actions.toggle_armed()
        self._notify(f"Actions {'armed' if armed else 'disarmed'}")

    def _cycle_mesh_style(self) -> None:
        """Advance the face mesh style."""
        self.mesh_style = MeshStyle.next_style(self.mesh_style)
        self._notify(f"Mesh style: {self.mesh_style}")

    def _toggle_faces(self) -> None:
        """Enable or disable face tracking."""
        state = self.tracker.toggle_faces()
        self._notify(f"Face mesh: {'on' if state else 'off'}")

    def _toggle_hands(self) -> None:
        """Enable or disable hand tracking."""
        state = self.tracker.toggle_hands()
        self._notify(f"Hand tracking: {'on' if state else 'off'}")

    def _toggle_glow(self) -> None:
        """Enable or disable the glow pass everywhere."""
        state = self.renderer.toggle_glow()
        self.holo.set_glow(state)
        self._notify(f"Neon glow: {'on' if state else 'off'}")

    def _toggle_palette(self) -> None:
        """Enable or disable gesture-driven recolouring."""
        self.palette_follow = not self.palette_follow
        self._active_palette = -1
        self._notify(f"Gesture palette: {'on' if self.palette_follow else 'off'}")

    def _toggle_holo(self) -> None:
        """Show or hide the holographic side panel."""
        self.holo_enabled = not self.holo_enabled
        self._notify(f"Holo panel: {'on' if self.holo_enabled else 'off'}")

    def _toggle_breathing(self) -> None:
        """Start or stop the hologram's breathing animation."""
        state = "on" if self.holo.toggle_breathing() else "off"
        self._notify(f"Holo breathing: {state}")

    def _cycle_holo_density(self) -> None:
        """Advance the hologram's point-cloud density."""
        self.holo.cycle_density()
        self._notify(
            f"Holo density: x{self.holo.density}/edge "
            f"({self.holo.point_count} dots)"
        )

    def _toggle_drowsiness(self) -> None:
        """Enable or disable the eye watchdog."""
        self.drowsiness_enabled = not self.drowsiness_enabled
        self.drowsiness.reset()
        state = "on" if self.drowsiness_enabled else "off"
        self._notify(f"Drowsiness watch: {state}")

    def _clear_canvas(self) -> None:
        """Erase every air-drawing stroke."""
        self._notify(f"Canvas cleared ({self.canvas.clear()} strokes)")

    def _undo_stroke(self) -> None:
        """Remove the most recent air-drawing stroke."""
        self._notify(
            "Stroke removed" if self.canvas.undo() else "Nothing to undo"
        )

    def _toggle_hud(self) -> None:
        """Show or hide the HUD."""
        self.hud_enabled = not self.hud_enabled
        self._notify(f"HUD: {'on' if self.hud_enabled else 'off'}")

    def _cycle_theme(self) -> None:
        """Advance to the next base colour preset."""
        self._theme_index = (self._theme_index + 1) % len(self.themes)
        self.base_theme = self.themes[self._theme_index]
        self._active_palette = -1
        self.renderer.set_theme(self.base_theme)
        self.holo.set_theme(self.base_theme)
        self._notify(f"Theme: {self.base_theme.name}")

    def _toggle_fullscreen(self) -> None:
        """Switch the preview window between fullscreen and normal."""
        self.fullscreen = not self.fullscreen
        cv2.setWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL,
        )

    # ------------------------------------------------------------------ #
    # Capture output
    # ------------------------------------------------------------------ #
    def _save_snapshot(self) -> None:
        """Write the current composed canvas to ``output/``."""
        if self._last_frame is None:
            self._notify("Snapshot skipped: no frame available yet")
            return
        path = save_snapshot(self._last_frame, OUTPUT_DIR)
        self._notify(
            f"Snapshot saved: {path.name}" if path else "Snapshot failed"
        )

    def _toggle_recording(self) -> None:
        """Start or stop recording the composed canvas."""
        if self._last_frame is None and not self.recorder.is_recording:
            self._notify("Recording skipped: no frame available yet")
            return
        rate = self.fps.fps if self.fps.fps > 5 else float(self.config.fps)
        _, message = self.recorder.toggle(self._last_frame, fps=rate)
        self._notify(message)

    # ------------------------------------------------------------------ #
    # Shutdown
    # ------------------------------------------------------------------ #
    def shutdown(self) -> None:
        """Release every resource the application acquired."""
        self.running = False
        path = self.recorder.stop()
        if path is not None:
            print(f"[NeonVision AI] Recording saved: {path}")
        self.camera.release()
        self.tracker.close()
        cv2.destroyAllWindows()
        print(f"[NeonVision AI] Session: {self.fps.summary()}")


def build_parser() -> argparse.ArgumentParser:
    """Define the command-line interface.

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="neonvision",
        description=(
            "NeonVision AI - real-time neon face mesh, hand gesture "
            "recognition and gesture-driven controls."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    capture = parser.add_argument_group("capture")
    capture.add_argument(
        "--camera",
        type=resolve_source,
        default=0,
        help="webcam device index, or a path to a video file",
    )
    capture.add_argument("--width", type=int, default=1280, help="capture width")
    capture.add_argument("--height", type=int, default=720, help="capture height")
    capture.add_argument("--fps", type=int, default=30, help="requested capture FPS")
    capture.add_argument(
        "--no-flip", action="store_true", help="disable the mirrored preview"
    )
    capture.add_argument(
        "--no-thread",
        action="store_true",
        help="read frames synchronously instead of on a worker thread",
    )

    tracking = parser.add_argument_group("tracking")
    tracking.add_argument(
        "--max-faces", type=int, default=1, help="maximum faces to track"
    )
    tracking.add_argument(
        "--max-hands", type=int, default=2, help="maximum hands to track"
    )
    tracking.add_argument(
        "--no-refine",
        action="store_true",
        help="skip iris/lip refinement (468 landmarks instead of 478)",
    )
    tracking.add_argument(
        "--hand-complexity",
        type=int,
        choices=(0, 1),
        default=1,
        help="hand model complexity; 0 is faster, 1 is more accurate",
    )
    tracking.add_argument(
        "--face-confidence",
        type=float,
        default=0.5,
        help="face detection/tracking confidence threshold",
    )
    tracking.add_argument(
        "--hand-confidence",
        type=float,
        default=0.6,
        help="hand detection/tracking confidence threshold",
    )
    tracking.add_argument(
        "--gesture-smoothing",
        type=int,
        default=5,
        help="frames used for the gesture majority vote; 1 disables it",
    )
    tracking.add_argument(
        "--hand-idle-stride",
        type=int,
        default=2,
        help=(
            "while no hand is tracked, run hand inference on 1 of every N "
            "frames; 1 runs it on every frame"
        ),
    )
    tracking.add_argument(
        "--infer-scale",
        type=float,
        default=1.0,
        help="run inference on a down-scaled copy of the frame (0.25-1.0)",
    )

    controls = parser.add_argument_group("controls")
    controls.add_argument(
        "--mode",
        choices=ActionMode.ORDER,
        default=ActionMode.IDLE,
        help="initial interaction mode",
    )
    controls.add_argument(
        "--dry-run",
        action="store_true",
        help="recognise and log gestures without sending anything to the OS",
    )
    controls.add_argument(
        "--no-actions",
        action="store_true",
        help="start with OS actions disarmed (press A to arm)",
    )
    controls.add_argument(
        "--ear-threshold",
        type=float,
        default=0.18,
        help="eye aspect ratio below which the eyes count as closed",
    )
    controls.add_argument(
        "--drowsy-seconds",
        type=float,
        default=1.2,
        help="continuous eye closure before the drowsiness alarm fires",
    )
    controls.add_argument(
        "--no-drowsiness", action="store_true", help="disable the eye watchdog"
    )
    controls.add_argument(
        "--no-alarm", action="store_true", help="keep the alarm silent (visual only)"
    )

    look = parser.add_argument_group("appearance")
    look.add_argument(
        "--mesh-style",
        choices=MeshStyle.ORDER,
        default=MeshStyle.HYBRID,
        help="initial face mesh style",
    )
    look.add_argument(
        "--node-stride",
        type=int,
        default=8,
        help="draw every n-th face landmark as a node; 0 disables nodes",
    )
    look.add_argument(
        "--no-glow", action="store_true", help="disable the blurred neon glow pass"
    )
    look.add_argument(
        "--no-palette",
        action="store_true",
        help="do not recolour the mesh from the finger count",
    )
    look.add_argument(
        "--no-holo", action="store_true", help="hide the holographic side panel"
    )
    look.add_argument(
        "--holo-width",
        type=float,
        default=0.34,
        help="holographic panel width, as a fraction of the frame width",
    )
    look.add_argument(
        "--holo-density",
        type=int,
        default=4,
        help=(
            "points interpolated along each mesh edge for the holographic "
            "cloud; 0 plots the raw landmarks only, 4 gives ~5800 dots"
        ),
    )
    look.add_argument(
        "--breathe-speed",
        type=float,
        default=1.9,
        help="holographic breathing rate, in radians per second",
    )
    look.add_argument(
        "--breathe-amount",
        type=float,
        default=0.07,
        help="peak dot-spacing change while breathing (0 disables motion)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Program entry point.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    try:
        config = build_parser().parse_args(argv)
    except FileNotFoundError as error:
        print(f"[NeonVision AI] {error}", file=sys.stderr)
        return 1

    config.infer_scale = min(1.0, max(0.25, config.infer_scale))
    config.holo_width = min(0.6, max(0.15, config.holo_width))
    print(f"[NeonVision AI] Source    -> {describe_source(config.camera)}")
    app = NeonVisionApp(config)
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
