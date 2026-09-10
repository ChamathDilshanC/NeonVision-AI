"""
NeonVision AI - Gesture Action Engine
=====================================

Translates tracked landmarks into operating-system actions: cursor movement,
clicks and drags, media keys, slide navigation and system volume.

Three ideas keep this usable rather than chaotic:

* **Modes.** Air mouse, media control and drawing all want the same hand, so
  only one interaction mode is live at a time (see :class:`ActionMode`). The
  app starts in ``IDLE``, where nothing is sent to the OS at all.
* **Edge triggering with cooldowns.** A gesture held for a second spans ~40
  frames. Discrete actions fire only on the *transition* into a pose, and
  then refuse to fire again until their cooldown expires, so one deliberate
  fist is one mute -- not forty.
* **Normalised geometry.** Every threshold is expressed as a fraction of palm
  size or of frame width, so behaviour does not change with camera
  resolution or how far away the user sits.

Set ``dry_run=True`` to record intent without touching the OS. That is what
the test harness uses, and what ``--dry-run`` exposes on the command line.

Author: ChamathDilshanC
Project: NeonVision AI
License: MIT
"""

from __future__ import annotations

import platform
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np

from core.hand_tracker import HandLandmarkIndex, HandResult

# --------------------------------------------------------------------- #
# Optional OS integration
# --------------------------------------------------------------------- #
_IS_WINDOWS: Final[bool] = platform.system() == "Windows"

try:
    import pyautogui

    # The default 0.1 s pause after every call would cap the loop at 10 FPS.
    pyautogui.PAUSE = 0.0
    # FAILSAFE stays enabled on purpose: dragging the pointer into a screen
    # corner aborts automation, which is the standard emergency stop. The
    # resulting exception is caught below and disarms the engine.
    POINTER_AVAILABLE: Final[bool] = True
    SCREEN_SIZE: Final[Tuple[int, int]] = tuple(pyautogui.size())  # type: ignore
except Exception:  # noqa: BLE001 - any import/display failure degrades safely
    pyautogui = None  # type: ignore[assignment]
    POINTER_AVAILABLE = False
    SCREEN_SIZE = (1920, 1080)

try:
    import winsound

    _BEEP_AVAILABLE: Final[bool] = _IS_WINDOWS
except ImportError:  # pragma: no cover - non-Windows
    winsound = None  # type: ignore[assignment]
    _BEEP_AVAILABLE = False


class ActionMode:
    """Interaction modes. Only one is live at a time."""

    IDLE: Final[str] = "idle"
    MOUSE: Final[str] = "mouse"
    MEDIA: Final[str] = "media"
    DRAW: Final[str] = "draw"

    ORDER: Final[Tuple[str, ...]] = (IDLE, MOUSE, MEDIA, DRAW)

    LABELS: Final[Dict[str, str]] = {
        IDLE: "IDLE (tracking only)",
        MOUSE: "AIR MOUSE",
        MEDIA: "MEDIA / SLIDES",
        DRAW: "AIR DRAWING",
    }

    @classmethod
    def next_mode(cls, current: str) -> str:
        """Return the mode following ``current``, wrapping around."""
        try:
            index = cls.ORDER.index(current)
        except ValueError:
            return cls.IDLE
        return cls.ORDER[(index + 1) % len(cls.ORDER)]


@dataclass(slots=True)
class ActionEvent:
    """A single action the engine performed (or would have, when dry-run)."""

    name: str
    detail: str = ""
    performed: bool = True

    def __str__(self) -> str:
        mark = "" if self.performed else " (dry-run)"
        return f"{self.name}{': ' + self.detail if self.detail else ''}{mark}"


@dataclass(slots=True)
class DrowsinessState:
    """Current eyelid state, as reported by :class:`DrowsinessMonitor`."""

    ear: float = 0.0
    eyes_closed: bool = False
    closed_for: float = 0.0
    alarm: bool = False


class Buzzer:
    """Non-blocking audible alert.

    ``winsound.Beep`` blocks for the full duration of the tone, which would
    stall the render loop, so tones are played on a short-lived daemon thread
    and suppressed while one is already sounding.
    """

    def __init__(self, cooldown: float = 1.2, enabled: bool = True) -> None:
        self.cooldown = float(cooldown)
        #: Whether a real tone generator exists, as opposed to the bell.
        self.available = _BEEP_AVAILABLE
        #: User-facing switch; ``--no-alarm`` clears it for a silent alarm.
        self.enabled = bool(enabled)
        self._last: float = 0.0
        self._active = threading.Event()

    def beep(self, frequency: int = 900, duration_ms: int = 320) -> bool:
        """Sound a tone if enabled and the cooldown has elapsed.

        Args:
            frequency: Tone frequency in Hz (Windows only).
            duration_ms: Tone length in milliseconds.

        Returns:
            ``True`` if a tone was started.
        """
        if not self.enabled:
            return False
        now = time.perf_counter()
        if now - self._last < self.cooldown or self._active.is_set():
            return False
        self._last = now
        self._active.set()
        threading.Thread(
            target=self._play,
            args=(int(frequency), int(duration_ms)),
            name="NeonVisionBuzzer",
            daemon=True,
        ).start()
        return True

    def _play(self, frequency: int, duration_ms: int) -> None:
        try:
            if winsound is not None:
                winsound.Beep(frequency, duration_ms)
            else:
                # Portable fallback: the terminal bell.
                sys.stdout.write("\a")
                sys.stdout.flush()
        except Exception:  # noqa: BLE001 - an alert must never crash the loop
            pass
        finally:
            self._active.clear()


class VolumeController:
    """System volume, with an absolute backend and a portable fallback.

    Backends, in preference order:

    * ``system`` -- the Windows core-audio endpoint via ``pycaw``, which can
      *set* an absolute level and read the real one back.
    * ``keys`` -- synthesised ``volumeup`` / ``volumedown`` presses. Portable,
      but relative only: the true level cannot be read, so an internal
      estimate is nudged towards the target a few steps per update.
    * ``none`` -- neither is available; calls are no-ops.

    Args:
        dry_run: Compute everything but never change the real volume.
    """

    #: Approximate level change per media-key press on Windows.
    KEY_STEP: Final[float] = 0.02
    #: Maximum key presses issued in one update, to stay responsive.
    MAX_KEY_PRESSES: Final[int] = 4
    #: Ignore target changes smaller than this to avoid pointless churn.
    MIN_DELTA: Final[float] = 0.02

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = bool(dry_run)
        self.backend = "none"
        self._endpoint = None
        self._estimate: float = 0.5
        self._last_applied: float = -1.0

        if _IS_WINDOWS:
            self._endpoint = self._open_endpoint()
            if self._endpoint is not None:
                self.backend = "system"
        if self.backend == "none" and POINTER_AVAILABLE:
            self.backend = "keys"

        if self.backend == "system":
            level = self.level
            if level is not None:
                self._estimate = level

    @staticmethod
    def _open_endpoint():
        """Return a pycaw endpoint-volume interface, or ``None``."""
        try:
            from pycaw.utils import AudioUtilities

            speakers = AudioUtilities.GetSpeakers()
            endpoint = getattr(speakers, "EndpointVolume", None)
            if endpoint is not None:
                return endpoint
            # Older pycaw releases expose the raw device only.
            from ctypes import POINTER, cast

            from comtypes import CLSCTX_ALL
            from pycaw.api.endpointvolume import IAudioEndpointVolume

            device = getattr(speakers, "_dev", speakers)
            interface = device.Activate(
                IAudioEndpointVolume._iid_, CLSCTX_ALL, None
            )
            return cast(interface, POINTER(IAudioEndpointVolume))
        except Exception:  # noqa: BLE001 - absence is expected off Windows
            return None

    @property
    def level(self) -> Optional[float]:
        """Current volume in ``[0, 1]``, or ``None`` if unreadable."""
        if self.backend == "system" and self._endpoint is not None:
            try:
                return float(self._endpoint.GetMasterVolumeLevelScalar())
            except Exception:  # noqa: BLE001
                return None
        return self._estimate if self.backend == "keys" else None

    @property
    def display_level(self) -> float:
        """Best-known level in ``[0, 1]``, for the HUD."""
        level = self.level
        return self._estimate if level is None else level

    def set_level(self, fraction: float) -> Optional[float]:
        """Drive the system volume towards ``fraction``.

        Args:
            fraction: Desired level in ``[0, 1]``.

        Returns:
            The level actually targeted, or ``None`` if nothing was done.
        """
        target = float(min(1.0, max(0.0, fraction)))
        if abs(target - self._last_applied) < self.MIN_DELTA:
            return None
        self._last_applied = target
        self._estimate = target

        if self.dry_run or self.backend == "none":
            return target
        if self.backend == "system" and self._endpoint is not None:
            try:
                self._endpoint.SetMasterVolumeLevelScalar(target, None)
                return target
            except Exception:  # noqa: BLE001
                return None
        return self._nudge_with_keys(target)

    def _nudge_with_keys(self, target: float) -> Optional[float]:
        """Approach ``target`` with a few media-key presses."""
        if pyautogui is None:
            return None
        current = self._estimate
        steps = int(round((target - current) / self.KEY_STEP))
        steps = max(-self.MAX_KEY_PRESSES, min(self.MAX_KEY_PRESSES, steps))
        if steps == 0:
            return None
        key = "volumeup" if steps > 0 else "volumedown"
        try:
            for _ in range(abs(steps)):
                pyautogui.press(key)
        except Exception:  # noqa: BLE001
            return None
        return target

    def toggle_mute(self) -> bool:
        """Toggle mute. Returns ``True`` when the request was issued."""
        if self.dry_run:
            return True
        if self.backend == "system" and self._endpoint is not None:
            try:
                self._endpoint.SetMute(not bool(self._endpoint.GetMute()), None)
                return True
            except Exception:  # noqa: BLE001
                return False
        if pyautogui is not None:
            try:
                pyautogui.press("volumemute")
                return True
            except Exception:  # noqa: BLE001
                return False
        return False

    @property
    def muted(self) -> Optional[bool]:
        """Mute state, when the backend can report it."""
        if self.backend == "system" and self._endpoint is not None:
            try:
                return bool(self._endpoint.GetMute())
            except Exception:  # noqa: BLE001
                return None
        return None


class DrowsinessMonitor:
    """Eye-closure watchdog driven by the Eye Aspect Ratio.

    EAR is the ratio of an eye's vertical lid separation to its horizontal
    width, so it collapses towards zero as the eyelid closes and is
    invariant to head distance and image scale.

    A blink is 100-300 ms, so the alarm requires the ratio to stay below
    threshold for :attr:`closed_seconds` before firing -- long enough to
    ignore ordinary blinking.

    Args:
        ear_threshold: Ratio below which the eyes count as closed.
        closed_seconds: Continuous closure required to raise the alarm.
        cooldown: Minimum gap between alarms.
    """

    def __init__(
        self,
        ear_threshold: float = 0.18,
        closed_seconds: float = 1.2,
        cooldown: float = 2.5,
    ) -> None:
        self.ear_threshold = float(ear_threshold)
        self.closed_seconds = float(closed_seconds)
        self.cooldown = float(cooldown)
        self.buzzer = Buzzer(cooldown=cooldown)
        self._closed_since: Optional[float] = None
        self._last_alarm: float = 0.0
        self.state = DrowsinessState()

    def update(
        self, ear: Optional[float], now: Optional[float] = None
    ) -> DrowsinessState:
        """Fold one frame's EAR into the watchdog.

        Args:
            ear: Eye aspect ratio, or ``None`` when no face is tracked.
            now: Timestamp override, for deterministic tests.

        Returns:
            The updated :class:`DrowsinessState`.
        """
        moment = time.perf_counter() if now is None else now
        if ear is None or ear <= 0.0:
            self._closed_since = None
            self.state = DrowsinessState()
            return self.state

        closed = ear < self.ear_threshold
        if not closed:
            self._closed_since = None
            self.state = DrowsinessState(ear=ear)
            return self.state

        if self._closed_since is None:
            self._closed_since = moment
        elapsed = moment - self._closed_since
        alarm = elapsed >= self.closed_seconds
        if alarm and (moment - self._last_alarm) >= self.cooldown:
            self._last_alarm = moment
            self.buzzer.beep()
        self.state = DrowsinessState(
            ear=ear, eyes_closed=True, closed_for=elapsed, alarm=alarm
        )
        return self.state

    def reset(self) -> None:
        """Forget any in-progress closure."""
        self._closed_since = None
        self.state = DrowsinessState()


class GestureActionEngine:
    """Maps hand gestures to OS actions, one mode at a time.

    Args:
        mode: Initial :class:`ActionMode`.
        dry_run: Record intent without touching the OS.
        armed: Master switch; when ``False`` nothing is dispatched.
        screen_size: Cursor target size; defaults to the real screen.
        active_margin: Fraction of the frame trimmed from each edge to form
            the cursor's active region, so the screen edges stay reachable
            without moving the hand out of view.
        smoothing: Exponential smoothing factor for cursor motion, in
            ``[0, 1)``. Higher is steadier but laggier.
        cooldown: Default gap between repeats of a discrete action.
    """

    #: Pinch ratios (fingertip gap / palm size) with hysteresis, so a hand
    #: hovering at the boundary cannot chatter between click and release.
    PINCH_CLOSE: Final[float] = 0.32
    PINCH_OPEN: Final[float] = 0.46
    #: Pinch span mapped onto the 0-100% volume range.
    VOLUME_MIN_RATIO: Final[float] = 0.22
    VOLUME_MAX_RATIO: Final[float] = 1.05
    #: Swipe must cover this fraction of frame width inside the time window.
    SWIPE_DISTANCE: Final[float] = 0.17
    SWIPE_WINDOW: Final[float] = 0.45
    #: Poses that gate each behaviour.
    VOLUME_POSE: Final[str] = "Gun"
    SWIPE_POSE: Final[str] = "Open Palm"

    def __init__(
        self,
        mode: str = ActionMode.IDLE,
        dry_run: bool = False,
        armed: bool = True,
        screen_size: Optional[Tuple[int, int]] = None,
        active_margin: float = 0.18,
        smoothing: float = 0.4,
        cooldown: float = 0.7,
    ) -> None:
        self.mode = mode if mode in ActionMode.ORDER else ActionMode.IDLE
        self.dry_run = bool(dry_run)
        self.armed = bool(armed)
        self.screen_width, self.screen_height = screen_size or SCREEN_SIZE
        self.active_margin = float(min(0.4, max(0.0, active_margin)))
        self.smoothing = float(min(0.95, max(0.0, smoothing)))
        self.cooldown = float(cooldown)

        self.volume = VolumeController(dry_run=dry_run)
        self.pointer_available = POINTER_AVAILABLE

        self._cursor: Optional[Tuple[float, float]] = None
        self._pinching = False
        self._dragging = False
        self._last_fired: Dict[str, float] = {}
        self._previous_pose: Dict[str, str] = {}
        self._swipe_history: Dict[str, Deque[Tuple[float, float]]] = {}
        self._events: List[ActionEvent] = []
        self._log: Deque[str] = deque(maxlen=6)
        self.pinch_ratio: float = 0.0
        self.failsafe_tripped = False

    # ------------------------------------------------------------------ #
    # Frame update
    # ------------------------------------------------------------------ #
    def update(
        self,
        hands: Sequence[HandResult],
        frame_size: Tuple[int, int],
        now: Optional[float] = None,
    ) -> List[ActionEvent]:
        """Process one frame's hands and dispatch any resulting actions.

        Args:
            hands: Hands detected this frame; the first is treated as primary.
            frame_size: ``(width, height)`` of the frame the hands came from.
            now: Timestamp override, for deterministic tests.

        Returns:
            The actions performed this frame.
        """
        moment = time.perf_counter() if now is None else now
        self._events = []

        if not hands:
            self._release_drag(moment)
            self._previous_pose.clear()
            self._swipe_history.clear()
            self.pinch_ratio = 0.0
            return self._events

        primary = hands[0]
        self.pinch_ratio = self._pinch_ratio(primary)

        if not self.armed or self.mode == ActionMode.IDLE:
            # Poses are still tracked so a mode switch does not fire a stale
            # transition on its first frame.
            self._remember_poses(hands)
            return self._events

        if self.mode == ActionMode.MOUSE:
            self._update_mouse(primary, moment)
        elif self.mode == ActionMode.MEDIA:
            self._update_media(hands, frame_size, moment)
        else:  # DRAW is handled by the canvas; nothing is sent to the OS.
            self._release_drag(moment)

        self._remember_poses(hands)
        return self._events

    def _remember_poses(self, hands: Sequence[HandResult]) -> None:
        """Store this frame's poses for next frame's edge detection."""
        for slot, hand in enumerate(hands):
            self._previous_pose[f"{hand.label}:{slot}"] = hand.gesture

    # ------------------------------------------------------------------ #
    # Air mouse
    # ------------------------------------------------------------------ #
    def _update_mouse(self, hand: HandResult, now: float) -> None:
        """Drive the cursor from the index fingertip, pinch to click/drag."""
        tip = hand.normalized[HandLandmarkIndex.INDEX_TIP, :2]
        target = self._to_screen(float(tip[0]), float(tip[1]))

        if self._cursor is None:
            self._cursor = target
        else:
            alpha = 1.0 - self.smoothing
            self._cursor = (
                self._cursor[0] + (target[0] - self._cursor[0]) * alpha,
                self._cursor[1] + (target[1] - self._cursor[1]) * alpha,
            )
        # Round rather than truncate, so the far screen edge stays reachable
        # despite the floating-point division in _to_screen().
        x, y = int(round(self._cursor[0])), int(round(self._cursor[1]))
        if self._call(lambda: pyautogui.moveTo(x, y, _pause=False)):
            self._events.append(
                ActionEvent("move", f"{x},{y}", performed=not self.dry_run)
            )

        # Pinch with hysteresis: press on close, release on open.
        if not self._pinching and self.pinch_ratio < self.PINCH_CLOSE:
            self._pinching = True
            self._dragging = True
            if self._call(lambda: pyautogui.mouseDown()):
                self._events.append(
                    ActionEvent("mouse_down", "pinch", performed=not self.dry_run)
                )
                self._note("click")
        elif self._pinching and self.pinch_ratio > self.PINCH_OPEN:
            self._pinching = False
            self._release_drag(now)

    def _release_drag(self, now: float) -> None:
        """Release a held pinch, if any."""
        del now  # kept for signature symmetry with the other handlers
        self._pinching = False
        if not self._dragging:
            return
        self._dragging = False
        if self._call(lambda: pyautogui.mouseUp()):
            self._events.append(
                ActionEvent("mouse_up", "release", performed=not self.dry_run)
            )

    def _to_screen(self, nx: float, ny: float) -> Tuple[float, float]:
        """Map a normalized landmark to screen coordinates.

        The frame's outer ``active_margin`` is trimmed and the remainder
        stretched across the whole screen, so the user can reach every screen
        edge without moving a hand outside the camera's view.
        """
        span = 1.0 - 2.0 * self.active_margin
        if span <= 0.0:
            span = 1.0
        fx = (nx - self.active_margin) / span
        fy = (ny - self.active_margin) / span
        fx = min(1.0, max(0.0, fx))
        fy = min(1.0, max(0.0, fy))
        return fx * (self.screen_width - 1), fy * (self.screen_height - 1)

    # ------------------------------------------------------------------ #
    # Media, slides and volume
    # ------------------------------------------------------------------ #
    def _update_media(
        self,
        hands: Sequence[HandResult],
        frame_size: Tuple[int, int],
        now: float,
    ) -> None:
        """Handle play/pause, mute, volume and slide swipes."""
        del frame_size  # swipes use normalized coordinates
        for slot, hand in enumerate(hands):
            key = f"{hand.label}:{slot}"
            previous = self._previous_pose.get(key, "")
            pose = hand.gesture

            if pose == "Peace" and previous != "Peace":
                self._fire("play_pause", lambda: pyautogui.press("playpause"))
            elif pose == "Fist" and previous != "Fist":
                self._fire_mute()

            if pose == self.VOLUME_POSE:
                self._update_volume(hand)

            self._track_swipe(key, hand, now)

    def _update_volume(self, hand: HandResult) -> None:
        """Map the thumb-index span onto the system volume."""
        span = self.VOLUME_MAX_RATIO - self.VOLUME_MIN_RATIO
        fraction = (self._pinch_ratio(hand) - self.VOLUME_MIN_RATIO) / span
        fraction = min(1.0, max(0.0, fraction))
        applied = self.volume.set_level(fraction)
        if applied is not None:
            self._events.append(
                ActionEvent(
                    "volume",
                    f"{applied * 100:.0f}%",
                    performed=not self.dry_run,
                )
            )
            self._note(f"vol {applied * 100:.0f}%")

    def _fire_mute(self) -> None:
        """Toggle mute, respecting the shared cooldown."""
        if not self._ready("mute"):
            return
        if self.dry_run or self.volume.toggle_mute():
            self._events.append(
                ActionEvent("mute", "toggle", performed=not self.dry_run)
            )
            self._note("mute")

    def _track_swipe(self, key: str, hand: HandResult, now: float) -> None:
        """Detect a horizontal open-palm swipe and send an arrow key."""
        history = self._swipe_history.setdefault(key, deque(maxlen=48))
        if hand.gesture != self.SWIPE_POSE:
            history.clear()
            return

        wrist_x = float(hand.normalized[HandLandmarkIndex.WRIST, 0])
        history.append((now, wrist_x))
        while history and now - history[0][0] > self.SWIPE_WINDOW:
            history.popleft()
        if len(history) < 4:
            return

        travel = history[-1][1] - history[0][1]
        if abs(travel) < self.SWIPE_DISTANCE:
            return
        # The preview is mirrored, so a rightward on-screen sweep is a
        # rightward gesture from the user's point of view.
        direction = "right" if travel > 0 else "left"
        if self._fire(f"slide_{direction}", lambda: pyautogui.press(direction)):
            history.clear()

    # ------------------------------------------------------------------ #
    # Dispatch helpers
    # ------------------------------------------------------------------ #
    def _pinch_ratio(self, hand: HandResult) -> float:
        """Thumb-to-index distance as a fraction of palm size."""
        palm = hand.palm_size
        if palm < 1e-6:
            return 0.0
        gap = float(
            np.linalg.norm(
                hand.points[HandLandmarkIndex.THUMB_TIP]
                - hand.points[HandLandmarkIndex.INDEX_TIP]
            )
        )
        return gap / palm

    def _ready(self, name: str, cooldown: Optional[float] = None) -> bool:
        """Return ``True`` if ``name`` may fire now, and mark it fired."""
        window = self.cooldown if cooldown is None else cooldown
        now = time.perf_counter()
        if now - self._last_fired.get(name, 0.0) < window:
            return False
        self._last_fired[name] = now
        return True

    def _fire(self, name: str, action) -> bool:
        """Run a discrete action once, subject to its cooldown.

        Args:
            name: Cooldown key and event name.
            action: Zero-argument callable performing the OS call.

        Returns:
            ``True`` when the action was dispatched (or dry-run recorded).
        """
        if not self._ready(name):
            return False
        if not self._call(action):
            return False
        self._events.append(ActionEvent(name, performed=not self.dry_run))
        self._note(name)
        return True

    def _call(self, action) -> bool:
        """Invoke an OS call defensively.

        Returns:
            ``True`` if the call ran, or would have under dry-run.
        """
        if self.dry_run:
            return True
        if not self.pointer_available or pyautogui is None:
            return False
        try:
            action()
            return True
        except Exception as error:  # noqa: BLE001 - never kill the loop
            # pyautogui's corner failsafe lands here: honour it by disarming.
            if type(error).__name__ == "FailSafeException":
                self.armed = False
                self.failsafe_tripped = True
                self._note("FAILSAFE - actions disarmed")
            return False

    def _note(self, message: str) -> None:
        """Append a line to the rolling on-screen action log."""
        self._log.appendleft(message)

    # ------------------------------------------------------------------ #
    # State for the HUD
    # ------------------------------------------------------------------ #
    def set_mode(self, mode: str) -> str:
        """Switch mode, releasing anything the previous mode was holding."""
        self._release_drag(time.perf_counter())
        self.mode = mode if mode in ActionMode.ORDER else ActionMode.IDLE
        self._cursor = None
        self._swipe_history.clear()
        return self.mode

    def cycle_mode(self) -> str:
        """Advance to the next mode."""
        return self.set_mode(ActionMode.next_mode(self.mode))

    def toggle_armed(self) -> bool:
        """Flip the master switch, releasing any held button."""
        if self.armed:
            self._release_drag(time.perf_counter())
        self.armed = not self.armed
        self.failsafe_tripped = False
        return self.armed

    @property
    def mode_label(self) -> str:
        """Human-readable mode name."""
        return ActionMode.LABELS.get(self.mode, self.mode.upper())

    @property
    def recent(self) -> List[str]:
        """Most recent action notes, newest first."""
        return list(self._log)

    def status_line(self) -> str:
        """Single-line summary for the HUD."""
        if self.dry_run:
            state = "DRY-RUN"
        elif not self.armed:
            state = "DISARMED"
        elif not self.pointer_available:
            state = "NO POINTER"
        else:
            state = "ARMED"
        return f"Mode {self.mode_label}   [{state}]"

    def describe_backends(self) -> str:
        """Backend summary for the startup banner."""
        pointer = "pyautogui" if self.pointer_available else "unavailable"
        beep = "winsound" if _BEEP_AVAILABLE else "terminal bell"
        return (
            f"pointer={pointer} volume={self.volume.backend} alert={beep}"
        )
