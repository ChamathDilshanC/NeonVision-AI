"""
NeonVision AI - Neon Rendering Engine
=====================================

All drawing for the holographic overlay lives here.

**How the neon glow is produced.** A thin ``cv2.line`` looks flat, so every
stroke is rendered twice, in two different buffers:

1. *Glow pass* - the geometry is stamped onto a half-resolution scratch layer
   several times with decreasing thickness and increasing brightness. That
   already produces a concentric falloff; one ``cv2.GaussianBlur`` then bleeds
   it into a soft halo, and the layer is upscaled and composited onto the
   frame with ``cv2.addWeighted`` so highlights bloom and clip to white.
2. *Core pass* - the same geometry is drawn crisply on top of the composite,
   giving the bright filament inside the halo.

The pass order matters for performance: geometry from *every* subject (faces
and hands) is queued first, so the expensive blur runs exactly **once per
frame** no matter how many landmarks are on screen. Rendering the glow at half
resolution shrinks that blur's cost to roughly a quarter, which is what keeps
a full 2 500-edge face tesselation inside a 30+ FPS budget.

Author: ChamathDilshanC
Project: NeonVision AI
License: MIT
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Dict, Final, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.face_mesh import FaceMeshResult, FaceTopology
from core.hand_tracker import FINGERTIPS, HAND_CONNECTIONS, HandResult

Frame = np.ndarray
Color = Tuple[int, int, int]

_FONT: Final[int] = cv2.FONT_HERSHEY_SIMPLEX
_FONT_BOLD: Final[int] = cv2.FONT_HERSHEY_DUPLEX

#: Gesture-driven palette. Held-up finger count selects the mesh colour.
#: Values are the project's brand hex codes converted to OpenCV BGR order.
FINGER_PALETTE: Final[Dict[int, Color]] = {
    1: (255, 0, 67),    # #4300FF - electric indigo
    2: (248, 101, 0),   # #0065F8 - azure
    3: (255, 202, 0),   # #00CAFF - cyan
    4: (222, 255, 0),   # #00FFDE - aquamarine
}

#: Human-readable palette names, for the HUD.
PALETTE_NAMES: Final[Dict[int, str]] = {
    1: "#4300FF Indigo",
    2: "#0065F8 Azure",
    3: "#00CAFF Cyan",
    4: "#00FFDE Aqua",
}


def _dim(color: Color, factor: float) -> Color:
    """Scale a BGR colour towards black.

    Args:
        color: Source colour in BGR order.
        factor: Multiplier in ``[0, 1]``.

    Returns:
        The scaled colour, clamped to the valid 0-255 range.
    """
    return (
        int(max(0, min(255, color[0] * factor))),
        int(max(0, min(255, color[1] * factor))),
        int(max(0, min(255, color[2] * factor))),
    )


def _lighten(color: Color, amount: float) -> Color:
    """Blend a BGR colour towards white by ``amount`` in ``[0, 1]``."""
    return (
        int(max(0, min(255, color[0] + (255 - color[0]) * amount))),
        int(max(0, min(255, color[1] + (255 - color[1]) * amount))),
        int(max(0, min(255, color[2] + (255 - color[2]) * amount))),
    )


@dataclass(slots=True)
class NeonTheme:
    """Colour and intensity configuration for the overlay.

    All colours are BGR tuples, matching OpenCV's channel order.
    """

    name: str = "Cyan Holo"
    mesh: Color = (255, 190, 40)
    mesh_core: Color = (255, 255, 210)
    contour: Color = (255, 255, 120)
    iris: Color = (255, 120, 255)
    node: Color = (255, 255, 255)
    hand: Color = (255, 220, 60)
    hand_core: Color = (255, 255, 220)
    fingertip: Color = (255, 90, 255)
    hud: Color = (255, 235, 120)
    hud_accent: Color = (255, 120, 255)
    hud_warning: Color = (80, 200, 255)
    panel: Color = (28, 14, 4)

    #: Additive weight of the blurred glow layer.
    glow_strength: float = 0.85
    #: Blur radius in full-resolution pixels; scaled down for the glow buffer.
    glow_radius: int = 21
    #: Resolution factor of the glow buffer (0.5 = half resolution).
    glow_scale: float = 0.5
    #: Thickness of the widest (dimmest) glow stamp, in glow-buffer pixels.
    glow_spread: int = 5

    @staticmethod
    def presets() -> Tuple["NeonTheme", ...]:
        """Built-in themes, cycled with the ``C`` key at runtime."""
        return (
            NeonTheme(),
            NeonTheme(
                name="Matrix Green",
                mesh=(40, 255, 80),
                mesh_core=(210, 255, 220),
                contour=(120, 255, 160),
                iris=(180, 255, 255),
                hand=(60, 255, 120),
                hand_core=(220, 255, 230),
                fingertip=(160, 255, 255),
                hud=(120, 255, 160),
                hud_accent=(200, 255, 220),
                panel=(4, 24, 10),
            ),
            NeonTheme(
                name="Magenta Pulse",
                mesh=(200, 40, 255),
                mesh_core=(245, 215, 255),
                contour=(230, 120, 255),
                iris=(255, 255, 140),
                hand=(220, 60, 255),
                hand_core=(250, 220, 255),
                fingertip=(255, 240, 140),
                hud=(230, 140, 255),
                hud_accent=(255, 240, 160),
                panel=(24, 4, 20),
            ),
            NeonTheme(
                name="Amber Circuit",
                mesh=(40, 170, 255),
                mesh_core=(200, 240, 255),
                contour=(90, 210, 255),
                iris=(255, 240, 160),
                hand=(60, 190, 255),
                hand_core=(210, 245, 255),
                fingertip=(255, 255, 200),
                hud=(90, 210, 255),
                hud_accent=(255, 245, 190),
                panel=(6, 16, 28),
            ),
        )


def theme_for_finger_count(base: NeonTheme, count: int) -> NeonTheme:
    """Recolour a theme from the gesture palette.

    Holding up one to four fingers selects a brand colour; any other count
    (none, or a full open palm) leaves the theme untouched, so the palette is
    something the user opts into rather than a constant flicker.

    Args:
        base: Theme supplying every colour the palette does not override.
        count: Extended finger count, ``0``-``5``.

    Returns:
        ``base`` itself when the count is outside ``1``-``4``, otherwise a
        recoloured copy. The core and contour colours are derived from the
        palette colour so the glow, filament and outline stay a family.
    """
    color = FINGER_PALETTE.get(int(count))
    if color is None:
        return base
    return replace(
        base,
        name=PALETTE_NAMES.get(int(count), base.name),
        mesh=color,
        mesh_core=_lighten(color, 0.72),
        contour=_lighten(color, 0.35),
        hand=color,
        hand_core=_lighten(color, 0.75),
        hud=_lighten(color, 0.3),
        hud_accent=_lighten(color, 0.6),
    )


class MeshStyle:
    """Selectable face-mesh drawing styles."""

    TESSELATION: Final[str] = "tesselation"
    CONTOURS: Final[str] = "contours"
    HYBRID: Final[str] = "hybrid"
    OFF: Final[str] = "off"

    ORDER: Final[Tuple[str, ...]] = (HYBRID, TESSELATION, CONTOURS, OFF)

    @classmethod
    def next_style(cls, current: str) -> str:
        """Return the style following ``current`` in :attr:`ORDER`."""
        try:
            index = cls.ORDER.index(current)
        except ValueError:
            return cls.ORDER[0]
        return cls.ORDER[(index + 1) % len(cls.ORDER)]


@dataclass(slots=True)
class _StrokeBatch:
    """One queued polyline batch awaiting the glow and core passes."""

    segments: np.ndarray
    color: Color
    core_color: Color
    core_thickness: int
    antialias: bool


@dataclass(slots=True)
class _NodeBatch:
    """One queued batch of landmark dots, drawn with ``cv2.circle``."""

    points: np.ndarray
    color: Color
    radius: int
    #: Draw a soft outer ring; used for accent nodes such as fingertips.
    ring: bool = True
    #: Optional brighter centre pip, for the holographic dot mesh.
    core_color: Optional[Color] = None


@dataclass(slots=True)
class _ScatterBatch:
    """A dense point cloud, drawn by vectorised NumPy assignment.

    Thousands of dots per frame make a per-point ``cv2.circle`` loop the
    dominant cost, so these are stamped as small discs with fancy indexing
    instead -- one whole-array write per disc offset, regardless of how many
    points there are.
    """

    points: np.ndarray
    color: Color
    core_color: Optional[Color]
    radius: int


def _clip_to_interior(
    points: np.ndarray, width: int, height: int, margin: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Drop points within ``margin`` pixels of the edge.

    Clipping the batch once lets the caller stamp a disc with plain shifted
    writes, instead of bounds-checking every pixel of every dot.

    Args:
        points: ``(N, 2)`` integer pixel positions.
        width: Target width in pixels.
        height: Target height in pixels.
        margin: Safety border, normally the disc radius.

    Returns:
        ``(xs, ys)`` arrays holding only the safely paintable points.
    """
    xs = points[:, 0]
    ys = points[:, 1]
    if margin <= 0:
        inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    else:
        inside = (
            (xs >= margin)
            & (xs < width - margin)
            & (ys >= margin)
            & (ys < height - margin)
        )
    if inside.all():
        return xs, ys
    return xs[inside], ys[inside]


@lru_cache(maxsize=16)
def _disc_offsets(radius: int) -> Tuple[Tuple[int, int], ...]:
    """Integer ``(dx, dy)`` offsets filling a disc of the given radius.

    Cached because the renderer asks for the same handful of radii every
    frame. A disc rather than a square keeps dense dots looking round even
    though they are drawn without antialiasing.
    """
    span = range(-radius, radius + 1)
    limit = radius * radius + radius // 2
    return tuple(
        (dx, dy) for dy in span for dx in span if dx * dx + dy * dy <= limit
    )


@dataclass(slots=True)
class RenderStats:
    """Counters describing what the last frame contained."""

    faces: int = 0
    hands: int = 0
    edges: int = 0
    nodes: int = 0
    extras: Dict[str, str] = field(default_factory=dict)


class NeonRenderer:
    """Two-pass neon renderer for face-mesh and hand overlays.

    Typical use inside the main loop::

        renderer.begin_frame(frame)
        renderer.add_face(face, MeshStyle.HYBRID)
        renderer.add_hand(hand)
        frame = renderer.render()
        renderer.draw_hud(frame, fps_label="FPS 60", gesture="Gesture: 3")

    Args:
        theme: Colour configuration; defaults to the cyan preset.
        glow_enabled: Set ``False`` to skip the blur pass entirely (useful on
            very low-power CPUs; costs the halo, keeps the wireframe).
        node_stride: Draw every *n*-th face landmark as a dot. Higher values
            are faster and sparser; ``0`` disables the dots.
    """

    #: Segment count above which a batch is treated as dense; see
    #: :meth:`_stamp_glow`.
    DENSE_BATCH_SEGMENTS: Final[int] = 600

    def __init__(
        self,
        theme: Optional[NeonTheme] = None,
        glow_enabled: bool = True,
        node_stride: int = 8,
    ) -> None:
        self.theme = theme or NeonTheme()
        self.glow_enabled = bool(glow_enabled)
        self.node_stride = max(0, int(node_stride))

        self._frame: Optional[Frame] = None
        self._frame_size: Tuple[int, int] = (0, 0)
        self._glow: Optional[Frame] = None
        self._glow_upscaled: Optional[Frame] = None
        self._strokes: List[_StrokeBatch] = []
        self._nodes: List[_NodeBatch] = []
        self._scatters: List[_ScatterBatch] = []
        self._stats = RenderStats()
        self._scan_offset: int = 0
        # Bounding box of everything stamped into the glow layer this frame,
        # in glow-buffer coordinates: (x0, y0, x1, y1), or None when empty.
        self._glow_bounds: Optional[Tuple[int, int, int, int]] = None

    # ------------------------------------------------------------------ #
    # Frame lifecycle
    # ------------------------------------------------------------------ #
    def begin_frame(self, frame: Frame) -> None:
        """Bind a frame and clear the queued geometry and glow buffer.

        Args:
            frame: The BGR frame that will be drawn on, in place.
        """
        height, width = frame.shape[:2]
        if (width, height) != self._frame_size:
            self._allocate_buffers(width, height)

        self._frame = frame
        self._strokes.clear()
        self._nodes.clear()
        self._scatters.clear()
        self._stats = RenderStats()
        self._glow_bounds = None
        if self._glow is not None:
            self._glow[:] = 0

    def _allocate_buffers(self, width: int, height: int) -> None:
        """(Re)allocate the glow buffers for a new frame size."""
        self._frame_size = (width, height)
        scale = self.theme.glow_scale
        glow_width = max(1, int(width * scale))
        glow_height = max(1, int(height * scale))
        self._glow = np.zeros((glow_height, glow_width, 3), dtype=np.uint8)
        self._glow_upscaled = np.zeros((height, width, 3), dtype=np.uint8)

    # ------------------------------------------------------------------ #
    # Geometry queueing
    # ------------------------------------------------------------------ #
    def add_face(self, face: FaceMeshResult, style: str = MeshStyle.HYBRID) -> None:
        """Queue one face mesh for rendering.

        Args:
            face: Landmark result from
                :class:`core.face_mesh.FaceMeshDetector`.
            style: One of the :class:`MeshStyle` constants.
        """
        if style == MeshStyle.OFF:
            return

        self._stats.faces += 1

        if style in (MeshStyle.TESSELATION, MeshStyle.HYBRID):
            # 2 500+ edges: non-antialiased lines keep this affordable.
            self._queue_face_strokes(
                face,
                FaceTopology.TESSELATION,
                self.theme.mesh,
                self.theme.mesh_core,
                core_thickness=1,
                antialias=False,
            )

        if style in (MeshStyle.CONTOURS, MeshStyle.HYBRID):
            self._queue_face_strokes(
                face,
                FaceTopology.CONTOURS,
                self.theme.contour,
                self.theme.mesh_core,
                core_thickness=1,
                antialias=True,
            )
            self._queue_face_strokes(
                face,
                FaceTopology.FACE_OVAL,
                self.theme.contour,
                self.theme.mesh_core,
                core_thickness=2,
                antialias=True,
            )

        if face.has_irises:
            self._queue_face_strokes(
                face,
                FaceTopology.IRISES,
                self.theme.iris,
                self.theme.iris,
                core_thickness=1,
                antialias=True,
            )
            centers = face.points[
                [FaceTopology.LEFT_IRIS_CENTER, FaceTopology.RIGHT_IRIS_CENTER]
            ]
            self._queue_nodes(centers, self.theme.iris, radius=2)

        if self.node_stride and style != MeshStyle.CONTOURS:
            self._queue_nodes(
                face.points[:: self.node_stride], self.theme.node, radius=1
            )

    def add_hand(self, hand: HandResult) -> None:
        """Queue one hand skeleton, with fingertips accented.

        Args:
            hand: Result from :class:`core.hand_tracker.HandTracker`.
        """
        self._stats.hands += 1
        self._queue_strokes(
            full_res=hand.points[HAND_CONNECTIONS],
            glow_res=hand.scaled_segments(self.theme.glow_scale),
            color=self.theme.hand,
            core_color=self.theme.hand_core,
            core_thickness=2,
            antialias=True,
        )
        self._queue_nodes(hand.points, self.theme.hand_core, radius=3)
        self._queue_nodes(hand.points[FINGERTIPS], self.theme.fingertip, radius=6)

    def add_segments(
        self,
        segments: np.ndarray,
        color: Optional[Color] = None,
        core_color: Optional[Color] = None,
        core_thickness: int = 2,
        antialias: bool = True,
    ) -> None:
        """Queue arbitrary geometry through the neon pipeline.

        This is the low-level entry point used by the holographic preview and
        the air-drawing canvas: anything shaped like ``cv2.polylines`` input
        gets the same glow-plus-filament treatment as the face mesh.

        Args:
            segments: ``int32`` array of shape ``(batches, points, 2)`` in
                full-resolution frame coordinates.
            color: Halo colour; defaults to the theme's mesh colour.
            core_color: Filament colour; defaults to the theme's core colour.
            core_thickness: Filament thickness in pixels.
            antialias: Smooth the filament. Leave off for very dense batches.
        """
        if segments.size == 0:
            return
        geometry = np.ascontiguousarray(segments, dtype=np.int32)
        glow_geometry = (
            geometry.astype(np.float32) * self.theme.glow_scale
        ).astype(np.int32)
        self._queue_strokes(
            full_res=geometry,
            glow_res=glow_geometry,
            color=color or self.theme.mesh,
            core_color=core_color or self.theme.mesh_core,
            core_thickness=core_thickness,
            antialias=antialias,
        )

    def add_paint(self, strokes: Sequence[Tuple[np.ndarray, Color]]) -> None:
        """Queue air-drawing strokes.

        Args:
            strokes: ``(polyline, colour)`` pairs from
                :meth:`controls.air_canvas.AirCanvas.polylines`.
        """
        for polyline, color in strokes:
            self.add_segments(
                polyline,
                color=color,
                core_color=_lighten(color, 0.7),
                core_thickness=3,
                antialias=True,
            )

    def _queue_face_strokes(
        self,
        face: FaceMeshResult,
        connections: np.ndarray,
        color: Color,
        core_color: Color,
        core_thickness: int,
        antialias: bool,
    ) -> None:
        """Queue a connection table of a face at both resolutions."""
        self._queue_strokes(
            full_res=face.segments(connections),
            glow_res=face.scaled_segments(connections, self.theme.glow_scale),
            color=color,
            core_color=core_color,
            core_thickness=core_thickness,
            antialias=antialias,
        )

    def _queue_strokes(
        self,
        full_res: np.ndarray,
        glow_res: np.ndarray,
        color: Color,
        core_color: Color,
        core_thickness: int,
        antialias: bool,
    ) -> None:
        """Stamp geometry into the glow layer and queue the crisp core pass."""
        self._stats.edges += int(full_res.shape[0])
        if self.glow_enabled and self._glow is not None:
            self._stamp_glow(glow_res, color)
        self._strokes.append(
            _StrokeBatch(
                segments=full_res,
                color=color,
                core_color=core_color,
                core_thickness=core_thickness,
                antialias=antialias,
            )
        )

    def _stamp_glow(self, segments: np.ndarray, color: Color) -> None:
        """Draw one batch several times, widest and dimmest first.

        Drawing thick-and-dim before thin-and-bright creates the radial
        falloff that the later blur turns into a soft halo.

        Dense batches (the face tesselation, at 1 300+ edges) get one stamp
        fewer: their strokes already overlap heavily, so the extra pass costs
        real milliseconds while adding almost nothing the blur does not.
        """
        assert self._glow is not None
        spread = max(1, self.theme.glow_spread)
        if segments.shape[0] > self.DENSE_BATCH_SEGMENTS:
            spread = min(spread, 3)
        for step in range(spread, 0, -2):
            factor = 0.30 + 0.70 * (1.0 - (step - 1) / spread)
            cv2.polylines(
                self._glow,
                segments,
                isClosed=False,
                color=_dim(color, factor),
                thickness=step,
                lineType=cv2.LINE_8,
            )
        self._grow_bounds(segments, margin=spread)

    def _grow_bounds(self, segments: np.ndarray, margin: int) -> None:
        """Extend the glow-layer dirty region to cover ``segments``."""
        if segments.size == 0:
            return
        flat = segments.reshape(-1, 2)
        x0 = int(flat[:, 0].min()) - margin
        y0 = int(flat[:, 1].min()) - margin
        x1 = int(flat[:, 0].max()) + margin
        y1 = int(flat[:, 1].max()) + margin
        if self._glow_bounds is None:
            self._glow_bounds = (x0, y0, x1, y1)
            return
        previous = self._glow_bounds
        self._glow_bounds = (
            min(previous[0], x0),
            min(previous[1], y0),
            max(previous[2], x1),
            max(previous[3], y1),
        )

    def _queue_nodes(
        self,
        points: np.ndarray,
        color: Color,
        radius: int,
        ring: bool = True,
        core_color: Optional[Color] = None,
    ) -> None:
        """Queue a batch of landmark dots for the core pass."""
        self._stats.nodes += int(points.shape[0])
        self._nodes.append(
            _NodeBatch(
                points=points,
                color=color,
                radius=max(1, radius),
                ring=ring,
                core_color=core_color,
            )
        )

    def add_points(
        self,
        points: np.ndarray,
        color: Optional[Color] = None,
        core_color: Optional[Color] = None,
        radius: int = 2,
        glow_radius: int = 1,
        intensity: float = 1.0,
    ) -> None:
        """Queue a batch of glowing dots -- no connecting lines.

        This is the point-cloud counterpart to :meth:`add_segments`, used by
        the holographic dot mesh.

        The halo is produced by *scattering* the dots straight into the
        half-resolution glow buffer with NumPy fancy indexing rather than by
        stamping hundreds of circles: the single Gaussian pass that already
        runs each frame turns those lit pixels into soft blooms. Nearly 500
        landmarks therefore cost one vectorised write instead of ~1 500
        ``cv2.circle`` calls.

        Args:
            points: ``(N, 2)`` array of full-resolution pixel positions.
            color: Halo colour; defaults to the theme's mesh colour.
            core_color: Bright centre colour; defaults to the theme's core.
            radius: Crisp dot radius in pixels.
            glow_radius: Half-width of the scattered glow seed, in
                glow-buffer pixels. ``1`` writes a 3x3 block per dot.
            intensity: Scales the halo brightness in ``[0, 1]``, which is how
                the breathing animation pulses the glow.
        """
        if points.size == 0:
            return
        halo = color or self.theme.mesh
        pip = core_color or self.theme.mesh_core
        pixels = np.ascontiguousarray(points, dtype=np.int32).reshape(-1, 2)
        self._stats.nodes += int(pixels.shape[0])

        if self.glow_enabled and self._glow is not None:
            self._scatter_glow(pixels, _dim(halo, max(0.0, min(1.0, intensity))),
                               glow_radius)
        self._queue_nodes(
            pixels, halo, radius=max(1, radius), ring=False, core_color=pip
        )

    def add_point_cloud(
        self,
        points: np.ndarray,
        color: Optional[Color] = None,
        core_color: Optional[Color] = None,
        radius: int = 2,
        glow_radius: int = 1,
        intensity: float = 1.0,
    ) -> None:
        """Queue a *dense* point cloud -- thousands of dots, no lines.

        Identical in effect to :meth:`add_points`, but both passes are
        vectorised: the halo seed and the crisp dot are stamped with NumPy
        fancy indexing rather than one ``cv2.circle`` per point. At the ~5 000
        points an interpolated Lidar-style cloud needs, that is the difference
        between roughly 25 ms and under 2 ms per frame.

        Args:
            points: ``(N, 2)`` array of full-resolution pixel positions.
            color: Dot colour; defaults to the theme's mesh colour.
            core_color: Brighter centre colour; defaults to the theme's core.
            radius: Dot radius in pixels.
            glow_radius: Half-width of the scattered glow seed, in
                glow-buffer pixels. ``0`` writes a single pixel per dot.
            intensity: Scales halo brightness in ``[0, 1]``.
        """
        if points.size == 0:
            return
        halo = color or self.theme.mesh
        pip = core_color or self.theme.mesh_core
        pixels = np.ascontiguousarray(points, dtype=np.int32).reshape(-1, 2)
        self._stats.nodes += int(pixels.shape[0])

        if self.glow_enabled and self._glow is not None:
            self._scatter_glow(
                pixels,
                _dim(halo, max(0.0, min(1.0, intensity))),
                max(0, glow_radius),
            )
        self._scatters.append(
            _ScatterBatch(
                points=pixels,
                color=halo,
                core_color=pip,
                radius=max(1, radius),
            )
        )

    def _draw_scatter(self, frame: Frame, batch: _ScatterBatch) -> None:
        """Stamp a dense dot batch onto the frame with fancy indexing.

        Each disc offset is a single masked assignment over the whole batch,
        so cost scales with the disc area, not with the point count.
        """
        height, width = frame.shape[:2]
        radius = batch.radius

        # Clip once for the whole batch rather than bounds-checking inside
        # the offset loop: the mask, its three ANDs and two fancy-index
        # copies would otherwise be repeated for every pixel of the disc.
        # Dots nearer the edge than the disc radius are dropped, which is
        # invisible -- they would have been clipped anyway.
        xs, ys = _clip_to_interior(batch.points, width, height, radius)
        if xs.size == 0:
            return

        # Outer disc in the base colour, then a smaller bright centre, which
        # reads as a lit point rather than a flat blob.
        passes = [(radius, batch.color)]
        if batch.core_color is not None and radius >= 2:
            passes.append((radius - 1, batch.core_color))

        for disc_radius, disc_color in passes:
            for dx, dy in _disc_offsets(disc_radius):
                frame[ys + dy, xs + dx] = disc_color

    def _scatter_glow(
        self, pixels: np.ndarray, color: Color, spread: int
    ) -> None:
        """Write dot seeds into the glow buffer with vectorised indexing."""
        assert self._glow is not None
        glow_height, glow_width = self._glow.shape[:2]
        scaled = (pixels.astype(np.float32) * self.theme.glow_scale).astype(np.int32)
        self._grow_bounds(scaled.reshape(-1, 1, 2), margin=spread + 2)

        # Clipped once for the batch, then each offset is a bare shifted
        # write. A dense cloud saturates the half-resolution buffer with
        # single pixels alone, which is why spread 0 is usually enough.
        xs, ys = _clip_to_interior(scaled, glow_width, glow_height, spread)
        if xs.size == 0:
            return
        if spread <= 0:
            self._glow[ys, xs] = color
            return
        for dy in range(-spread, spread + 1):
            for dx in range(-spread, spread + 1):
                self._glow[ys + dy, xs + dx] = color

    # ------------------------------------------------------------------ #
    # Composition
    # ------------------------------------------------------------------ #
    def render(self) -> Frame:
        """Composite the glow, then draw the crisp core geometry.

        Returns:
            The bound frame, modified in place.

        Raises:
            RuntimeError: :meth:`begin_frame` was not called first.
        """
        if self._frame is None:
            raise RuntimeError("begin_frame() must be called before render()")

        frame = self._frame
        queued = self._strokes or self._scatters
        if self.glow_enabled and queued and self._glow is not None:
            self._composite_glow(frame)

        for batch in self._strokes:
            cv2.polylines(
                frame,
                batch.segments,
                isClosed=False,
                color=batch.core_color,
                thickness=batch.core_thickness,
                lineType=cv2.LINE_AA if batch.antialias else cv2.LINE_8,
            )

        for cloud in self._scatters:
            self._draw_scatter(frame, cloud)

        for nodes in self._nodes:
            self._draw_nodes(frame, nodes)

        return frame

    def _composite_glow(self, frame: Frame) -> None:
        """Blur the glow layer once, upscale it and add it to the frame.

        Only the region that geometry was actually stamped into is processed.
        Faces and hands typically occupy a third of the viewport, and blur,
        upscale and blend are all linear in pixel count -- so restricting the
        three passes to that region, rather than the full frame, is the
        single largest saving in the render path.
        """
        assert self._glow is not None and self._glow_upscaled is not None
        kernel = self._glow_kernel()
        region = self._dirty_region(kernel)
        if region is None:
            return
        gx0, gy0, gx1, gy1 = region

        glow_roi = self._glow[gy0:gy1, gx0:gx1]
        cv2.GaussianBlur(glow_roi, (kernel, kernel), 0, dst=glow_roi)

        scale = self.theme.glow_scale
        width, height = self._frame_size
        fx0 = max(0, min(width, int(gx0 / scale)))
        fy0 = max(0, min(height, int(gy0 / scale)))
        fx1 = max(fx0 + 1, min(width, int(round(gx1 / scale))))
        fy1 = max(fy0 + 1, min(height, int(round(gy1 / scale))))

        upscaled_roi = self._glow_upscaled[fy0:fy1, fx0:fx1]
        cv2.resize(
            glow_roi,
            (fx1 - fx0, fy1 - fy0),
            dst=upscaled_roi,
            interpolation=cv2.INTER_LINEAR,
        )
        frame_roi = frame[fy0:fy1, fx0:fx1]
        cv2.addWeighted(
            frame_roi,
            1.0,
            upscaled_roi,
            self.theme.glow_strength,
            0.0,
            dst=frame_roi,
        )

    def _dirty_region(self, kernel: int) -> Optional[Tuple[int, int, int, int]]:
        """Clamp the stamped bounding box, padded for the blur's spill."""
        if self._glow_bounds is None or self._glow is None:
            return None
        glow_height, glow_width = self._glow.shape[:2]
        pad = kernel  # the halo spreads roughly one kernel beyond the stroke
        x0, y0, x1, y1 = self._glow_bounds
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(glow_width, x1 + pad)
        y1 = min(glow_height, y1 + pad)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None
        return x0, y0, x1, y1

    def _glow_kernel(self) -> int:
        """Odd Gaussian kernel size, scaled to the glow buffer resolution."""
        scaled = int(self.theme.glow_radius * self.theme.glow_scale)
        return max(3, scaled | 1)

    def _draw_nodes(self, frame: Frame, batch: _NodeBatch) -> None:
        """Draw one dot batch.

        Accent nodes get a soft outer ring; the holographic dot mesh instead
        gets a brighter centre pip, which reads as a lit point rather than a
        drawn circle.
        """
        color = batch.color
        radius = batch.radius
        height, width = frame.shape[:2]
        pip = batch.core_color
        pip_radius = max(1, radius - 1)
        for x, y in batch.points:
            cx, cy = int(x), int(y)
            # Dots are cheap but not free: skipping off-frame centres avoids
            # the clipping work when a subject is only partly in view.
            if not (-radius <= cx <= width + radius):
                continue
            if not (-radius <= cy <= height + radius):
                continue
            center = (cx, cy)
            if batch.ring and radius > 2:
                cv2.circle(
                    frame, center, radius + 2, _dim(color, 0.35), 1, cv2.LINE_AA
                )
            cv2.circle(frame, center, radius, color, -1, cv2.LINE_AA)
            if pip is not None and radius >= 2:
                cv2.circle(frame, center, pip_radius, pip, -1, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    # HUD
    # ------------------------------------------------------------------ #
    def draw_hud(
        self,
        frame: Frame,
        fps_label: str,
        gesture: str,
        status_lines: Sequence[str] = (),
        hints: str = "",
        show_scanline: bool = True,
    ) -> Frame:
        """Draw the holographic HUD on top of a rendered frame.

        Args:
            frame: Frame returned by :meth:`render`.
            fps_label: Performance string, e.g. ``"FPS 58 | 17.2 ms"``.
            gesture: Primary gesture readout, e.g. ``"Gesture: 3 Fingers"``.
            status_lines: Extra detail lines shown under the gesture.
            hints: Single-line key legend drawn along the bottom edge.
            show_scanline: Animate a sweeping scan bar for the holo effect.

        Returns:
            The same frame, modified in place.
        """
        height, width = frame.shape[:2]
        self._draw_corner_brackets(frame)
        if show_scanline:
            self._draw_scanline(frame)

        panel_width = max(330, int(width * 0.29))
        panel_height = 96 + 22 * len(status_lines)
        self._draw_panel(frame, 18, 18, panel_width, panel_height)

        self._neon_text(frame, "NEONVISION AI", (34, 50), 0.72, self.theme.hud, 2)
        self._neon_text(frame, fps_label, (34, 78), 0.52, self.theme.hud_accent, 1)
        self._neon_text(frame, gesture, (34, 108), 0.68, self.theme.hud_accent, 2)

        y = 132
        for line in status_lines:
            self._neon_text(frame, line, (34, y), 0.46, self.theme.hud, 1)
            y += 22

        if hints:
            self._neon_text(
                frame,
                hints,
                (24, height - 18),
                0.44,
                _dim(self.theme.hud, 0.85),
                1,
            )
        return frame

    def draw_hand_badge(self, frame: Frame, hand: HandResult) -> Frame:
        """Label a hand in-scene with its handedness and finger count.

        Args:
            frame: Frame to draw on.
            hand: The hand to annotate.

        Returns:
            The same frame, modified in place.
        """
        x, y, box_width, _ = hand.bbox
        label = f"{hand.label.upper()} - {hand.finger_count} | {hand.gesture}"
        origin = (max(8, x), max(28, y - 14))
        cv2.rectangle(
            frame,
            (origin[0] - 6, origin[1] - 18),
            (origin[0] + max(150, box_width), origin[1] + 8),
            _dim(self.theme.hand, 0.22),
            -1,
        )
        cv2.rectangle(
            frame,
            (origin[0] - 6, origin[1] - 18),
            (origin[0] + max(150, box_width), origin[1] + 8),
            self.theme.hand,
            1,
            cv2.LINE_AA,
        )
        self._neon_text(frame, label, origin, 0.48, self.theme.hand_core, 1)
        return frame

    def draw_face_bracket(self, frame: Frame, face: FaceMeshResult) -> Frame:
        """Draw a targeting bracket around a tracked face.

        Args:
            frame: Frame to draw on.
            face: The face to annotate.

        Returns:
            The same frame, modified in place.
        """
        x, y, box_width, box_height = face.bbox
        pad = int(0.08 * max(box_width, box_height))
        x0, y0 = x - pad, y - pad
        x1, y1 = x + box_width + pad, y + box_height + pad
        arm = max(12, int(0.18 * box_width))
        color = self.theme.hud_accent
        for corner_x, step_x in ((x0, arm), (x1, -arm)):
            for corner_y, step_y in ((y0, arm), (y1, -arm)):
                cv2.line(
                    frame,
                    (corner_x, corner_y),
                    (corner_x + step_x, corner_y),
                    color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.line(
                    frame,
                    (corner_x, corner_y),
                    (corner_x, corner_y + step_y),
                    color,
                    2,
                    cv2.LINE_AA,
                )
        return frame

    def draw_message(self, frame: Frame, text: str) -> Frame:
        """Centre a large notice on the frame, e.g. ``"NO SUBJECT DETECTED"``."""
        height, width = frame.shape[:2]
        size, _ = cv2.getTextSize(text, _FONT_BOLD, 0.9, 2)
        origin = ((width - size[0]) // 2, (height + size[1]) // 2)
        self._neon_text(frame, text, origin, 0.9, self.theme.hud_warning, 2, bold=True)
        return frame

    def draw_alert_banner(
        self,
        frame: Frame,
        text: str,
        subtext: str = "",
        pulse: float = 1.0,
    ) -> Frame:
        """Draw a full-width warning band, used by the drowsiness alarm.

        Args:
            frame: Frame to draw on.
            text: Headline, e.g. ``"DROWSINESS DETECTED"``.
            subtext: Optional second line.
            pulse: ``0``-``1`` intensity, so the caller can animate it.

        Returns:
            The same frame, modified in place.
        """
        height, width = frame.shape[:2]
        band_height = 78 if subtext else 58
        top = max(0, height // 2 - band_height // 2)
        bottom = min(height, top + band_height)
        strip = frame[top:bottom, 0:width]
        tint = np.empty_like(strip)
        # Alarm red, in BGR.
        tint[:] = (0, 0, 190)
        alpha = 0.35 + 0.35 * max(0.0, min(1.0, pulse))
        cv2.addWeighted(strip, 1.0 - alpha, tint, alpha, 0.0, dst=strip)
        cv2.rectangle(
            frame, (0, top), (width - 1, bottom - 1), (60, 60, 255), 2
        )

        size, _ = cv2.getTextSize(text, _FONT_BOLD, 0.95, 2)
        origin = ((width - size[0]) // 2, top + 40)
        self._neon_text(frame, text, origin, 0.95, (210, 230, 255), 2, bold=True)
        if subtext:
            sub_size, _ = cv2.getTextSize(subtext, _FONT, 0.5, 1)
            self._neon_text(
                frame,
                subtext,
                ((width - sub_size[0]) // 2, top + 66),
                0.5,
                (190, 210, 255),
                1,
            )
        return frame

    def draw_action_log(
        self, frame: Frame, lines: Sequence[str], title: str = "ACTIONS"
    ) -> Frame:
        """List recent gesture actions in the bottom-right corner.

        Args:
            frame: Frame to draw on.
            lines: Newest-first action notes.
            title: Panel heading.

        Returns:
            The same frame, modified in place.
        """
        if not lines:
            return frame
        height, width = frame.shape[:2]
        panel_width = 240
        panel_height = 30 + 20 * len(lines)
        x = width - panel_width - 18
        y = height - panel_height - 34
        if x < 0 or y < 0:
            return frame

        self._draw_panel(frame, x, y, panel_width, panel_height)
        self._neon_text(
            frame, title, (x + 12, y + 20), 0.42, self.theme.hud_accent, 1
        )
        for index, line in enumerate(lines):
            # Older entries fade out as they sink down the list.
            fade = 0.95 - 0.13 * index
            self._neon_text(
                frame,
                line[:26],
                (x + 12, y + 42 + index * 20),
                0.4,
                _dim(self.theme.hud, max(0.35, fade)),
                1,
            )
        return frame

    def draw_pen_cursor(
        self, frame: Frame, position: Tuple[int, int], active: bool, color: Color
    ) -> Frame:
        """Mark the air-drawing brush tip.

        Args:
            frame: Frame to draw on.
            position: Fingertip position in pixels.
            active: Whether the pen is currently down.
            color: Current palette colour.

        Returns:
            The same frame, modified in place.
        """
        radius = 13 if active else 9
        cv2.circle(frame, position, radius + 3, _dim(color, 0.4), 1, cv2.LINE_AA)
        cv2.circle(
            frame, position, radius, color, -1 if active else 2, cv2.LINE_AA
        )
        if active:
            cv2.circle(
                frame, position, radius - 5, _lighten(color, 0.8), -1, cv2.LINE_AA
            )
        return frame

    # ------------------------------------------------------------------ #
    # HUD primitives
    # ------------------------------------------------------------------ #
    def _neon_text(
        self,
        frame: Frame,
        text: str,
        origin: Tuple[int, int],
        scale: float,
        color: Color,
        thickness: int,
        bold: bool = False,
    ) -> None:
        """Draw text in layers to fake a cheap glow around the glyphs.

        The HUD is drawn after the blur pass, so a blurred halo would cost a
        second full-frame convolution. Stacking a dark backing, a dim wide
        stroke and a bright core is visually equivalent here and effectively
        free.

        Small text gets fewer, tighter layers: at HUD sizes a wide offset
        shadow is thicker than the glyph strokes themselves, which reads as
        ghosting rather than glow.
        """
        font = _FONT_BOLD if bold else _FONT
        small = scale < 0.5
        backing_offset = 0 if small else 1
        cv2.putText(
            frame,
            text,
            (origin[0] + backing_offset, origin[1] + backing_offset),
            font,
            scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
        if not small:
            cv2.putText(
                frame, text, origin, font, scale,
                _dim(color, 0.45), thickness + 1, cv2.LINE_AA,
            )
        cv2.putText(
            frame, text, origin, font, scale, color, thickness, cv2.LINE_AA
        )

    def _draw_panel(
        self, frame: Frame, x: int, y: int, width: int, height: int
    ) -> None:
        """Blend a translucent HUD panel with a neon border in place."""
        frame_height, frame_width = frame.shape[:2]
        x1 = min(frame_width, x + width)
        y1 = min(frame_height, y + height)
        if x1 <= x or y1 <= y:
            return

        roi = frame[y:y1, x:x1]
        tint = np.empty_like(roi)
        tint[:] = self.theme.panel
        cv2.addWeighted(roi, 0.35, tint, 0.65, 0.0, dst=roi)
        cv2.rectangle(
            frame, (x, y), (x1, y1), _dim(self.theme.hud, 0.8), 1, cv2.LINE_AA
        )
        # Accent tick along the panel's top-left edge.
        cv2.line(
            frame, (x, y), (x + max(24, width // 5), y), self.theme.hud_accent, 2
        )

    def _draw_corner_brackets(self, frame: Frame) -> None:
        """Frame the viewport with four holographic corner brackets."""
        height, width = frame.shape[:2]
        arm = max(28, width // 22)
        margin = 12
        color = _dim(self.theme.hud, 0.9)
        corners = (
            ((margin, margin), (arm, arm)),
            ((width - margin, margin), (-arm, arm)),
            ((margin, height - margin), (arm, -arm)),
            ((width - margin, height - margin), (-arm, -arm)),
        )
        for (cx, cy), (dx, dy) in corners:
            cv2.line(frame, (cx, cy), (cx + dx, cy), color, 2, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + dy), color, 2, cv2.LINE_AA)

    def _draw_scanline(self, frame: Frame) -> None:
        """Sweep a translucent bar down the frame, one step per call."""
        height, width = frame.shape[:2]
        band = max(2, height // 180)
        self._scan_offset = (self._scan_offset + max(2, height // 90)) % height
        top = self._scan_offset
        bottom = min(height, top + band)
        if bottom <= top:
            return
        strip = frame[top:bottom, 0:width]
        tint = np.empty_like(strip)
        tint[:] = self.theme.hud
        cv2.addWeighted(strip, 0.82, tint, 0.18, 0.0, dst=strip)

    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #
    def set_theme(self, theme: NeonTheme) -> None:
        """Swap the active theme, reallocating buffers if the scale changed."""
        rescale = theme.glow_scale != self.theme.glow_scale
        self.theme = theme
        if rescale and self._frame_size != (0, 0):
            self._allocate_buffers(*self._frame_size)

    def toggle_glow(self) -> bool:
        """Flip the glow pass on or off.

        Returns:
            The new state of :attr:`glow_enabled`.
        """
        self.glow_enabled = not self.glow_enabled
        return self.glow_enabled

    @property
    def stats(self) -> RenderStats:
        """Counters for the geometry queued in the current frame."""
        return self._stats

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<NeonRenderer theme={self.theme.name!r} "
            f"glow={self.glow_enabled} size={self._frame_size}>"
        )


# ---------------------------------------------------------------------- #
# Head pose
# ---------------------------------------------------------------------- #
@dataclass(slots=True)
class HeadPose:
    """Head orientation derived from the face mesh.

    Attributes:
        yaw: Left/right rotation in degrees; positive turns to the subject's
            left as seen in a mirrored preview.
        pitch: Up/down rotation in degrees; positive looks down.
        roll: In-plane tilt in degrees; positive tilts clockwise on screen.
        basis: ``3x3`` array whose rows are the head's right, down and
            forward axes in frame space.
        valid: ``False`` when the landmarks were degenerate.
    """

    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    basis: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    valid: bool = False

    def label(self) -> str:
        """Compact readout, e.g. ``"Y -12  P  4  R  2"``."""
        if not self.valid:
            return "Y  --  P  --  R  --"
        return f"Y {self.yaw:>4.0f}  P {self.pitch:>4.0f}  R {self.roll:>4.0f}"


#: Landmarks used to build the head basis: eye corners, forehead and chin.
_RIGHT_EYE_OUTER: Final[int] = 33
_LEFT_EYE_OUTER: Final[int] = 263
_FOREHEAD: Final[int] = 10
_CHIN: Final[int] = 152


def estimate_head_pose(face: FaceMeshResult) -> HeadPose:
    """Estimate head orientation from three facial reference directions.

    Rather than solving a full PnP problem, this builds an orthonormal basis
    straight from the mesh: the eye line gives the head's right axis, the
    forehead-to-chin line gives its down axis, and their cross product gives
    the forward (face-normal) axis. Re-orthogonalising the down axis against
    the normal keeps the basis clean when the two source directions are not
    perfectly perpendicular.

    Angles are then read off the resulting axes -- roll from the eye line's
    tilt in the image plane, yaw and pitch from where the face normal points.
    They are close approximations rather than an exact Euler decomposition,
    which is all the holographic preview and HUD need.

    Args:
        face: Landmark result carrying normalized x/y/z.

    Returns:
        A :class:`HeadPose`; ``valid`` is ``False`` if it could not be built.
    """
    width, height = face.frame_size
    if width <= 0 or height <= 0 or face.normalized.shape[0] <= _CHIN:
        return HeadPose()

    # MediaPipe's z is scaled roughly like x, so multiply it by width too to
    # land in a consistent, near-metric space.
    points = face.normalized[:, :3].astype(np.float64)
    points = points * (width, height, width)

    right = points[_LEFT_EYE_OUTER] - points[_RIGHT_EYE_OUTER]
    down = points[_CHIN] - points[_FOREHEAD]
    if np.linalg.norm(right) < 1e-6 or np.linalg.norm(down) < 1e-6:
        return HeadPose()

    right /= np.linalg.norm(right)
    forward = np.cross(right, down)
    if np.linalg.norm(forward) < 1e-6:
        return HeadPose()
    forward /= np.linalg.norm(forward)
    down = np.cross(forward, right)

    basis = np.stack([right, down, forward]).astype(np.float32)
    clamp = lambda value: float(max(-1.0, min(1.0, value)))  # noqa: E731
    return HeadPose(
        yaw=math.degrees(math.asin(clamp(forward[0]))),
        pitch=math.degrees(math.asin(clamp(forward[1]))),
        roll=math.degrees(math.atan2(right[1], right[0])),
        basis=basis,
        valid=True,
    )


# ---------------------------------------------------------------------- #
# Holographic side preview
# ---------------------------------------------------------------------- #
class HoloPreview:
    """Standalone breathing dot-cloud face mesh, beside the camera feed.

    The panel is a genuine projection, not a copy of the overlay: MediaPipe
    returns a z coordinate per landmark, so the cloud is centred, scaled to a
    unit size, given a light perspective divide and projected into the panel.
    Keeping the depth lets the dots be sorted into depth bands and drawn
    far-to-near with rising radius and brightness, which is what reads as a
    floating hologram rather than a flat stipple.

    **Dots only, by design.** No connecting lines are ever drawn here; the
    wireframe lives on the camera feed, and the panel is its point-cloud
    counterpart.

    **High density by interpolation.** 478 landmarks alone leave visible gaps,
    so extra points are generated *along every mesh edge*: for each of the
    1 322 tesselation edges, ``density`` points are linearly interpolated
    between its two endpoints. At the default that is 1 322 x 4 + 478 =
    **5 766 dots**, spaced roughly 2-3 px apart on screen, which reads as a
    high-resolution Lidar scan rather than a sparse stipple. The interpolation
    happens in the unit-cloud space *before* the breathing and perspective
    passes, so the generated points carry correct depth and pulse with the
    rest of the cloud.

    **Breathing, not rotating.** The cloud never spins. Instead a sine wave
    drives three quantities at once each frame:

    * the *spacing* between dots, which expands and contracts around the
      centroid,
    * the dot *radius*, and
    * the halo *intensity*.

    The phase is offset by each dot's distance from the centre, so the pulse
    travels outward as a slow radial ripple instead of the whole cloud
    scaling in lockstep -- that is what makes it look alive rather than
    mechanically zoomed.

    Args:
        theme: Colour theme, kept in step with the main renderer.
        width_ratio: Panel width as a fraction of the camera frame's width.
        density: Points interpolated along each mesh edge. ``0`` plots the
            raw landmarks only; ``4`` (the default) gives ~5 800 dots.
        breathe_speed: Breathing rate in radians per second.
        breathe_amount: Peak spacing change, as a fraction of the cloud size.
        glow_enabled: Whether the panel gets the blurred halo.
    """

    #: Number of depth bands used for the far-to-near size/brightness ramp.
    DEPTH_BANDS: Final[int] = 4
    #: Perspective strength; larger is flatter.
    FOCAL: Final[float] = 2.6
    #: How far the breathing phase lags with distance from the centre, in
    #: radians. This is what turns a uniform pulse into a travelling ripple.
    RIPPLE: Final[float] = 1.5
    #: Dot radius range, in pixels, from the furthest to the nearest band.
    #: Dense clouds want small dots, or neighbours merge into a solid mass.
    DOT_RADIUS_MIN: Final[int] = 1
    DOT_RADIUS_MAX: Final[int] = 2
    #: Density presets cycled by the ``K`` key: points added per mesh edge.
    DENSITY_LEVELS: Final[Tuple[int, ...]] = (0, 2, 4, 6)

    def __init__(
        self,
        theme: Optional[NeonTheme] = None,
        width_ratio: float = 0.34,
        density: int = 4,
        breathe_speed: float = 1.9,
        breathe_amount: float = 0.07,
        glow_enabled: bool = True,
    ) -> None:
        self.theme = theme or NeonTheme()
        self.width_ratio = float(min(0.6, max(0.15, width_ratio)))
        self.breathe_speed = float(breathe_speed)
        self.breathe_amount = float(min(0.4, max(0.0, breathe_amount)))
        self.breathing_enabled = True

        self._density = 0
        self._edge_a: np.ndarray = FaceTopology.TESSELATION[:, 0]
        self._edge_b: np.ndarray = FaceTopology.TESSELATION[:, 1]
        self._fractions: Optional[np.ndarray] = None
        self.set_density(density)

        self._renderer = NeonRenderer(
            theme=self._panel_theme(self.theme),
            glow_enabled=glow_enabled,
            node_stride=0,
        )
        self._panel: Optional[Frame] = None
        self._canvas: Optional[Frame] = None
        self._panel_size: Tuple[int, int] = (0, 0)
        self._feed_size: Tuple[int, int] = (0, 0)
        self._phase: float = 0.0
        self._breath: float = 0.0
        self.pose = HeadPose()

    # ------------------------------------------------------------------ #
    # Composition
    # ------------------------------------------------------------------ #
    def compose(
        self,
        feed: Frame,
        face: Optional[FaceMeshResult],
        dt: float = 1.0 / 30.0,
        status: Sequence[str] = (),
    ) -> Frame:
        """Build the wide canvas: holographic panel left, camera feed right.

        Args:
            feed: The rendered camera frame.
            face: Face to project, or ``None`` for the idle panel.
            dt: Seconds since the last call, which advances the breathing
                phase. Driving the animation from elapsed time rather than a
                frame counter keeps the pulse at the same real-world rate
                whatever the frame rate does.
            status: Extra lines printed under the pose readout.

        Returns:
            The composed canvas. Reused between frames, so treat it as
            valid only until the next call.
        """
        height, width = feed.shape[:2]
        panel_width = max(180, int(width * self.width_ratio))
        if (width, height) != self._feed_size or self._panel is None:
            self._allocate(width, height, panel_width)

        assert self._panel is not None and self._canvas is not None
        panel = self._panel
        panel[:] = 0

        if self.breathing_enabled:
            self._phase = (self._phase + self.breathe_speed * dt) % (2.0 * math.pi)
            self._breath = math.sin(self._phase)
        else:
            self._breath = 0.0

        self.pose = estimate_head_pose(face) if face is not None else HeadPose()
        self._draw_panel(panel, face, status)

        self._canvas[:, :panel_width] = panel
        self._canvas[:, panel_width:] = feed
        # Seam: a bright rule so the panel reads as a separate instrument.
        # It sits on the panel's last column, not the feed's first, so the
        # camera image is copied through untouched.
        cv2.line(
            self._canvas,
            (panel_width - 1, 0),
            (panel_width - 1, height - 1),
            _dim(self.theme.hud, 0.85),
            1,
        )
        return self._canvas

    def _allocate(self, width: int, height: int, panel_width: int) -> None:
        """(Re)allocate the panel and canvas buffers for a new feed size."""
        self._feed_size = (width, height)
        self._panel_size = (panel_width, height)
        self._panel = np.zeros((height, panel_width, 3), dtype=np.uint8)
        self._canvas = np.zeros((height, width + panel_width, 3), dtype=np.uint8)

    # ------------------------------------------------------------------ #
    # Panel contents
    # ------------------------------------------------------------------ #
    def _draw_panel(
        self, panel: Frame, face: Optional[FaceMeshResult], status: Sequence[str]
    ) -> None:
        """Render the hologram and its labels into the panel."""
        panel_width, panel_height = self._panel_size
        self._renderer.begin_frame(panel)

        if face is not None:
            projected = self._project(face)
            if projected is not None:
                self._queue_dots(projected)
        self._renderer.render()

        self._draw_platform(panel)
        self._draw_labels(panel, face is not None, status)
        cv2.rectangle(
            panel,
            (4, 4),
            (panel_width - 5, panel_height - 5),
            _dim(self.theme.hud, 0.5),
            1,
        )

    def _project(self, face: FaceMeshResult) -> Optional[np.ndarray]:
        """Project the landmark cloud into panel space, mid-breath.

        Returns:
            ``(N, 3)`` float32 array of panel-space ``x``, ``y`` and depth,
            or ``None`` when the landmarks are unusable.
        """
        width, height = face.frame_size
        if width <= 0 or height <= 0:
            return None

        points = face.normalized[:, :3].astype(np.float32)
        points = points * np.float32((width, height, width))

        # Centre and scale robustly: the median and a high percentile of the
        # radius, rather than the mean and the maximum. A single mis-tracked
        # landmark would otherwise rescale the whole hologram, and even
        # ordinary jitter in the outermost point would make it flicker in
        # size -- which would fight the deliberate breathing below.
        points -= np.median(points, axis=0)
        extent = float(np.percentile(np.linalg.norm(points, axis=1), 95.0))
        if extent < 1e-3:
            return None
        points /= extent  # unit size, so panel scale is resolution-independent

        points = self._densify(points)

        if self.breathe_amount > 0.0:
            # Radius is recomputed on the dense cloud so the interpolated
            # points ripple in step with the landmarks around them.
            radius = np.linalg.norm(points, axis=1)
            points = points * self._breathing_scale(radius)[:, None]

        panel_width, panel_height = self._panel_size
        scale = min(panel_width, panel_height) * 0.36
        # Perspective divide: nearer landmarks (smaller z) spread wider.
        depth = np.clip(points[:, 2], -0.95, 0.95)
        perspective = self.FOCAL / (self.FOCAL + depth)

        projected = np.empty((points.shape[0], 3), dtype=np.float32)
        projected[:, 0] = panel_width * 0.5 + points[:, 0] * perspective * scale
        projected[:, 1] = panel_height * 0.46 + points[:, 1] * perspective * scale
        projected[:, 2] = depth
        return projected

    def _densify(self, points: np.ndarray) -> np.ndarray:
        """Interpolate extra points along every mesh edge.

        Args:
            points: ``(478, 3)`` unit-cloud landmark positions.

        Returns:
            ``(478 + edges * density, 3)`` -- the landmarks followed by the
            interpolated points. Returns the input unchanged at density ``0``.
        """
        if self._fractions is None:
            return points
        start = points[self._edge_a]
        delta = points[self._edge_b] - start
        # (edges, density, 3) -- one vectorised lerp for the whole mesh.
        interpolated = start[:, None, :] + delta[:, None, :] * self._fractions
        return np.vstack((points, interpolated.reshape(-1, 3)))

    def _dot_sizes(self) -> Tuple[int, int, float]:
        """Dot radii for the current density.

        Returns:
            ``(far_radius, near_radius, radius_pulse)`` -- the last being how
            much the breath is allowed to grow a dot, in pixels.
        """
        if self._density == 0:
            return 2, 3, 1.0
        if self._density <= 4:
            return 1, 2, 0.0
        return 1, 1, 0.0

    def _breathing_scale(self, radius: np.ndarray) -> np.ndarray:
        """Per-dot spacing multiplier for the current breath.

        Each dot's phase lags by its distance from the centroid, so the
        expansion sweeps outward as a ripple. The whole computation is one
        vectorised sine over ~478 values -- microseconds per frame.

        Args:
            radius: Per-dot distance from the centroid, in unit-cloud terms.

        Returns:
            ``(N,)`` float32 multipliers centred on ``1.0``.
        """
        phase = self._phase - radius * self.RIPPLE
        return (1.0 + self.breathe_amount * np.sin(phase)).astype(np.float32)

    def _queue_dots(self, projected: np.ndarray) -> None:
        """Queue landmark dots in depth bands, far/small/dim to near/big/bright.

        No edges are queued: the panel is a pure point cloud. The breath also
        drives dot radius and halo intensity here, so the cloud pulses in
        brightness as it expands rather than merely changing size.
        """
        if projected.shape[0] == 0:
            return

        points2d = projected[:, :2].astype(np.int32)
        # Larger z is further away, so descending order draws back-to-front.
        order = np.argsort(-projected[:, 2])
        bands = np.array_split(order, self.DEPTH_BANDS)

        # 0 at the trough of the breath, 1 at the peak.
        swell = 0.5 + 0.5 * self._breath
        # Dot size has to fall as density rises. Interpolated points sit only
        # ~2-3 px apart on screen, so a 3 px dot would fuse with its
        # neighbours and turn the cloud back into the wireframe this panel
        # exists to avoid. Radius pulsing is therefore reserved for the sparse
        # mode, where there is room; dense clouds pulse via spacing and glow
        # intensity instead, which stays legible.
        small, large, pulse = self._dot_sizes()
        # A dense cloud already fills the half-resolution glow buffer with one
        # pixel per dot, so widening the seed would cost work for no visible
        # halo. Only the sparse landmark-only mode needs a wider seed.
        seed = 0 if self._density >= 2 else 1

        for index, band in enumerate(bands):
            if band.size == 0:
                continue
            nearness = (index + 1) / self.DEPTH_BANDS
            radius = small + (large - small) * nearness
            self._renderer.add_point_cloud(
                points2d[band],
                color=_dim(self.theme.mesh, 0.35 + 0.65 * nearness),
                core_color=_dim(self.theme.mesh_core, 0.4 + 0.6 * nearness),
                # Nearer bands swell slightly more, which reads as depth.
                radius=int(round(radius + pulse * swell * nearness)),
                glow_radius=seed,
                intensity=0.6 + 0.4 * swell,
            )

    def _draw_platform(self, panel: Frame) -> None:
        """Draw the projector plinth the hologram appears to stand on.

        The rings breathe with the cloud -- in antiphase, so the plinth widens
        as the hologram contracts. It reads as the projector working.
        """
        panel_width, panel_height = self._panel_size
        center = (panel_width // 2, int(panel_height * 0.82))
        swell = 1.0 - 0.06 * self._breath
        rings = (
            int(panel_width * 0.30 * swell),
            int(panel_width * 0.20 * swell),
        )
        for index, radius in enumerate(rings):
            cv2.ellipse(
                panel,
                center,
                (radius, max(4, radius // 5)),
                0,
                0,
                360,
                _dim(self.theme.hud, 0.55 - 0.2 * index),
                1,
                cv2.LINE_AA,
            )
        cv2.line(
            panel,
            (center[0] - int(panel_width * 0.30), center[1]),
            (center[0] + int(panel_width * 0.30), center[1]),
            _dim(self.theme.hud, 0.3),
            1,
            cv2.LINE_AA,
        )

    def _draw_labels(
        self, panel: Frame, has_face: bool, status: Sequence[str]
    ) -> None:
        """Write the panel heading, pose readout and status lines."""
        panel_width, panel_height = self._panel_size
        self._renderer._neon_text(
            panel, "HOLO MESH", (18, 34), 0.62, self.theme.hud, 2
        )
        self._renderer._neon_text(
            panel,
            "HEAD POSE / DEG",
            (18, 56),
            0.38,
            _dim(self.theme.hud, 0.8),
            1,
        )
        self._renderer._neon_text(
            panel, self.pose.label(), (18, 80), 0.5, self.theme.hud_accent, 1
        )

        y = 104
        for line in status:
            self._renderer._neon_text(
                panel, line[:30], (18, y), 0.4, _dim(self.theme.hud, 0.9), 1
            )
            y += 20

        if not has_face:
            text = "NO SIGNAL"
            size, _ = cv2.getTextSize(text, _FONT_BOLD, 0.6, 2)
            self._renderer._neon_text(
                panel,
                text,
                ((panel_width - size[0]) // 2, panel_height // 2),
                0.6,
                self.theme.hud_warning,
                2,
                bold=True,
            )

    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #
    @staticmethod
    def _panel_theme(theme: NeonTheme) -> NeonTheme:
        """Trim the glow spread for the panel.

        The hologram is split into depth bands, and each band is its own
        polyline batch that pays for a full set of glow stamps. Narrowing the
        spread drops one stamp per band -- a third of the panel's line work --
        which is invisible at panel scale because the bands overlap anyway.
        """
        return replace(theme, glow_spread=min(3, theme.glow_spread))

    def set_theme(self, theme: NeonTheme) -> None:
        """Adopt a new theme, keeping the internal renderer in step."""
        self.theme = theme
        self._renderer.set_theme(self._panel_theme(theme))

    def set_glow(self, enabled: bool) -> None:
        """Enable or disable the panel's glow pass."""
        self._renderer.glow_enabled = bool(enabled)

    def toggle_breathing(self) -> bool:
        """Start or stop the breathing animation.

        Stopping settles the cloud at its neutral size rather than freezing
        it mid-breath, so a paused hologram always looks deliberate.

        Returns:
            The new breathing state.
        """
        self.breathing_enabled = not self.breathing_enabled
        if not self.breathing_enabled:
            self._phase = 0.0
            self._breath = 0.0
        return self.breathing_enabled

    def set_density(self, density: int) -> int:
        """Set how many points are interpolated along each mesh edge.

        Rebuilds the interpolation fraction table, which is constant for a
        given density and therefore computed once here rather than per frame.

        Args:
            density: Points added per edge; clamped to ``0``-``12``.

        Returns:
            The density actually applied.
        """
        self._density = int(max(0, min(12, density)))
        if self._density <= 0:
            self._fractions = None
        else:
            # Fractions strictly between the endpoints -- the endpoints are
            # already present as landmarks, so including them would stack
            # duplicate dots on every vertex.
            steps = np.linspace(
                0.0, 1.0, self._density + 2, dtype=np.float32
            )[1:-1]
            self._fractions = steps.reshape(1, -1, 1)
        return self._density

    def cycle_density(self) -> int:
        """Advance to the next density preset in :attr:`DENSITY_LEVELS`."""
        levels = self.DENSITY_LEVELS
        try:
            index = levels.index(self._density)
        except ValueError:
            index = -1
        return self.set_density(levels[(index + 1) % len(levels)])

    @property
    def density(self) -> int:
        """Points interpolated along each mesh edge."""
        return self._density

    @property
    def point_count(self) -> int:
        """Total dots the cloud will plot for a 478-landmark face."""
        return FaceTopology.REFINED_LANDMARK_COUNT + (
            int(self._edge_a.shape[0]) * self._density
        )

    @property
    def breath(self) -> float:
        """Current breath value in ``[-1, 1]``, for HUD readouts."""
        return self._breath

    @property
    def panel_width(self) -> int:
        """Current panel width in pixels."""
        return self._panel_size[0]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<HoloPreview density={self._density} "
            f"points={self.point_count} "
            f"breathing={self.breathing_enabled} panel={self._panel_size}>"
        )
