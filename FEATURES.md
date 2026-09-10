# NeonVision AI — Feature Guide

Every feature in NeonVision AI, what it does, **how to turn it on**, and how to
confirm it is actually working.

**Author:** ChamathDilshanC

---

## Contents

1. [Quick start](#1-quick-start)
2. [The mode system — read this first](#2-the-mode-system--read-this-first)
3. [Vision features](#3-vision-features)
4. [Holographic UI features](#4-holographic-ui-features)
5. [Gesture control features](#5-gesture-control-features)
6. [Safety & monitoring features](#6-safety--monitoring-features)
7. [Video I/O features](#7-video-io-features)
8. [Complete keyboard reference](#8-complete-keyboard-reference)
9. [Complete gesture reference](#9-complete-gesture-reference)
10. [Complete command-line reference](#10-complete-command-line-reference)
11. [Reading the HUD](#11-reading-the-hud)
12. [Recipes](#12-recipes)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Quick start

```bash
# Windows
.venv\Scripts\python.exe main.py

# macOS / Linux
python main.py
```

A window opens showing two halves: the **holographic dot panel** on the left
and your **camera feed with the neon mesh** on the right.

On first launch, four things are live immediately with no configuration:

| Live at startup | What you should see |
| --- | --- |
| Face mesh + neon glow | Glowing wireframe on your face |
| Holo dot panel | Breathing point cloud, left side |
| Hand tracking + finger counting | `Gesture: 3 Fingers` in the HUD |
| Drowsiness watchdog | `Eyes OPEN (EAR 0.31)` in the HUD |

Gesture control of your computer is **deliberately off** at startup. See the
next section.

**Recommended first run** — watch the gestures work without letting them touch
your machine:

```bash
python main.py --dry-run
```

---

## 2. The mode system — read this first

The air mouse, media control and air drawing all want the same hand, so only
**one interaction mode is live at a time**. Press `TAB` to cycle:

```
IDLE  ->  AIR MOUSE  ->  MEDIA / SLIDES  ->  AIR DRAWING  ->  IDLE ...
```

| Mode | Sends to the OS? | What your hand does |
| --- | --- | --- |
| **IDLE** *(startup default)* | No | Nothing. Tracking and visuals only. |
| **AIR MOUSE** | Yes | Index fingertip moves the cursor; pinch clicks/drags. |
| **MEDIA / SLIDES** | Yes | Play/pause, mute, volume, slide arrows. |
| **AIR DRAWING** | No | Paints neon strokes on screen. |

### Three independent safety layers

| Layer | Control | Effect |
| --- | --- | --- |
| **Mode** | `TAB` | `IDLE` sends nothing. This is the startup state. |
| **Arm switch** | `A` | Disarms every mode at once. Shown as `[DISARMED]`. |
| **Dry-run** | `--dry-run` | Recognises everything, sends nothing. `[DRY-RUN]`. |

Plus a hardware-style emergency stop: shoving the mouse pointer into a screen
corner trips PyAutoGUI's failsafe, which **disarms the engine automatically**.

**Air drawing is not gated by the arm switch**, because it paints inside the
app window and never touches your OS. It works even when disarmed.

Start a specific mode directly:

```bash
python main.py --mode mouse      # idle | mouse | media | draw
```

---

## 3. Vision features

### 3.1 Face mesh (468 / 478 landmarks)

**What:** A dense wireframe over your face. 478 landmarks with iris
refinement, 468 without.

| | |
| --- | --- |
| **Activate** | On by default |
| **Toggle** | `V` |
| **Change style** | `M` cycles `hybrid` → `tesselation` → `contours` → `off` |
| **Start style** | `--mesh-style tesselation` |
| **Confirm** | HUD shows `Faces 1` and `Edges 1511` |

| Style | What it draws |
| --- | --- |
| `hybrid` *(default)* | Full tesselation **plus** feature contours |
| `tesselation` | The dense 1 322-edge triangle mesh only |
| `contours` | Feature outlines only — eyes, lips, brows, face oval |
| `off` | No face mesh (hands and paint still draw) |

Drop iris refinement for ~2–3 ms/frame: `--no-refine` (gives 468 landmarks;
the iris rings and eye dots disappear).

### 3.2 Neon glow effect

**What:** Every stroke is drawn multiple times at decreasing thickness into a
half-resolution buffer, blurred once, upscaled, and blended additively — then
over-drawn with a crisp bright filament.

| | |
| --- | --- |
| **Activate** | On by default |
| **Toggle** | `G` (affects the main mesh *and* the holo dots) |
| **Disable at start** | `--no-glow` (saves ~5 ms/frame) |
| **Confirm** | HUD shows `Glow ON`; strokes have soft halos |

### 3.3 Hand tracking & finger counting

**What:** Up to two hands, 21 landmarks each, with the extended-finger count
and a named pose.

| | |
| --- | --- |
| **Activate** | On by default |
| **Toggle** | `H` |
| **Two hands** | `--max-hands 2` (default) |
| **Confirm** | HUD shows `Gesture: 3 Fingers` and `Pose R:Peace` |

Finger detection is **rotation invariant** — it compares fingertip-to-wrist
distance against knuckle-to-wrist distance, normalised by palm size, so
counting keeps working with your hand tilted or upside down.

Stabilisation: the reported pose is a majority vote over the last 5 frames, so
single-frame glitches do not flicker on screen. Tune with
`--gesture-smoothing N` (`1` disables it, higher is steadier but laggier).

### 3.4 Landmark nodes

**What:** Small dots on every n-th face landmark, for the scanner look.

| | |
| --- | --- |
| **Activate** | On by default (every 8th landmark) |
| **Density** | `--node-stride 4` (denser) / `--node-stride 16` (sparser) |
| **Disable** | `--node-stride 0` |

---

## 4. Holographic UI features

### 4.1 The HOLO MESH dot panel (high-density cloud)

**What:** A standalone floating point cloud of your head on black, left of the
camera feed. **Dots only — no connecting lines are ever drawn here.** The
wireframe lives on the camera feed; this panel is its point-cloud counterpart.

**High density by interpolation.** The 478 landmarks alone leave visible gaps,
so extra points are generated *along every mesh edge*: for each of the 1 322
tesselation edges, `density` points are linearly interpolated between its two
endpoints. The result reads as a high-resolution Lidar scan rather than a
sparse stipple.

| Density (`K`) | Dots plotted | Spacing on a real face | Look |
| --- | --- | --- | --- |
| `x0/edge` | 478 | ~12.6 px | Landmarks only |
| `x2/edge` | 3 122 | ~4.2 px | Dense |
| **`x4/edge`** *(default)* | **5 766** | **~2.5 px** | **Tightly packed** |
| `x6/edge` | 8 410 | ~1.8 px | Extremely dense |

Dot radius shrinks automatically as density rises (2–3 px at `x0`, 1–2 px at
`x4`, 1 px at `x6`), so neighbours stay separate instead of fusing back into a
wireframe.

It is also a real 3D projection: MediaPipe's per-landmark `z` is centred,
scaled and given a perspective divide, then dots are sorted into 4 depth bands
and drawn far-to-near with rising radius and brightness. Interpolation happens
*before* the breathing and perspective passes, so generated points carry
correct depth and pulse with the rest of the cloud.

| | |
| --- | --- |
| **Activate** | On by default at `x4/edge` |
| **Toggle panel** | `O` |
| **Hide at start** | `--no-holo` (saves ~6 ms/frame) |
| **Cycle density** | `K` → x0 → x2 → x4 → x6 |
| **Set at start** | `--holo-density 6` (accepts 0–12) |
| **Panel width** | `--holo-width 0.34` (fraction of frame width, 0.15–0.6) |
| **Confirm** | Panel shows `DOTS 5766  (x4/edge)` |

### 4.2 Breathing animation

**What:** The cloud never rotates. A sine wave drives three things together:
the **spacing** between dots, the dot **radius**, and the halo **intensity**.
Each dot's phase is offset by its distance from the centre, so the pulse
travels outward as a slow radial ripple rather than the whole cloud scaling in
lockstep. The projector plinth pulses in antiphase.

| | |
| --- | --- |
| **Activate** | On by default |
| **Toggle** | `L` (freezes at neutral size, not mid-breath) |
| **Speed** | `--breathe-speed 1.9` (radians/second; higher = faster) |
| **Depth** | `--breathe-amount 0.07` (peak spacing swing; `0` = still) |
| **Confirm** | Panel shows a moving `BREATHE [\|\|\|\|....]` bar |

Examples:

```bash
python main.py --breathe-speed 3.2 --breathe-amount 0.12   # energetic
python main.py --breathe-speed 1.0 --breathe-amount 0.04   # calm, slow
python main.py --breathe-amount 0                           # frozen cloud
```

### 4.3 Head-pose readout

**What:** Live yaw, pitch and roll in degrees, from an orthonormal basis built
out of your eye line and forehead-to-chin axis.

| | |
| --- | --- |
| **Activate** | Automatic whenever a face is tracked |
| **Confirm** | Panel shows `Y -14 P 1 R 4`; turn your head and watch it move |

`Y` = yaw (turn left/right), `P` = pitch (nod up/down), `R` = roll (tilt).

### 4.4 Gesture-controlled colour palette

**What:** The number of fingers you hold up recolours **both** the main
wireframe and the holographic dots, using the project's brand colours.

| Fingers | Hex | OpenCV BGR | Colour |
| --- | --- | --- | --- |
| **1** | `#4300FF` | `(255, 0, 67)` | Electric indigo |
| **2** | `#0065F8` | `(248, 101, 0)` | Azure |
| **3** | `#00CAFF` | `(255, 202, 0)` | Cyan |
| **4** | `#00FFDE` | `(222, 255, 0)` | Aquamarine |
| 0 or 5 | — | — | Keeps the base theme |

| | |
| --- | --- |
| **Activate** | On by default — just hold up 1–4 fingers |
| **Toggle** | `P` |
| **Disable at start** | `--no-palette` |
| **Confirm** | HUD `Theme` line changes to `#00CAFF Cyan`; panel shows `PALETTE 3F` |

Works in **every mode**, including `IDLE` — it is a visual response, not an OS
action. Holding 0 or 5 fingers deliberately keeps the base theme, so the
palette is something you opt into rather than a constant flicker.

### 4.5 Base colour themes

**What:** Four full colour schemes underneath the gesture palette.

| | |
| --- | --- |
| **Activate** | Starts on `Cyan Holo` |
| **Cycle** | `C` → Cyan Holo → Matrix Green → Magenta Pulse → Amber Circuit |
| **Confirm** | HUD `Theme` line shows the name |

The gesture palette overrides the theme's mesh colours while 1–4 fingers are
up; the theme returns at 0 or 5 fingers.

### 4.6 HUD, action log and effects

| Element | Toggle | What it shows |
| --- | --- | --- |
| Main HUD panel | `B` | FPS, gesture, mode, counts, stage timings |
| Action log | `B` | Last 6 gesture actions, bottom-right |
| Toast messages | always | Confirmations of key presses, bottom-left |
| Corner brackets | always | Holographic viewport frame |
| Face bracket | always | Targeting corners around each face |
| Hand badge | always | `RIGHT - 5 \| Open Palm` above each hand |
| Scanline | always | Sweeping bar down the frame |
| Fullscreen | `F` | Fills the display |

---

## 5. Gesture control features

> All of section 5 requires a mode other than `IDLE` (`TAB`) **and** the arm
> switch on (`A`). The HUD must read `[ARMED]`, not `[IDLE]` or `[DISARMED]`.

### 5.1 Air mouse

**What:** Your index fingertip drives the OS cursor.

| | |
| --- | --- |
| **Activate** | `TAB` until the HUD shows `Mode AIR MOUSE` |
| **Or start in it** | `--mode mouse` |
| **Move** | Point your index finger and move your hand |
| **Confirm** | HUD shows `Pinch 0.87` and the action log shows `move` |

The inner 64 % of the frame maps to the whole screen, so you can reach every
screen edge without moving your hand out of the camera's view. Cursor motion is
exponentially smoothed to remove tracking jitter.

### 5.2 Pinch to click and drag

**What:** Touching thumb to index presses the mouse button; separating them
releases it. A quick pinch is a **click**; holding it while moving is a
**drag**.

| | |
| --- | --- |
| **Activate** | AIR MOUSE mode |
| **Click** | Pinch and release quickly |
| **Drag** | Pinch, move your hand, then release |
| **Confirm** | Action log shows `click`, then `mouse_up` |

Thresholds use hysteresis so a hand hovering at the boundary cannot chatter:

| Threshold | Value (÷ palm size) | Meaning |
| --- | --- | --- |
| `PINCH_CLOSE` | `0.32` | Below this, the button presses |
| `PINCH_OPEN` | `0.46` | Above this, it releases |

Losing the hand also releases the button, so a button can never get stuck down.

### 5.3 Media control

**What:** Play/pause and mute with two poses.

| | |
| --- | --- |
| **Activate** | `TAB` until the HUD shows `Mode MEDIA / SLIDES` |
| **Or start in it** | `--mode media` |
| **Play / Pause** | Show **Peace** ✌ (index + middle) |
| **Mute** | Show a **Fist** ✊ |
| **Confirm** | Action log shows `play_pause` / `mute` |

Actions are **edge-triggered**: they fire once when you form the pose, not
every frame. Holding a fist mutes once, not forty times. Two separate
mechanisms keep it that way:

| Mechanism | Value | Adjust via |
| --- | --- | --- |
| Repeat cooldown | 0.7 s | `cooldown=` on `GestureActionEngine` |
| Pose detection steadiness | 5-frame vote | `--gesture-smoothing N` |

To re-fire the same action, drop the pose and form it again.

### 5.4 Slide controller (swipe)

**What:** An open-palm swipe sends the left/right arrow keys, for presentations.

| | |
| --- | --- |
| **Activate** | MEDIA / SLIDES mode |
| **Next slide** | Open palm, sweep **right** |
| **Previous slide** | Open palm, sweep **left** |
| **Confirm** | Action log shows `slide_right` / `slide_left` |

Requirements (so ordinary hand movement is not a swipe):

| Requirement | Value |
| --- | --- |
| Pose | Must be **Open Palm** (all 5 fingers) |
| Distance | ≥ 17 % of the frame width |
| Time window | Within 0.45 s |

A slow drift across the frame does **not** trigger it.

### 5.5 Volume control

**What:** The gap between thumb and index tip sets the system volume, like
pinching a dial. Normalised by palm size, so your distance from the camera does
not matter.

| | |
| --- | --- |
| **Activate** | MEDIA / SLIDES mode |
| **Pose** | **Gun** 👆 — thumb **and** index extended, other three folded |
| **Louder** | Widen the thumb–index gap |
| **Quieter** | Narrow the gap |
| **Confirm** | HUD shows `Volume 62% (system)`; log shows `vol 62%` |

| Gap (÷ palm size) | Volume |
| --- | --- |
| `0.22` or less | 0 % |
| `1.05` or more | 100 % |

Using a dedicated pose means volume only changes when you deliberately ask for
it — showing an open palm or a fist never moves it.

**Backends** — the HUD names the one in use:

| Shown | Meaning |
| --- | --- |
| `system` | Absolute control via pycaw (Windows). Reads the true level. |
| `keys` | Synthesised `volumeup`/`volumedown` presses. Relative only. |
| `none` | Neither available; volume does nothing. |

### 5.6 Air drawing

**What:** Your index fingertip paints glowing neon strokes in the current
palette colour.

| | |
| --- | --- |
| **Activate** | `TAB` until the HUD shows `Mode AIR DRAWING` |
| **Or start in it** | `--mode draw` |
| **Pen down** | Index finger **only** (middle, ring, pinky folded) |
| **Pen up** | Show **Peace**, an open palm, or a fist |
| **Undo stroke** | `Z` |
| **Clear all** | `X` |
| **Confirm** | HUD shows `DRAWING strokes 3 points 148`; a dot marks the tip |

Notes:

- Works **even when disarmed** — it draws in the app window and never touches
  your OS.
- Strokes are stored as vectors, not pixels, so undo is exact and they survive
  a resolution change.
- Stroke colour is whatever the palette is when you start the stroke — hold up
  3 fingers, then point, to draw in cyan.
- Drawings persist when you leave DRAW mode, so you can annotate and then
  switch to MEDIA to present.

---

## 6. Safety & monitoring features

### 6.1 Drowsiness alarm

**What:** Eye Aspect Ratio (EAR) from the face mesh — the ratio of eyelid
opening to eye width, which collapses toward zero as your eyes close. If they
stay closed past a threshold, a pulsing red banner appears and a tone sounds.

| | |
| --- | --- |
| **Activate** | On by default |
| **Toggle** | `D` |
| **Sensitivity** | `--ear-threshold 0.18` (higher = more sensitive) |
| **Patience** | `--drowsy-seconds 1.2` (how long eyes must stay closed) |
| **Silent mode** | `--no-alarm` (banner only, no beep) |
| **Disable at start** | `--no-drowsiness` |
| **Confirm** | HUD shows `Eyes OPEN (EAR 0.31)`; close your eyes for 2 s |

A normal blink is 100–300 ms, so the 1.2 s default ignores blinking. The alarm
has a 2.5 s cooldown, and the beep runs on a background thread so it never
stalls the video.

Beep backend: `winsound` on Windows, terminal bell elsewhere. The banner always
shows regardless.

### 6.2 Performance monitoring

**What:** Rolling FPS plus per-stage timings, so you can see where time goes.

| | |
| --- | --- |
| **Activate** | Always on in the HUD |
| **Confirm** | `FPS 46 \| 21.7 ms` and `Infer 12.9 ms  Render 2.8 ms` |
| **Session summary** | Printed on exit |

---

## 7. Video I/O features

### 7.1 Process a recorded video file

**What:** Run the entire pipeline over an existing video instead of a webcam.

```bash
python main.py --camera myclip.mp4
python main.py --camera "C:\Users\chamm\Videos\test.mp4"
```

| | |
| --- | --- |
| **Confirm** | Startup prints `Source -> file myclip.mp4` and `1280x720 @ 30.0 fps, 40 frames` |
| **Ends** | Prints `Video stream ended.` and exits cleanly |

Supported: `.mp4 .avi .mov .mkv .m4v .webm .wmv .mpg .mpeg`. A missing file
gives a clear error instead of a crash. Stream URLs (e.g. `rtsp://…`) are
passed straight through to OpenCV.

File sources always read synchronously, so **every** frame is processed —
threaded capture intentionally drops stale frames, which is right for a live
camera but wrong for playback.

### 7.2 MP4 recording

**What:** Records exactly what you see, **including** the holographic panel and
HUD.

| | |
| --- | --- |
| **Start / stop** | `R` |
| **Saved to** | `output/neonvision_YYYYMMDD_HHMMSS.mp4` |
| **Confirm** | HUD shows `REC 00:12 (360f)`; toast names the file |

The encoder is sized from the actual frame being written and tries `mp4v` →
`avc1` → `XVID` until one opens. If the frame size ever changes mid-recording
the file is closed rather than silently losing content. Recording also stops
cleanly on quit.

### 7.3 PNG snapshots

| | |
| --- | --- |
| **Take one** | `S` |
| **Saved to** | `output/neonvision_YYYYMMDD_HHMMSS.png` |
| **Confirm** | Toast shows `Snapshot saved: neonvision_...png` |

### 7.4 Camera selection & capture settings

```bash
python main.py --camera 1                    # second webcam
python main.py --width 1920 --height 1080    # 1080p
python main.py --fps 60                      # request 60 fps
python main.py --no-flip                     # disable the mirrored preview
```

The preview is mirrored by default for a natural selfie view.

---

## 8. Complete keyboard reference

| Key | Action | Feature |
| --- | --- | --- |
| `Q` / `Esc` | Quit | — |
| `TAB` | Cycle interaction mode | [§2](#2-the-mode-system--read-this-first) |
| `A` | Arm / disarm all OS actions | [§2](#2-the-mode-system--read-this-first) |
| `M` | Mesh style: hybrid → tesselation → contours → off | [§3.1](#31-face-mesh-468--478-landmarks) |
| `V` | Toggle face mesh | [§3.1](#31-face-mesh-468--478-landmarks) |
| `H` | Toggle hand tracking | [§3.3](#33-hand-tracking--finger-counting) |
| `G` | Toggle neon glow | [§3.2](#32-neon-glow-effect) |
| `C` | Cycle base colour theme | [§4.5](#45-base-colour-themes) |
| `P` | Toggle gesture colour palette | [§4.4](#44-gesture-controlled-colour-palette) |
| `O` | Show / hide holographic panel | [§4.1](#41-the-holo-mesh-dot-panel-high-density-cloud) |
| `L` | Toggle breathing animation | [§4.2](#42-breathing-animation) |
| `K` | Cycle dot density: x0 → x2 → x4 → x6 | [§4.1](#41-the-holo-mesh-dot-panel-high-density-cloud) |
| `D` | Toggle drowsiness watchdog | [§6.1](#61-drowsiness-alarm) |
| `X` | Clear the air-drawing canvas | [§5.6](#56-air-drawing) |
| `Z` | Undo the last stroke | [§5.6](#56-air-drawing) |
| `B` | Toggle HUD + action log | [§4.6](#46-hud-action-log-and-effects) |
| `S` | Save a PNG snapshot | [§7.3](#73-png-snapshots) |
| `R` | Start / stop MP4 recording | [§7.2](#72-mp4-recording) |
| `F` | Toggle fullscreen | [§4.6](#46-hud-action-log-and-effects) |

Keys are case-insensitive. The window must have focus.

---

## 9. Complete gesture reference

### Recognised poses

Order is `(thumb, index, middle, ring, pinky)`.

| Pose name | Fingers up | Used for |
| --- | --- | --- |
| **Fist** | none | **Mute** (MEDIA) |
| **Open Palm** | all 5 | **Slide swipe** (MEDIA) |
| **Peace** | index + middle | **Play/pause** (MEDIA), pen up (DRAW) |
| **Pointing** | index | **Pen down** (DRAW), cursor (MOUSE) |
| **Gun** | thumb + index | **Volume dial** (MEDIA) |
| **Thumbs Up** | thumb | — |
| **Rock On** | index + pinky | — |
| **Call Me** | thumb + pinky | — |
| **Three Up** | index + middle + ring | — |
| **Four Up** | index + middle + ring + pinky | — |
| **Trio** | thumb + index + middle | — |
| **Pinky Out** | pinky | — |
| **OK** | middle + ring + pinky, thumb pinched to index | — |

Unlisted combinations report as `2 Fingers`, `3 Fingers`, etc.

### What each mode listens for

| Mode | Gesture | Result |
| --- | --- | --- |
| **any** | 1 / 2 / 3 / 4 fingers | Recolour both meshes |
| **AIR MOUSE** | Index fingertip position | Move cursor |
| **AIR MOUSE** | Pinch thumb + index | Press / release mouse button |
| **MEDIA** | Peace | Play / pause |
| **MEDIA** | Fist | Mute toggle |
| **MEDIA** | Gun, vary the gap | Set system volume |
| **MEDIA** | Open palm, sweep sideways | Left / right arrow key |
| **DRAW** | Index only | Pen down — paint |
| **DRAW** | Peace / palm / fist | Pen up |

### Tuning constants

Edit these in the source if the defaults do not suit you:

| Constant | File | Default | Controls |
| --- | --- | --- | --- |
| `PINCH_CLOSE` | `controls/gesture_actions.py` | `0.32` | Click press point |
| `PINCH_OPEN` | `controls/gesture_actions.py` | `0.46` | Click release point |
| `VOLUME_MIN_RATIO` | `controls/gesture_actions.py` | `0.22` | Gap = 0 % volume |
| `VOLUME_MAX_RATIO` | `controls/gesture_actions.py` | `1.05` | Gap = 100 % volume |
| `SWIPE_DISTANCE` | `controls/gesture_actions.py` | `0.17` | Swipe travel needed |
| `SWIPE_WINDOW` | `controls/gesture_actions.py` | `0.45` | Swipe time window (s) |
| `FINGER_RATIO` | `core/hand_tracker.py` | `1.12` | Finger-extended threshold |
| `THUMB_RATIO` | `core/hand_tracker.py` | `1.10` | Thumb-extended threshold |
| `OK_PINCH_RATIO` | `core/hand_tracker.py` | `0.42` | OK-sign pinch tightness |

---

## 10. Complete command-line reference

`python main.py --help` prints this list too.

### Capture

| Flag | Default | Effect |
| --- | --- | --- |
| `--camera` | `0` | Device index, or a path to a video file |
| `--width` | `1280` | Capture width |
| `--height` | `720` | Capture height |
| `--fps` | `30` | Requested capture frame rate |
| `--no-flip` | off | Disable the mirrored preview |
| `--no-thread` | off | Read frames synchronously |

### Tracking

| Flag | Default | Effect |
| --- | --- | --- |
| `--max-faces` | `1` | Maximum faces to track |
| `--max-hands` | `2` | Maximum hands to track |
| `--no-refine` | off | Skip iris refinement (468 instead of 478) |
| `--hand-complexity` | `1` | `0` is faster, `1` more accurate * |
| `--face-confidence` | `0.5` | Face detection/tracking threshold |
| `--hand-confidence` | `0.6` | Hand detection/tracking threshold |
| `--gesture-smoothing` | `5` | Frames in the pose majority vote |
| `--hand-idle-stride` | `2` | Sample hand inference every N frames while idle |
| `--infer-scale` | `1.0` | Run inference on a down-scaled copy (0.25–1.0) |

\* `--hand-complexity` only applies on the older MediaPipe `solutions`
backend; the newer `tasks` backend has no complexity dial. The startup banner
names your backend.

### Controls

| Flag | Default | Effect |
| --- | --- | --- |
| `--mode` | `idle` | Initial mode: `idle`, `mouse`, `media`, `draw` |
| `--dry-run` | off | Recognise gestures, send nothing to the OS |
| `--no-actions` | off | Start disarmed |
| `--ear-threshold` | `0.18` | EAR below which eyes count as closed |
| `--drowsy-seconds` | `1.2` | Eye closure before the alarm |
| `--no-drowsiness` | off | Disable the eye watchdog |
| `--no-alarm` | off | Silent alarm (banner only) |

### Appearance

| Flag | Default | Effect |
| --- | --- | --- |
| `--mesh-style` | `hybrid` | `hybrid`, `tesselation`, `contours`, `off` |
| `--node-stride` | `8` | Draw every n-th landmark as a node; `0` disables |
| `--no-glow` | off | Disable the blurred glow pass |
| `--no-palette` | off | Do not recolour from the finger count |
| `--no-holo` | off | Hide the holographic panel |
| `--holo-width` | `0.34` | Panel width as a fraction of the frame (0.15–0.6) |
| `--holo-density` | `4` | Points interpolated per mesh edge (0–12) |
| `--breathe-speed` | `1.9` | Breathing rate, radians/second |
| `--breathe-amount` | `0.07` | Peak dot-spacing swing; `0` freezes it |
| `--holo-density` | `4` | Points interpolated per mesh edge (0–12) |

### Environment variables

| Variable | Effect |
| --- | --- |
| `NEONVISION_MODEL_DIR` | Where MediaPipe `.task` bundles are cached |

---

## 11. Reading the HUD

### Left panel — HOLO MESH

```
HOLO MESH
HEAD POSE / DEG
Y -14 P  1 R  4        <- yaw / pitch / roll, degrees
LANDMARKS 478          <- 478 = iris refinement on
DOTS 5766  (x4/edge)   <- cloud size and density (K)
BREATHE [||||....]     <- live breath phase (L)
RENDER DOT CLOUD       <- confirms dots-only rendering
PALETTE 3F             <- gesture palette active, 3 fingers
```

### Right panel — main HUD

```
NEONVISION AI
FPS  46 | 21.7 ms                    <- smoothed FPS and frame time
Gesture: 3 Fingers                   <- primary finger count
Mode MEDIA / SLIDES   [ARMED]        <- mode + safety state
Faces 1   Hands 1   Edges 1511       <- what was drawn this frame
Mesh HYBRID   Theme #00CAFF Cyan     <- style and active colour
Infer  12.9 ms   Render   2.8 ms     <- where the time goes
Pose R:Peace                         <- per-hand pose name
Volume  62%  (system)                <- MEDIA mode only
Pinch 0.87                           <- AIR MOUSE mode only
PEN UP  strokes 3  points 148        <- AIR DRAWING mode only
Eyes OPEN (EAR 0.31)                 <- drowsiness watchdog
REC 00:12 (360f)                     <- while recording
```

### Safety state, at a glance

| Shown | Meaning |
| --- | --- |
| `[ARMED]` | Gestures **will** control your computer |
| `[DISARMED]` | Press `A` to enable |
| `[DRY-RUN]` | Recognising only, nothing sent |
| `[NO POINTER]` | PyAutoGUI unavailable |
| `Mode IDLE (tracking only)` | No mode selected — press `TAB` |

---

## 12. Recipes

**Learn the gestures safely**

```bash
python main.py --dry-run
```

Everything is recognised and logged; nothing reaches your OS.

**Present a slide deck hands-free**

```bash
python main.py --mode media --no-drowsiness
```

Open palm, sweep sideways to change slides. Peace to play/pause embedded video.

**Maximum frame rate**

```bash
python main.py --infer-scale 0.6 --no-holo --no-refine
```

**Best-looking capture for a demo video**

```bash
python main.py --width 1920 --height 1080 --breathe-speed 2.4 --breathe-amount 0.1
```

Then press `R` to record, `B` to hide the HUD for a clean shot.

**Late-night fatigue monitor only**

```bash
python main.py --no-holo --mesh-style contours --ear-threshold 0.20 --drowsy-seconds 1.0
```

**Analyse recorded footage**

```bash
python main.py --camera myclip.mp4 --mode idle
```

**Calm, minimal look**

```bash
python main.py --mesh-style contours --breathe-speed 1.0 --breathe-amount 0.04 --node-stride 0
```

---

## 13. Troubleshooting

**Gestures do nothing**
Check the HUD. It must show a mode other than `IDLE` (`TAB`) **and**
`[ARMED]` (`A`). `[DRY-RUN]` means you launched with `--dry-run`.

**The cursor moves when I don't want it to**
Press `A` to disarm, or `TAB` back to `IDLE`. Or shove the pointer into a
screen corner — the failsafe disarms the engine automatically.

**"Unable to open capture source 0"**
The camera is in use by another app, or blocked. On Windows check
*Settings → Privacy & security → Camera*. Try `--camera 1`.

**Volume does nothing**
The HUD names the backend. `keys` means synthesised media keys, which some
applications capture before the OS sees them. `none` means no backend is
available — on Windows, `pip install pycaw` enables absolute control.

**No beep on the drowsiness alarm**
`winsound` is Windows-only; elsewhere it falls back to the terminal bell. The
banner always shows. `--no-alarm` silences it deliberately.

**Finger count is wrong**
Keep your whole hand in frame with your palm toward the camera. Raise
`--gesture-smoothing` for steadier reporting. For persistent bias, adjust
`FINGER_RATIO` in `core/hand_tracker.py`.

**Pinch clicks too easily / not enough**
Adjust `PINCH_CLOSE` and `PINCH_OPEN` in `controls/gesture_actions.py`. They
are fractions of palm size and have deliberate hysteresis.

**Swipes don't change slides**
The pose must be a full **Open Palm**, the sweep must cover 17 % of the frame
width within 0.45 s, and the target application needs keyboard focus.

**Palette colours don't change**
Press `P` to confirm the palette is on. Only 1–4 fingers map to colours; 0 and
5 intentionally keep the base theme.

**Low frame rate**
In order of impact: `--no-holo`, `--holo-density 2`, `--infer-scale 0.6`,
`--no-glow`, `--no-refine`.

**Editor says the packages are missing**
Select `.venv` as your interpreter (`Ctrl+Shift+P` → *Python: Select
Interpreter*). The packages are installed there, not in the global Python.

---

## License

MIT © ChamathDilshanC
