# NeonVision AI - System Architecture

## 1. Overview
**NeonVision AI** is a real-time computer vision application built with Python. It leverages advanced machine learning models to perform highly accurate face mesh generation and hand gesture recognition. The application renders a futuristic "Neon/Holographic" digital mesh over the user's face and tracks hand movements via a standard webcam.

## 2. Technology Stack
* **Language:** Python 3.9+
* **Core Libraries:**
  * `opencv-python` (cv2) - For video capture, image processing, and rendering graphics.
  * `mediapipe` - Google's ML framework for Face Mesh and Hand Tracking.
  * `numpy` - For matrix operations and coordinate transformations.

## 3. High-Level Architecture
The system follows a modular pipeline architecture, processing video frames sequentially.

### Modules:
1. **Camera Interface (`camera.py`):** 
   Handles initializing the webcam, capturing frames, and releasing resources.
2. **Vision Tracker (`tracker.py`):**
   Single front-end over both pipelines. Owns inference down-scaling and
   idle-frame sampling for hands, and returns one `TrackingFrame` per frame.
3. **Face Mesh Tracker (`face_mesh.py`):** 
   Wraps the MediaPipe Face Mesh solution. Extracts 468 3D facial landmarks from a given frame.
4. **Hand Tracker (`hand_tracker.py`):** 
   Wraps the MediaPipe Hands solution. Extracts 21 hand landmarks and recognizes basic gestures (e.g., counting fingers).
5. **MediaPipe Runtime (`mp_runtime.py`):**
   Backend layer that adapts to whichever MediaPipe Python API is installed
   (the classic `solutions` graphs, or the newer `tasks` landmarkers used by
   Python 3.13 wheels), and caches the `.task` model bundles the latter needs.
6. **Rendering Engine (`renderer.py`):** 
   Takes the original frame and the detected landmarks, applying custom OpenCV drawing functions (circles, lines, glowing neon effects) to create the UI overlay.
7. **Gesture Actions (`controls/gesture_actions.py`):**
   Mode-based interaction layer: air mouse and pinch clicks, media keys,
   slide swipes, system volume, and the Eye-Aspect-Ratio drowsiness watchdog.
8. **Air Canvas (`controls/air_canvas.py`):**
   Captures fingertip strokes as vector geometry for the renderer to paint.
9. **Video I/O (`utils/video_io.py`):**
   Capture-source resolution, MP4 recording and PNG snapshots.
10. **Main Application (`main.py`):** 
   The entry point of the application. Orchestrates the flow between the camera interface, trackers, and renderer.

## 4. Data Flow (Pipeline)
1. **Input:** `cv2.VideoCapture` reads a raw BGR frame from the webcam.
2. **Preprocessing:** The frame is horizontally flipped (for a mirror effect) and converted from BGR to RGB (required by MediaPipe).
3. **Inference (ML Processing):**
   * The RGB frame is passed to the **Face Mesh Tracker**.
   * The RGB frame is passed to the **Hand Tracker**.
4. **Post-processing & Coordinates:** MediaPipe returns normalized coordinates (0.0 to 1.0). These are multiplied by the frame's width and height to get pixel coordinates.
5. **Rendering:** 
   * Draw the custom glowing neon lines connecting facial landmarks.
   * Render the side "HOLO MESH" panel as a breathing high-density dot
     cloud: points interpolated along every mesh edge (~5 800 dots), drawn
     by vectorised NumPy scatter, with sine-driven spacing and glow
     intensity and no rotation.
   * Draw hand landmarks and display the gesture output (e.g., "3 Fingers") as text on the screen.
   * Overlay FPS (Frames Per Second) for performance monitoring.
6. **Output:** `cv2.imshow` displays the final processed frame to the user.

## 5. Directory Structure
```text
neonvision-ai/
│
├── main.py                 # Application entry point
├── core/
│   ├── __init__.py
│   ├── camera.py           # Webcam handling
│   ├── tracker.py          # Unified face + hand front-end
│   ├── face_mesh.py        # MediaPipe face mesh logic
│   ├── hand_tracker.py     # MediaPipe hand tracking logic
│   └── mp_runtime.py       # MediaPipe backend adapter + model cache
│
├── controls/
│   ├── __init__.py
│   ├── gesture_actions.py  # Cursor, media, slides, volume, EAR alarm
│   └── air_canvas.py       # Neon air-drawing strokes
│
├── utils/
│   ├── __init__.py
│   ├── renderer.py         # Neon effects, palette, breathing dot hologram
│   ├── video_io.py         # MP4 recording, snapshots, source resolution
│   └── fps_counter.py      # FPS calculation utility
│
├── models/                 # Tasks model bundles (auto-downloaded)
├── output/                 # Snapshots and recordings
├── requirements.txt        # Python dependencies
├── architecture.md         # Architecture documentation
└── README.md               # Project setup and running instructions