# NeonVision AI - Technology Stack

This document details the technologies, frameworks, libraries, and developer tools powering **NeonVision AI**, along with the rationale behind each technical choice.

---

## 1. Programming Language & Runtime

| Component | Technology | Version | Purpose |
|---|---|---|---|
| **Language** | Python | `3.10+` | Primary language for computer vision pipelines, offering rich library support and fast iteration. |
| **Virtual Env** | `venv` / `conda` | Default | Dependency isolation across development and deployment environments. |

---

## 2. Core Computer Vision & ML Libraries

### **Google MediaPipe**
* **Role:** Machine Learning Inference Engine (Face Mesh & Hand Tracking).
* **Package:** `mediapipe`
* **Why MediaPipe?**
  * **Ultra-low Latency:** Optimized for real-time mobile and desktop CPU inference without requiring heavy discrete GPUs.
  * **High Landmark Density:** Generates 468 3D facial landmarks and 21 hand landmarks simultaneously with minimal overhead.
  * **Built-in Pipeline Smoothing:** Features built-in filters to reduce jitter between consecutive frames.

### **OpenCV (Open Source Computer Vision)**
* **Role:** Video Capture, Image Processing & Rendering.
* **Package:** `opencv-python`
* **Why OpenCV?**
  * **Direct Hardware Access:** Provides reliable, low-overhead webcam capture and hardware control via `cv2.VideoCapture`.
  * **Graphics & Geometry Overlay:** Efficient primitives for drawing lines, circles, contours, and text directly onto image buffers.
  * **Color Space Transformations:** Fast frame flipping and BGR $\leftrightarrow$ RGB color conversion.

### **NumPy**
* **Role:** Vectorized Mathematical Computations.
* **Package:** `numpy`
* **Why NumPy?**
  * **Coordinate Transformation:** Converts normalized coordinates ($[0.0, 1.0]$) into screen-pixel matrix coordinates efficiently using vectorized operations.
  * **Array Manipulation:** OpenCV frames are fundamentally NumPy arrays; direct slicing and masking operations allow efficient image filtering and effects.

---

## 3. Recommended Utility Libraries

| Library | Package | Purpose |
|---|---|---|
| **PyAutoGUI** *(Optional)* | `pyautogui` | Emulating mouse clicks or keystrokes mapped to specific hand gestures. |
| **Pygame / Sounddevice** *(Optional)* | `sounddevice` | Triggering futuristic sound effects when gestures or face scans activate. |

---

## 4. Development & Code Quality Tools

* **Code Formatter:** `Black` (Strict PEP 8 compliance)
* **Linter:** `Flake8` (Static code analysis)
* **Version Control:** `Git` + `GitHub`
* **IDE/Editor:** VS Code with Python & Pylance extensions

---

## 5. Hardware & Platform Compatibility

* **OS Support:** Windows 10/11, macOS (Apple Silicon & Intel), Linux (Ubuntu 20.04+)
* **Camera Requirement:** Standard USB Webcam / Integrated Laptop Camera (720p @ 30 FPS minimum recommended)
* **Compute Footprint:** CPU-only real-time performance (~30-60 FPS on modern multi-core processors without requiring CUDA/GPU acceleration).

---

## 6. Dependency Snapshot (`requirements.txt`)

```text
opencv-python>=4.8.0.76
mediapipe>=0.10.0
numpy>=1.24.0