# 🚗 Driver Drowsiness Detection System

A real-time driver drowsiness detection system using **Eye Aspect Ratio (EAR)** computed from facial landmarks. When sustained eye closure is detected, the system triggers a visual alert and audio alarm — running entirely on CPU with a standard webcam.

> Built as a Computer Vision project using Python, OpenCV, and dlib.

---

## Demo

```
[webcam frame]
┌─────────────────────────────────────────┐
│ EAR: 0.081  [██░░░░░░░░░] ← threshold  │  ← HUD top bar
│ Blinks: 3   Closed: 21f                 │
│                                         │
│         [driver face]                   │
│         (red eye contours)              │
│                                         │
│  ⚠  WAKE UP!                           │  ← flashing alert banner
│  DROWSINESS DETECTED                    │
└─────────────────────────────────────────┘
```

---

## How It Works

The system computes **EAR (Eye Aspect Ratio)** every frame using 6 facial landmark points per eye:

```
      p2  p3
  p1          p4
      p6  p5

EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)
```

| Eye state  | Typical EAR |
|------------|-------------|
| Wide open  | ~0.30       |
| Blinking   | ~0.15 (2–4 frames) |
| **Drowsy** | **< 0.25 for 20+ frames** |

When EAR stays below the threshold for **20 consecutive frames** (~0.67 s at 30 fps), the system enters **DROWSY** state and triggers the alarm.

---

## Project Structure

```
mvip/
├── drowsiness_detection.py          # Main script
├── shape_predictor_68_face_landmarks.dat  # dlib model (download separately)
├── requirements.txt                 # Python dependencies
├── README.md
└── docs/
    ├── report.pdf                   # Project report
    └── presentation.pptx            # Slide deck
```

---

## Setup

### 1. System dependencies (Fedora)

```bash
sudo dnf install -y python3-devel cmake gcc g++ \
    blas-devel lapack-devel boost-devel \
    libv4l-devel sox
```

### 2. Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Or with `uv`:

```bash
uv venv
source .venv/bin/activate
uv add opencv-python dlib imutils scipy numpy
```

> **Note:** dlib compiles from source and takes 3–5 minutes. Ensure `cmake` and `gcc` are installed first.

### 3. Download the landmark model

```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

Place `shape_predictor_68_face_landmarks.dat` in the project root.

### 4. Run

```bash
python drowsiness_detection.py
```

Press **Q** to quit.

---

## Configuration

Three parameters at the top of `drowsiness_detection.py` control sensitivity:

| Parameter | Default | Effect |
|---|---|---|
| `EAR_THRESHOLD` | `0.25` | EAR below this = eye closed. Lower for narrow eyes. |
| `EAR_CONSEC_FRAMES` | `20` | Frames eye must stay closed to trigger alert. Raise to reduce false alarms. |
| `BLINK_FRAMES` | `4` | Closures shorter than this are ignored as normal blinks. |

**Calibration tip:** Run the script and watch the EAR readout in the HUD while opening and closing your eyes naturally. Set `EAR_THRESHOLD` to ~75% of your wide-open EAR value.

---

## Requirements

```
opencv-python
dlib
imutils
scipy
numpy
```

System: `sox` (for audio alarm via ALSA)

---

## Tech Stack

- **OpenCV** — video capture, frame processing, HUD rendering
- **dlib** — HOG face detection + 68-point facial landmark prediction
- **imutils** — threaded VideoStream, landmark utilities
- **scipy** — Euclidean distance for EAR computation
- **sox** — real-time sine wave alarm synthesis via ALSA (no audio file needed)

---

## Limitations

- Does not work with opaque sunglasses (landmarks are occluded)
- HOG face detector struggles in low light — use good lighting or an IR camera
- Designed for frontal face view; extreme head angles may miss detection

---

## Future Work

- [ ] Replace dlib with **MediaPipe FaceMesh** for better accuracy and pose tolerance
- [ ] Add **yawn detection** via Mouth Aspect Ratio (MAR)
- [ ] **Head pose estimation** to detect nodding microsleep
- [ ] EAR smoothing with a short rolling average

---

## References

- Soukupová & Čech (2016) — *Real-Time Eye Blink Detection using Facial Landmarks*, CVWW
- King (2009) — *Dlib-ml: A Machine Learning Toolkit*, JMLR
- NHTSA — *Drowsy Driving Facts*, U.S. Department of Transportation

---

## License

MIT
