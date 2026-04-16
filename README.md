# hand-cv

Real-time hand detection and gesture recognition for the Raspberry Pi 5,
using MediaPipe and OpenCV with a live camera HUD overlay.
Optionally accelerated by a Google Coral USB Accelerator (Edge TPU).

---

## Hardware Requirements

| Component | Spec |
|-----------|------|
| Board | Raspberry Pi 5 (any RAM; 4GB+ recommended) |
| Camera | Raspberry Pi Camera Module v2 (IMX219) or v3 (IMX708) |
| OS | Debian GNU/Linux 13 (Trixie), aarch64 |
| Display | HDMI monitor connected directly to the Pi (not SSH) |
| Storage | 2GB free minimum |
| Optional | Google Coral USB Accelerator |

> **Camera note:** The OV5647 (Camera Module v1) is considered legacy on
> Trixie. libcamera may have trouble starting its pipeline reliably. A
> Camera Module v2 (IMX219) or v3 (IMX708) is strongly recommended.
>
> **Display note:** OpenCV's `imshow` requires a local display session (HDMI).
> Running over SSH without X forwarding will crash with a Qt/xcb error.

---

## Why These Specific Choices Were Made

### Python 3.11 (not system Python 3.13)

Debian Trixie ships Python 3.13. **MediaPipe has no pre-built wheel for
Python 3.13 on aarch64.** Python 3.11 is the highest stable version with
published aarch64 wheels. Python 3.11.9 is therefore built from source and
installed to `/usr/local` via `make altinstall`.

### `rpicam-vid --no-raw` (Pi 5 addition)

The Pi 5 uses a new ISP called PiSP (not the VC4 used on Pi 4). By default,
`rpicam-vid` on Pi 5 opens a secondary raw stream in `BGGR_PISP_COMP1`
format alongside the YUV420 output. If there is no raw output sink, this
produces errors or stalls on some configurations. The `--no-raw` flag
suppresses this secondary stream and is harmless on Pi 4 too.

### No hardware H.264 encoder on Pi 5

Unlike Pi 4's VideoCore VI, Pi 5 has no dedicated video encode hardware.
This does not affect our pipeline because we use `--codec yuv420` raw frames
— no H.264 encoding is involved. The Pi 5's faster CPU cores more than
compensate when running MediaPipe.

### Coral USB Accelerator integration

The Coral accelerates only the **palm detection** stage of MediaPipe's hand
tracking pipeline. Here is why:

- Google published a Coral-compiled (`_edgetpu.tflite`) version of the
  **palm detector** model in their MediaPipe Coral examples repo.
- The **landmark model** (21 keypoints) was never compiled for the Edge TPU —
  it uses TFLite ops that the EdgeTPU compiler cannot partition fully.
- The palm detector is the most expensive part of the first-pass frame
  analysis. Offloading it to the Coral frees CPU cycles for landmark
  tracking and the HUD.
- `pycoral` has no Python 3.11 wheel for aarch64 on Trixie. We use
  `tflite-runtime` directly with the `libedgetpu` delegate instead.

---

## Project Structure

```
hand-cv/
├── setup.sh               # automated setup script
├── requirements.txt       # pip packages (for reference)
├── models/
│   └── palm_detection_builtin_256_integer_quant_edgetpu.tflite
├── src/
│   ├── main.py            # entry point — camera loop and display
│   ├── detector.py        # MediaPipe hand detection and gesture logic
│   ├── coral_detector.py  # Coral Edge TPU palm detection wrapper
│   ├── hud.py             # OpenCV HUD overlay drawing
│   └── config.py          # all tunable constants in one place
├── logs/                  # reserved for future logging
└── assets/                # reserved for fonts, icons etc.
```

---

## Setup

### First-time setup (automated)

```bash
cd ~/hand-cv
chmod +x setup.sh
./setup.sh
```

The script will:
1. Install system packages via `apt`
2. Install the Coral Edge TPU runtime (`libedgetpu1-std`)
3. Build Python 3.11.9 from source if not present (~15 min on Pi 5)
4. Create the venv at `~/hand-cv/venv`
5. Install pip packages including `tflite-runtime`
6. Download the EdgeTPU palm detection model
7. Verify everything

**After first-time Coral install:** unplug and replug the Coral USB
Accelerator (or reboot) before running so the udev rules take effect.

### Manual Coral runtime install (if you skipped or need to redo it)

```bash
# Add Coral repo (modern signed-by method — works on Trixie)
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
  | sudo gpg --dearmor -o /usr/share/keyrings/coral-edgetpu.gpg

echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu.gpg] \
  https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
  | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

sudo apt update
sudo apt install libedgetpu1-std   # standard clock (cooler)
# sudo apt install libedgetpu1-max # maximum clock (hotter, faster)

# Replug Coral USB after install
```

---

## Running

Always activate the venv first, then run from the project root:

```bash
source ~/hand-cv/venv/bin/activate

# CPU-only mode (default):
python src/main.py

# With Coral USB acceleration:
CORAL_ENABLE=1 python src/main.py
```

Press **`q`** or **`Esc`** to quit cleanly.

When Coral is active, a **"CORAL TPU"** label appears in the HUD top-right
corner below the FPS counter.

---

## How It Works

### Camera pipeline (Pi 5)

```
rpicam-vid subprocess (--no-raw flag required on Pi 5)
    │  raw YUV420 bytes → stdout pipe
    ▼
main.py: read_frame()
    │  np.frombuffer → reshape → cv2.cvtColor(YUV→BGR)
    ▼
BGR numpy array (640×480)
```

`rpicam-vid` streams `--codec yuv420` frames to stdout. Each frame is
exactly `width × height × 1.5` bytes of I420-packed YUV.

The `--no-raw` flag suppresses the secondary PISP_COMP1 raw stream that
Pi 5 opens by default (unused in our pipeline and can cause pipe errors).

### Detection pipeline

```
BGR frame
    │
    ├─ [If CORAL_ENABLE=1] ──────────────────────────────────────┐
    │   CoralPalmDetector.detect()                               │
    │   resizes to 256×256, runs on Edge TPU                     │
    │   returns palm bounding boxes (fast, ~2ms)                 │
    └────────────────────────────────────────────────────────────┤
    │                                                            │
    ▼ (both paths converge here)                                 │
MediaPipe Hands.process() on full BGR frame              ←───────┘
    │  returns hand_landmarks + handedness per hand
    ▼
detector.py: count_fingers()
    │  compares tip y-coord vs PIP y-coord for 4 fingers
    │  compares thumb tip x vs MCP x (mirrored for left/right)
    ▼
detector.py: detect_gesture()
    │  rule-based classifier using landmark geometry
    ▼
List of detection dicts (one per hand)
```

### Gesture detection logic

Pure geometry on 21 MediaPipe hand landmarks — no additional ML model.

| Gesture | Detection method |
|---------|-----------------|
| Open hand | All 5 fingers extended |
| Closed fist | 0 fingers extended |
| Thumbs up | Other fingers curled, thumb tip significantly above wrist |
| Thumbs down | Other fingers curled, thumb tip significantly below wrist |
| Pointing | Only index finger extended |
| Peace / V | Index and middle extended, others curled |
| N fingers | Generic fallback — reports finger count |

### HUD overlay

Each detected hand gets a colour-coded bounding box (green = right,
orange = left), a crosshair at the palm centre, and a floating label
showing handedness, gesture name, and finger count.

The global top-left panel shows total hand count and a summary per hand.
FPS is displayed top-right. A "CORAL TPU" badge appears when Coral is active.
A centred warning appears when no hand is detected.

---

## Configuration

All tunable values are in `src/config.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `CAMERA_WIDTH` | 640 | Frame width in pixels |
| `CAMERA_HEIGHT` | 480 | Frame height in pixels |
| `CAMERA_FPS` | 20 | Target frame rate |
| `MAX_HANDS` | 1 | Maximum hands to detect simultaneously |
| `DETECTION_CONFIDENCE` | 0.5 | MediaPipe initial detection threshold |
| `TRACKING_CONFIDENCE` | 0.3 | MediaPipe tracking threshold |
| `CORAL_MODEL_PATH` | `models/palm_detection_builtin_256_integer_quant_edgetpu.tflite` | Path to EdgeTPU palm model |
| `HUD_COLOR_PRIMARY` | Green | Right hand / general HUD colour |
| `HUD_COLOR_WARN` | Orange | Left hand colour |
| `HUD_COLOR_CORAL` | Cyan | Coral active indicator colour |

---

## Pip Packages

| Package | Version | Purpose |
|---------|---------|---------|
| `mediapipe` | 0.10.18 | Hand landmark detection |
| `opencv-python` | 4.x | Frame processing and display |
| `tflite-runtime` | latest | TFLite inference + EdgeTPU delegate |
| `cvzone` | 1.6.1 | Utility wrappers |
| `imutils` | latest | Image utility helpers |
| `numpy` | <2 | Array operations (pinned for mediapipe compat) |

---

## Troubleshooting

### Camera not detected on Pi 5 / Trixie

```bash
rpicam-hello --list-cameras
```

If no camera is listed, check:
- Ribbon cable seated properly at both ends
- Correct connector used (Pi 5 has two CSI ports labelled CAM0 / CAM1)
- `sudo dmesg | grep -i imx` for driver errors

For Camera Module v1 (OV5647), you may need to add to `/boot/firmware/config.txt`:
```
dtoverlay=ov5647
```
Note: OV5647 support is unreliable on Trixie — upgrade to v2 or v3 if possible.

### Coral not detected

```bash
lsusb | grep -i google
# Should show: ID 18d1:9302 Google Inc.
```

If it shows `1a6e:089a` instead, the firmware hasn't loaded yet — plug it
into a USB 3.0 port (blue) and run any inference once to trigger the firmware
upload, then check `lsusb` again.

```bash
# Check libedgetpu is installed
dpkg -l libedgetpu1-std
# Check the library is found
python3 -c "import ctypes.util; print(ctypes.util.find_library('edgetpu'))"
```

### rpicam-vid errors on Pi 5

If you see errors about `PISP_COMP1` or the process exits immediately, make
sure `--no-raw` is present in the camera command (it is in `main.py`).

### numpy version warning

`mediapipe` declares `numpy<2` but works with numpy 2.x at runtime. The pip
dependency warning can be ignored.

### TFLite warnings on startup

`inference_feedback_manager` and `cpuinfo` messages on stderr are harmless
MediaPipe internals. They are suppressed via `GLOG_minloglevel=3` in `main.py`.
