#!/usr/bin/env bash
# =============================================================================
# hand-cv setup script
# Raspberry Pi 5 — Debian Trixie (aarch64) — Python 3.11 venv
# Includes: Google Coral USB Accelerator (Edge TPU) support
# =============================================================================
set -e

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON311="/usr/local/bin/python3.11"
VENV_DIR="$PROJ_DIR/venv"
MODELS_DIR="$PROJ_DIR/models"

echo "========================================"
echo "  hand-cv setup (Pi 5 + Coral)"
echo "========================================"

# ── 1. Check we're on a Pi ────────────────────────────────────────────────────
if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
  echo "WARNING: This does not appear to be a Raspberry Pi. Continuing anyway."
fi

# ── 2. System dependencies ────────────────────────────────────────────────────
echo ""
echo "[1/7] Installing system dependencies..."
sudo apt update -qq
sudo apt install -y \
  build-essential \
  libssl-dev \
  zlib1g-dev \
  libncurses5-dev \
  libreadline-dev \
  libffi-dev \
  libgdbm-dev \
  libbz2-dev \
  libsqlite3-dev \
  liblzma-dev \
  libcap-dev \
  rpicam-apps \
  curl \
  wget \
  libusb-1.0-0 \
  gnupg

# ── 3. Coral Edge TPU runtime ─────────────────────────────────────────────────
# The Coral packages repo targets Debian/Ubuntu. On Trixie the key import
# needs the modern signed-by method instead of the legacy apt-key approach.
echo ""
echo "[2/7] Installing Coral Edge TPU runtime..."

CORAL_LIST="/etc/apt/sources.list.d/coral-edgetpu.list"
CORAL_KEY="/usr/share/keyrings/coral-edgetpu.gpg"

if [ ! -f "$CORAL_KEY" ]; then
  echo "  Adding Coral apt repository key..."
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | sudo gpg --dearmor -o "$CORAL_KEY"
fi

if [ ! -f "$CORAL_LIST" ]; then
  echo "  Adding Coral apt repository..."
  echo "deb [signed-by=$CORAL_KEY] https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
    | sudo tee "$CORAL_LIST" > /dev/null
fi

sudo apt update -qq

# libedgetpu1-std  = standard clock (cooler, recommended for USB without fan)
# libedgetpu1-max  = maximum clock (faster, runs hotter)
# Change to libedgetpu1-max if you have active cooling or don't mind the heat.
if ! dpkg -l libedgetpu1-std 2>/dev/null | grep -q '^ii'; then
  echo "  Installing libedgetpu1-std (standard clock frequency)..."
  sudo apt install -y libedgetpu1-std
  echo ""
  echo "  *** ACTION REQUIRED ***"
  echo "  Unplug and replug the Coral USB Accelerator so the new udev rules take effect."
  echo "  (Or reboot — the script will continue, but the Coral won't be usable until then.)"
  echo ""
fi

# ── 4. Python 3.11 (build from source if not present) ────────────────────────
echo ""
echo "[3/7] Checking for Python 3.11..."
if [ ! -f "$PYTHON311" ]; then
  echo "Python 3.11 not found — building from source (this takes ~15-20 min on Pi 5)..."
  # Extra deps for optimised build
  sudo apt install -y \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libncurses5-dev libgdbm-dev liblzma-dev libffi-dev libmpdec-dev \
    libuuid1 uuid-dev tk-dev
  cd /tmp
  PY_VER="3.11.9"
  wget -q "https://www.python.org/ftp/python/${PY_VER}/Python-${PY_VER}.tgz"
  tar -xf "Python-${PY_VER}.tgz"
  cd "Python-${PY_VER}"
  # Pi 5 has 4 performance cores — use all of them for the build
  ./configure --enable-optimizations --with-lto --prefix=/usr/local
  make -j4
  sudo make altinstall
  cd "$PROJ_DIR"
  echo "Python 3.11.9 built and installed."
else
  echo "Python 3.11 found: $($PYTHON311 --version)"
fi

# ── 5. Create venv ────────────────────────────────────────────────────────────
echo ""
echo "[4/7] Creating virtual environment with Python 3.11..."
if [ -d "$VENV_DIR" ]; then
  echo "Existing venv found — removing and recreating..."
  rm -rf "$VENV_DIR"
fi
"$PYTHON311" -m venv "$VENV_DIR" --system-site-packages
source "$VENV_DIR/bin/activate"
echo "Active Python: $(python --version)"

# ── 6. Install Python packages ────────────────────────────────────────────────
echo ""
echo "[5/7] Installing Python packages..."
pip install --upgrade pip --quiet

# Core vision stack
pip install mediapipe opencv-python cvzone imutils --quiet

# tflite-runtime: used for running EdgeTPU models via the Coral delegate.
# pycoral has no wheel for Python 3.11 / aarch64 — we use tflite-runtime
# directly and load the libedgetpu delegate manually instead.
pip install tflite-runtime --quiet

echo "Packages installed."

# ── 7. Download EdgeTPU hand detection model ──────────────────────────────────
# Google's MediaPipe Coral example ships a palm-detector _edgetpu.tflite model.
# This is the ONLY MediaPipe hand model officially compiled for the Edge TPU.
# The landmark model is NOT available in EdgeTPU format — landmarks still run
# on CPU via MediaPipe as normal. The Coral accelerates the palm detection
# stage only, which is the heaviest part of the first-pass detection.
echo ""
echo "[6/7] Downloading Coral-compatible palm detection model..."
mkdir -p "$MODELS_DIR"

PALM_MODEL="$MODELS_DIR/palm_detection_builtin_256_integer_quant_edgetpu.tflite"
PALM_MODEL_URL="https://github.com/google-ai-edge/mediapipe/raw/master/mediapipe/examples/coral/models/palm-detector-quantized_edgetpu.tflite"

if [ ! -f "$PALM_MODEL" ]; then
  if wget -q -O "$PALM_MODEL" "$PALM_MODEL_URL"; then
    echo "  Palm detection EdgeTPU model downloaded."
  else
    echo "  WARNING: Could not download palm detection model."
    echo "  The system will still work — just using CPU-only MediaPipe."
    echo "  You can download it later from:"
    echo "  $PALM_MODEL_URL"
    rm -f "$PALM_MODEL"  # remove empty file if wget failed
  fi
else
  echo "  Palm detection model already present."
fi

# ── 8. Verify ─────────────────────────────────────────────────────────────────
echo ""
echo "[7/7] Verifying installation..."
python - <<'PYEOF'
import sys
ok = True

try:
    import cv2
    print(f"  opencv        {cv2.__version__}   OK")
except ImportError as e:
    print(f"  opencv        FAILED: {e}"); ok = False

try:
    import mediapipe as mp
    print(f"  mediapipe     {mp.__version__}   OK")
except ImportError as e:
    print(f"  mediapipe     FAILED: {e}"); ok = False

try:
    import cvzone
    print(f"  cvzone        OK")
except ImportError as e:
    print(f"  cvzone        FAILED: {e}"); ok = False

try:
    import numpy as np
    print(f"  numpy         {np.__version__}   OK")
except ImportError as e:
    print(f"  numpy         FAILED: {e}"); ok = False

try:
    import tflite_runtime.interpreter as tflite
    print(f"  tflite-runtime   OK")
except ImportError as e:
    print(f"  tflite-runtime   FAILED: {e}"); ok = False

# Check libedgetpu is present on the system
import ctypes, ctypes.util
lib = ctypes.util.find_library("edgetpu")
if lib:
    print(f"  libedgetpu    {lib}   OK")
else:
    print(f"  libedgetpu    NOT FOUND — Coral runtime may not be installed")
    ok = False

import subprocess
result = subprocess.run(["which", "rpicam-vid"], capture_output=True, text=True)
if result.returncode == 0:
    print(f"  rpicam-vid    {result.stdout.strip()}   OK")
else:
    print(f"  rpicam-vid    FAILED — not found in PATH"); ok = False

# Check Pi 5 vs Pi 4 — just informational
try:
    with open("/proc/cpuinfo") as f:
        cpuinfo = f.read()
    if "Raspberry Pi 5" in cpuinfo or "BCM2712" in cpuinfo:
        print(f"  board         Raspberry Pi 5   OK")
    elif "Raspberry Pi 4" in cpuinfo or "BCM2711" in cpuinfo:
        print(f"  board         Raspberry Pi 4 (unexpected)")
    else:
        print(f"  board         Unknown Pi variant")
except Exception:
    pass

if ok:
    print("\n  All checks passed. Ready to run.")
else:
    print("\n  Some checks failed. Review errors above.")
    sys.exit(1)
PYEOF

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  Setup complete!"
echo ""
echo "  If you just installed libedgetpu for the first time:"
echo "    sudo reboot   (or unplug/replug the Coral USB Accelerator)"
echo ""
echo "  To run:"
echo "    source venv/bin/activate"
echo "    python src/main.py"
echo ""
echo "  To run with Coral acceleration enabled:"
echo "    CORAL_ENABLE=1 python src/main.py"
echo "========================================"
