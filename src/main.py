#!/usr/bin/env python3
"""
Hand CV — main entry point.

Camera frames are read from rpicam-vid piped to stdout as raw YUV420,
converted to BGR by OpenCV, then processed by MediaPipe hand detection.
The result is displayed with a HUD overlay in an OpenCV window.

Pi 5 notes:
  - Pi 5 uses the PiSP ISP (not VC4). The --codec yuv420 stdout pipe still
    works, but the --no-raw flag is required to suppress the secondary
    PISP_COMP1 raw stream that Pi 5 opens by default (it causes errors without
    a raw output sink).
  - No hardware H.264 encoder on Pi 5. YUV420 passthrough is unaffected.

Coral USB Accelerator:
  - Set env var CORAL_ENABLE=1 to activate Edge TPU palm detection.
  - Requires libedgetpu1-std installed and Coral USB plugged in.
  - Falls back to CPU-only MediaPipe silently if Coral is unavailable.

Run:
    source ~/hand-cv/venv/bin/activate
    python src/main.py

    # With Coral acceleration:
    CORAL_ENABLE=1 python src/main.py

Quit: press  q  or  Esc
"""

import subprocess
import sys
import time
import os
import signal
import numpy as np
import cv2

# ── Make sure src/ is on the path when run from project root ─────────────────
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, CAMERA_ROTATION, CAMERA_MIRROR,
    CORAL_MODEL_PATH, FAN_SPEED, FAN_SYSFS, PIPELINE_MODE,
    MAX_HANDS,
)
from detector import HandDetector
from hud import draw_hand, draw_global_hud
from coral_detector import CoralPalmDetector  # legacy Coral helper (unused in coral mode)


# ── Suppress noisy TFLite / absl stderr spam ─────────────────────────────────
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# ── Fan control ───────────────────────────────────────────────────────────────
_fan_original_state = None

def fan_set(speed: int):
    """Write a fan speed (0-4) to the sysfs interface via sudo."""
    try:
        subprocess.run(
            ["sudo", "sh", "-c", f"echo {speed} > {FAN_SYSFS}"],
            check=True, capture_output=True,
        )
    except Exception as e:
        print(f"[hand-cv] WARNING: could not set fan speed: {e}")

def fan_read() -> int:
    """Read the current fan state from sysfs."""
    try:
        with open(FAN_SYSFS) as f:
            return int(f.read().strip())
    except Exception:
        return 0

def fan_start():
    """Override fan speed if FAN_SPEED > 0, saving the current state."""
    global _fan_original_state
    if FAN_SPEED == 0:
        return
    _fan_original_state = fan_read()
    print(f"[hand-cv] Fan: setting speed {FAN_SPEED}/4 (was {_fan_original_state})")
    fan_set(FAN_SPEED)

def fan_stop():
    """Restore the fan to its original state on exit."""
    if FAN_SPEED == 0 or _fan_original_state is None:
        return
    print(f"[hand-cv] Fan: restoring speed to {_fan_original_state}")
    fan_set(_fan_original_state)

# ── Temperature ───────────────────────────────────────────────────────────────
TEMP_SYSFS = "/sys/class/thermal/thermal_zone0/temp"

def read_temp() -> float | None:
    """Read CPU temperature in Celsius. Returns None on failure."""
    try:
        with open(TEMP_SYSFS) as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return None


_ROTATION_MAP = {
    0:   None,
    90:  cv2.ROTATE_90_COUNTERCLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_CLOCKWISE,
}

def apply_rotation(frame):
    """Rotate and/or mirror frame to correct for physical camera orientation."""
    code = _ROTATION_MAP.get(CAMERA_ROTATION)
    if code is not None:
        frame = cv2.rotate(frame, code)
    if CAMERA_MIRROR:
        frame = cv2.flip(frame, 1)
    return frame


def build_camera_cmd():
    return [
        "rpicam-vid",
        "--width",        str(CAMERA_WIDTH),
        "--height",       str(CAMERA_HEIGHT),
        "--framerate",    str(CAMERA_FPS),
        "--codec",        "yuv420",
        "--timeout",      "0",          # run indefinitely
        "--nopreview",
        "--no-raw",                     # Pi 5: suppress the PISP_COMP1 raw stream
        "--buffer-count", "2",          # reduce pipeline latency
        "-o", "-",                      # pipe raw YUV420 frames to stdout
    ]


def read_frame(pipe, width, height):
    """
    Read one YUV420 (I420) frame from the pipe.
    Frame size = width * height * 3 // 2 bytes.
    Returns a BGR numpy array or None on EOF/error.
    """
    frame_size = width * height * 3 // 2
    raw = b""
    while len(raw) < frame_size:
        chunk = pipe.read(frame_size - len(raw))
        if not chunk:
            return None
        raw += chunk
    yuv = np.frombuffer(raw, dtype=np.uint8).reshape((height * 3 // 2, width))
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
    return bgr


def main():
    # ── Fan ──────────────────────────────────────────────────────────────────
    fan_start()

    # ── Pipeline selection ────────────────────────────────────────────────────
    pipeline  = None
    coral_active = False

    if PIPELINE_MODE == "coral":
        try:
            from coral_pipeline import CoralHandPipeline
            pipeline = CoralHandPipeline(max_hands=MAX_HANDS)
            coral_active = True
            print("[hand-cv] Pipeline: Coral (palm on TPU + landmarks on CPU)")
        except Exception as e:
            print(f"[hand-cv] WARNING: Coral pipeline failed to load: {e}")
            print("[hand-cv] Falling back to MediaPipe CPU pipeline.")

    if pipeline is None:
        pipeline = HandDetector()
        print("[hand-cv] Pipeline: MediaPipe CPU-only")

    # ── Camera subprocess ─────────────────────────────────────────────────────
    cmd = build_camera_cmd()
    print(f"[hand-cv] Starting camera: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # Give the camera a moment to initialise
    # Pi 5's PiSP pipeline can take slightly longer than Pi 4's VC4 pipeline.
    time.sleep(1.5)

    if proc.poll() is not None:
        print("[hand-cv] ERROR: rpicam-vid failed to start.")
        print("[hand-cv] Check: camera ribbon connected, rpicam-apps installed,")
        print("[hand-cv]        'rpicam-hello --list-cameras' shows a camera.")
        sys.exit(1)

    print(f"[hand-cv] Camera running at {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS}fps")
    if CAMERA_ROTATION != 0:
        print(f"[hand-cv] Camera rotation: {CAMERA_ROTATION}°")
    print("[hand-cv] Press  q  or  Esc  to quit.")

    cv2.namedWindow("Hand CV", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hand CV", CAMERA_WIDTH, CAMERA_HEIGHT)

    fps_counter = 0
    fps_display = 0.0
    fps_timer   = time.time()

    det_times   = []          # rolling window of detection durations (seconds)
    DET_WINDOW  = 30          # number of frames to average over
    det_ms_avg  = 0.0

    try:
        while True:
            # Drain stale buffered frames, keep only the freshest
            for _ in range(2):
                frame = read_frame(proc.stdout, CAMERA_WIDTH, CAMERA_HEIGHT)
                if frame is None:
                    break

            if frame is None:
                print("[hand-cv] Camera stream ended.")
                break

            # ── Orientation correction ───────────────────────────────────────
            frame = apply_rotation(frame)

            # ── Detection (timed) ────────────────────────────────────────────
            det_start  = time.perf_counter()
            detections = pipeline.process(frame)
            det_end    = time.perf_counter()

            det_times.append(det_end - det_start)
            if len(det_times) > DET_WINDOW:
                det_times.pop(0)
            det_ms_avg = (sum(det_times) / len(det_times)) * 1000

            # ── Draw landmarks on frame ──────────────────────────────────────
            for d in detections:
                pipeline.draw_landmarks(frame, d['landmarks'])

            # ── Draw per-hand HUD elements ───────────────────────────────────
            for i, d in enumerate(detections):
                draw_hand(frame, d, i)

            # ── FPS calculation ──────────────────────────────────────────────
            fps_counter += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps_display = fps_counter / elapsed
                fps_counter = 0
                fps_timer   = time.time()

            # ── Global HUD ───────────────────────────────────────────────────
            draw_global_hud(frame, detections, fps_display,
                            coral_active=coral_active,
                            temp_c=read_temp(),
                            det_ms=det_ms_avg)

            # ── Show ─────────────────────────────────────────────────────────
            cv2.imshow("Hand CV", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):   # q or Esc
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("[hand-cv] Shutting down…")
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=3)
        pipeline.close()
        cv2.destroyAllWindows()
        fan_stop()
        print("[hand-cv] Done.")


if __name__ == "__main__":
    main()
