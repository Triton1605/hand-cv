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

from config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, CORAL_MODEL_PATH
from detector import HandDetector
from hud import draw_hand, draw_global_hud
from coral_detector import CoralPalmDetector  # graceful no-op if unavailable


# ── Suppress noisy TFLite / absl stderr spam ─────────────────────────────────
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def build_camera_cmd():
    return [
        "rpicam-vid",
        "--width",     str(CAMERA_WIDTH),
        "--height",    str(CAMERA_HEIGHT),
        "--framerate", str(CAMERA_FPS),
        "--codec",     "yuv420",
        "--timeout",   "0",          # run indefinitely
        "--nopreview",
        "--no-raw",                  # Pi 5: suppress the PISP_COMP1 raw stream
	"--buffer-count", "2",
        "-o", "-",                   # pipe raw YUV420 frames to stdout
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
    # ── Coral setup ──────────────────────────────────────────────────────────
    coral_enabled = os.environ.get("CORAL_ENABLE", "0") == "1"
    coral = CoralPalmDetector(CORAL_MODEL_PATH) if coral_enabled else None

    if coral_enabled:
        if coral.available:
            print("[hand-cv] Coral Edge TPU palm detection: ENABLED")
        else:
            print("[hand-cv] WARNING: Coral requested but not available — "
                  "falling back to CPU-only mode.")
            coral = None

    # ── MediaPipe detector ────────────────────────────────────────────────────
    detector = HandDetector(coral_detector=coral)

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
    print("[hand-cv] Press  q  or  Esc  to quit.")

    cv2.namedWindow("Hand CV", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hand CV", CAMERA_WIDTH, CAMERA_HEIGHT)

    fps_counter = 0
    fps_display = 0.0
    fps_timer   = time.time()

    try:
        while True:
            #drain stale buffered frames, keep only the freshest
            for _ in range(2):
                frame = read_frame(proc.stdout, CAMERA_WIDTH, CAMERA_HEIGHT)
                if frame is None:
                    break
            
            if frame is None:
                print("[hand-cv] Camera stream ended.")
                break

            # ── Detection ────────────────────────────────────────────────────
            detections = detector.process(frame)

            # ── Draw landmarks on frame ──────────────────────────────────────
            for d in detections:
                detector.draw_landmarks(frame, d['landmarks'])

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
            coral_active = coral is not None and coral.available
            draw_global_hud(frame, detections, fps_display, coral_active=coral_active)

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
        detector.close()
        if coral is not None:
            coral.close()
        cv2.destroyAllWindows()
        print("[hand-cv] Done.")


if __name__ == "__main__":
    main()
