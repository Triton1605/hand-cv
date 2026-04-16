import os

# ─── Camera ───────────────────────────────────────────────────────────────────
CAMERA_WIDTH    = 640
CAMERA_HEIGHT   = 480
CAMERA_FPS      = 20

# ─── MediaPipe ────────────────────────────────────────────────────────────────
MAX_HANDS               = 4     # max hands detectable; more = slower
DETECTION_CONFIDENCE    = 0.5   # higher = stricter first-frame detection
TRACKING_CONFIDENCE     = 0.3   # higher = stricter frame-to-frame tracking

# ─── Coral Edge TPU ───────────────────────────────────────────────────────────
# Path to the EdgeTPU-compiled palm detection model.
# Set CORAL_ENABLE=1 in the environment to activate Coral acceleration.
# If the file is missing or the Coral is unplugged, the system falls back
# gracefully to CPU-only MediaPipe.
_SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT  = os.path.dirname(_SRC_DIR)  # hand-cv/ root

CORAL_MODEL_PATH = os.path.join(
    _PROJ_ROOT,
    "models",
    "palm_detection_builtin_256_integer_quant_edgetpu.tflite",
)

# ─── HUD ──────────────────────────────────────────────────────────────────────
HUD_FONT            = 0              # cv2.FONT_HERSHEY_SIMPLEX
HUD_FONT_SCALE      = 0.4
HUD_THICKNESS       = 1
HUD_COLOR_PRIMARY   = (0, 255, 0)   # green — right hand / general HUD
HUD_COLOR_WARN      = (0, 165, 255) # orange — left hand
HUD_COLOR_CORAL     = (255, 200, 0) # cyan-ish — Coral active indicator
HUD_COLOR_WHITE     = (255, 255, 255)
HUD_COLOR_BLACK     = (0, 0, 0)
HUD_BOX_ALPHA       = 0.4           # translucency of info panels

# ─── Gesture thresholds ───────────────────────────────────────────────────────
THUMB_UP_ANGLE_THRESH    = 160  # degrees — thumb extended upward
THUMB_DOWN_ANGLE_THRESH  = 20   # degrees — thumb pointing downward
OPEN_HAND_FINGER_COUNT   = 5
CLOSED_HAND_FINGER_COUNT = 0
