import os

# ─── Camera ───────────────────────────────────────────────────────────────────
CAMERA_WIDTH    = 384 #640
CAMERA_HEIGHT   = 288 #480
CAMERA_FPS      = 20

# Camera orientation in degrees. Rotates the captured frame before processing.
#   0   = normal (no rotation)
#   90  = camera rotated 90° clockwise   → correct with 90° counter-clockwise
#   180 = camera upside down             → correct with 180° rotation
#   270 = camera rotated 90° anti-clock  → correct with 90° clockwise
CAMERA_ROTATION = 0 #180

# Mirror mode — flips the frame horizontally so it acts like a mirror.
# True = mirrored, False = normal
#CAMERA_MIRROR = True
CAMERA_MIRROR = False

# ─── Fan ──────────────────────────────────────────────────────────────────────
# Fan speed while the program is running.
#   0 = let the Pi manage the fan itself (no override)
#   1 = force low speed
#   2 = force medium-low speed
#   3 = force medium-high speed
#   4 = force full speed
# On exit the fan is always restored to 0 (Pi thermal management takes over).
FAN_SPEED = 4
FAN_SYSFS = "/sys/class/thermal/cooling_device0/cur_state"

# ─── Pipeline mode ────────────────────────────────────────────────────────────
# "mediapipe" = standard CPU-only MediaPipe Hands pipeline (safe, proven)
# "coral"     = two-stage Coral palm detection + CPU TFLite landmark pipeline
#               Requires Coral USB plugged in and libedgetpu installed.
#               Falls back to "mediapipe" mode if Coral is unavailable.
PIPELINE_MODE = "mediapipe"
#PIPELINE_MODE = "coral"

MAX_HANDS               = 2     # max hands detectable; more = slower
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
