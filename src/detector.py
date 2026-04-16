import math
import mediapipe as mp
import numpy as np
import cv2
from config import (
    MAX_HANDS, DETECTION_CONFIDENCE, TRACKING_CONFIDENCE,
    CAMERA_WIDTH, CAMERA_HEIGHT
)

mp_hands      = mp.solutions.hands
mp_drawing    = mp.solutions.drawing_utils
mp_draw_style = mp.solutions.drawing_styles

# Landmark indices
WRIST       = 0
THUMB_CMC   = 1; THUMB_MCP  = 2; THUMB_IP   = 3; THUMB_TIP  = 4
INDEX_MCP   = 5; INDEX_PIP  = 6; INDEX_DIP  = 7; INDEX_TIP  = 8
MIDDLE_MCP  = 9; MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP    = 13; RING_PIP  = 14; RING_DIP   = 15; RING_TIP   = 16
PINKY_MCP   = 17; PINKY_PIP = 18; PINKY_DIP  = 19; PINKY_TIP  = 20

FINGER_TIPS = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_PIPS = [INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]


def _lm(hand_landmarks, idx, w, h):
    """Return pixel coords (x, y) for a landmark index."""
    lm = hand_landmarks.landmark[idx]
    return int(lm.x * w), int(lm.y * h)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle_3pts(a, b, c):
    """Angle at vertex b formed by points a-b-c, in degrees."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag = (math.hypot(*ba) * math.hypot(*bc)) + 1e-6
    return math.degrees(math.acos(max(-1, min(1, dot / mag))))


def count_fingers(hand_landmarks, w, h, handedness_label):
    """Return number of extended fingers (0-5)."""
    count = 0

    # Four fingers: tip above PIP = extended
    for tip_idx, pip_idx in zip(FINGER_TIPS, FINGER_PIPS):
        tip = _lm(hand_landmarks, tip_idx, w, h)
        pip = _lm(hand_landmarks, pip_idx, w, h)
        if tip[1] < pip[1]:   # y increases downward
            count += 1

    # Thumb: compare tip x to MCP x, mirrored for each hand
    thumb_tip = _lm(hand_landmarks, THUMB_TIP, w, h)
    thumb_mcp = _lm(hand_landmarks, THUMB_MCP, w, h)
    if handedness_label == "Right":
        if thumb_tip[0] < thumb_mcp[0]:
            count += 1
    else:
        if thumb_tip[0] > thumb_mcp[0]:
            count += 1

    return count


def detect_gesture(hand_landmarks, finger_count, w, h):
    """
    Return a gesture string based on landmarks + finger count.
    Priority: thumbs_up > thumbs_down > open > closed > counts
    """
    wrist      = _lm(hand_landmarks, WRIST,     w, h)
    thumb_tip  = _lm(hand_landmarks, THUMB_TIP, w, h)
    index_tip  = _lm(hand_landmarks, INDEX_TIP, w, h)
    index_pip  = _lm(hand_landmarks, INDEX_PIP, w, h)
    middle_tip = _lm(hand_landmarks, MIDDLE_TIP, w, h)
    middle_pip = _lm(hand_landmarks, MIDDLE_PIP, w, h)

    # ── Thumbs up: thumb tip clearly above wrist, other fingers curled ─────
    fingers_curled = (
        index_tip[1]  > index_pip[1] and
        middle_tip[1] > middle_pip[1]
    )
    thumb_high = thumb_tip[1] < wrist[1] - 30
    if fingers_curled and thumb_high:
        return "thumbs_up"

    # ── Thumbs down: thumb tip clearly below wrist, other fingers curled ───
    thumb_low = thumb_tip[1] > wrist[1] + 30
    if fingers_curled and thumb_low:
        return "thumbs_down"

    # ── Open / closed ───────────────────────────────────────────────────────
    if finger_count == 5:
        return "open"
    if finger_count == 0:
        return "closed_fist"

    # ── Pointing (index only) ───────────────────────────────────────────────
    ring_tip  = _lm(hand_landmarks, RING_TIP,  w, h)
    ring_pip  = _lm(hand_landmarks, RING_PIP,  w, h)
    pinky_tip = _lm(hand_landmarks, PINKY_TIP, w, h)
    pinky_pip = _lm(hand_landmarks, PINKY_PIP, w, h)

    index_up  = index_tip[1]  < index_pip[1]
    middle_dn = middle_tip[1] > middle_pip[1]
    ring_dn   = ring_tip[1]   > ring_pip[1]
    pinky_dn  = pinky_tip[1]  > pinky_pip[1]

    if index_up and middle_dn and ring_dn and pinky_dn:
        return "pointing"

    # ── Peace / victory ─────────────────────────────────────────────────────
    middle_up = middle_tip[1] < middle_pip[1]
    if index_up and middle_up and ring_dn and pinky_dn:
        return "peace"

    # ── Generic finger count ────────────────────────────────────────────────
    return f"{finger_count}_fingers"


class HandDetector:
    def __init__(self, coral_detector=None):
        """
        coral_detector: optional CoralPalmDetector instance.
        If provided and available, it will be used for palm detection.
        MediaPipe landmark tracking always runs on CPU.
        """
        self._coral = coral_detector
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_HANDS,
            min_detection_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE,
        )

    def process(self, bgr_frame):
        """
        Process a BGR frame. Returns list of dicts, one per detected hand:
          {
            'landmarks':   hand_landmarks,
            'handedness':  'Left' | 'Right',
            'finger_count': int,
            'gesture':     str,
            'bbox':        (x, y, w, h),      # pixel bounding box
            'center':      (cx, cy),           # palm centre pixels
            'coords_norm': (nx, ny),           # normalised 0-1 wrist position
          }

        When a Coral detector is active, it runs palm detection first and
        MediaPipe uses those ROIs. If Coral returns no palms, MediaPipe still
        runs its own full-frame detection as a fallback.
        """
        h, w = bgr_frame.shape[:2]
        rgb  = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        detections = []

        if not results.multi_hand_landmarks:
            return detections

        for hand_lm, hand_info in zip(
            results.multi_hand_landmarks,
            results.multi_handedness,
        ):
            label = hand_info.classification[0].label   # 'Left' or 'Right'

            # Bounding box
            xs = [int(lm.x * w) for lm in hand_lm.landmark]
            ys = [int(lm.y * h) for lm in hand_lm.landmark]
            x1, y1 = max(0, min(xs)-15), max(0, min(ys)-15)
            x2, y2 = min(w, max(xs)+15), min(h, max(ys)+15)

            # Wrist normalised
            wrist_lm  = hand_lm.landmark[WRIST]
            nx, ny    = round(wrist_lm.x, 3), round(wrist_lm.y, 3)

            # Palm centre = average of all landmarks
            cx = int(sum(xs) / len(xs))
            cy = int(sum(ys) / len(ys))

            fingers = count_fingers(hand_lm, w, h, label)
            gesture = detect_gesture(hand_lm, fingers, w, h)

            detections.append({
                'landmarks':    hand_lm,
                'handedness':   label,
                'finger_count': fingers,
                'gesture':      gesture,
                'bbox':         (x1, y1, x2 - x1, y2 - y1),
                'center':       (cx, cy),
                'coords_norm':  (nx, ny),
            })

        return detections

    def draw_landmarks(self, frame, landmarks):
        mp_drawing.draw_landmarks(
            frame,
            landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_draw_style.get_default_hand_landmarks_style(),
            mp_draw_style.get_default_hand_connections_style(),
        )

    def close(self):
        self.hands.close()
