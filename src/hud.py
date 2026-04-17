import cv2
import numpy as np
from config import (
    HUD_FONT, HUD_FONT_SCALE, HUD_THICKNESS,
    HUD_COLOR_PRIMARY, HUD_COLOR_WARN, HUD_COLOR_CORAL,
    HUD_COLOR_WHITE, HUD_COLOR_BLACK,
    HUD_BOX_ALPHA, CAMERA_WIDTH, CAMERA_HEIGHT,
)

GESTURE_EMOJI = {
    "open":         "OPEN HAND",
    "closed_fist":  "CLOSED FIST",
    "thumbs_up":    "THUMBS UP  ▲",
    "thumbs_down":  "THUMBS DWN ▼",
    "pointing":     "POINTING   ▶",
    "peace":        "PEACE / V",
    "1_fingers":    "1 FINGER",
    "2_fingers":    "2 FINGERS",
    "3_fingers":    "3 FINGERS",
    "4_fingers":    "4 FINGERS",
}


def _overlay_rect(frame, x, y, w, h, color=(0, 0, 0), alpha=0.45):
    """Draw a semi-transparent filled rectangle."""
    sub = frame[y:y+h, x:x+w]
    rect = np.full(sub.shape, color, dtype=np.uint8)
    cv2.addWeighted(rect, alpha, sub, 1 - alpha, 0, sub)
    frame[y:y+h, x:x+w] = sub


def _text(frame, text, x, y, color=HUD_COLOR_WHITE, scale=None, thickness=None):
    scale     = scale     or HUD_FONT_SCALE
    thickness = thickness or HUD_THICKNESS
    cv2.putText(frame, text, (x, y), HUD_FONT, scale, HUD_COLOR_BLACK,
                thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), HUD_FONT, scale, color,
                thickness, cv2.LINE_AA)


def draw_hand(frame, detection, idx):
    """Draw bbox, landmarks overlay, and per-hand info panel."""
    bx, by, bw, bh = detection['bbox']
    cx, cy          = detection['center']
    label           = detection['handedness']
    fingers         = detection['finger_count']
    gesture         = detection['gesture']
    nx, ny          = detection['coords_norm']

    box_color = HUD_COLOR_PRIMARY if label == "Right" else HUD_COLOR_WARN

    # ── Bounding box ────────────────────────────────────────────────────────
    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), box_color, 2)

    # ── Centre crosshair ────────────────────────────────────────────────────
    cv2.drawMarker(frame, (cx, cy), box_color,
                   cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)

    # ── Per-hand floating label above bbox ──────────────────────────────────
    gesture_str = GESTURE_EMOJI.get(gesture, gesture.upper())
    label_text  = f"{label} | {gesture_str} | {fingers}F"
    (tw, th), _ = cv2.getTextSize(label_text, HUD_FONT, HUD_FONT_SCALE, HUD_THICKNESS)
    lx = max(0, bx)
    ly = max(th + 8, by - 6)
    _overlay_rect(frame, lx, ly - th - 6, tw + 10, th + 10, alpha=0.5)
    _text(frame, label_text, lx + 4, ly, color=box_color)

    # ── Coordinate labels inside bounding box ───────────────────────────────
    coord_text = f"norm ({nx:.2f}, {ny:.2f})"
    box_text   = f"box {bw}x{bh}px"
    _text(frame, coord_text, bx + 4, by + bh - 18,
          color=HUD_COLOR_WHITE, scale=0.45, thickness=1)
    _text(frame, box_text, bx + 4, by + bh - 6,
          color=HUD_COLOR_WHITE, scale=0.45, thickness=1)

    # ── Pixel coordinate at centre ──────────────────────────────────────────
    px_text = f"px ({cx}, {cy})"
    _text(frame, px_text, cx + 8, cy - 8, color=box_color, scale=0.45, thickness=1)


def _temp_color(temp_c: float) -> tuple[int, int, int]:
    """
    Return a BGR color interpolated across the temperature scale:
      < 45°C  → blue
      45–60°C → green
      60–75°C → orange
      >= 75°C → red
    Smoothly interpolated between bands.
    """
    def lerp(a, b, t):
        t = max(0.0, min(1.0, t))
        return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))

    BLUE   = (255, 100,   0)   # BGR
    GREEN  = (  0, 200,   0)
    ORANGE = (  0, 145, 255)
    RED    = (  0,   0, 220)

    if temp_c < 45:
        return BLUE
    elif temp_c < 60:
        return lerp(BLUE, GREEN, (temp_c - 45) / 15)
    elif temp_c < 75:
        return lerp(GREEN, ORANGE, (temp_c - 60) / 15)
    else:
        return lerp(ORANGE, RED, (temp_c - 75) / 15)


def draw_global_hud(frame, detections, fps, coral_active: bool = False,
                    temp_c: float | None = None, det_ms: float = 0.0):
    """Draw the top-left status panel and FPS. Shows Coral indicator if active."""
    h, w = frame.shape[:2]

    # ── FPS – top right ─────────────────────────────────────────────────────
    fps_text = f"FPS: {fps:.1f}"
    (fw, fh), _ = cv2.getTextSize(fps_text, HUD_FONT, 0.55, 2)
    _overlay_rect(frame, w - fw - 16, 6, fw + 12, fh + 10, alpha=0.5)
    _text(frame, fps_text, w - fw - 10, fh + 10,
          color=HUD_COLOR_PRIMARY, scale=0.55, thickness=2)

    # ── Coral indicator – below FPS ─────────────────────────────────────────
    right_y = fh + 22
    if coral_active:
        coral_text = "CORAL TPU"
        (cw, ch), _ = cv2.getTextSize(coral_text, HUD_FONT, 0.45, 1)
        cx_pos = w - cw - 16
        _overlay_rect(frame, cx_pos - 2, right_y - ch - 2, cw + 10, ch + 8, alpha=0.5)
        _text(frame, coral_text, cx_pos + 2, right_y + 2,
              color=HUD_COLOR_CORAL, scale=0.45, thickness=1)
        right_y += 18

    # ── Temperature – below Coral indicator (or below FPS if no Coral) ──────
    if temp_c is not None:
        temp_f    = temp_c * 9 / 5 + 32
        temp_text = f"{temp_c:.1f}C  {temp_f:.1f}F"
        color     = _temp_color(temp_c)
        (tw, th), _ = cv2.getTextSize(temp_text, HUD_FONT, 0.45, 1)
        tx_pos = w - tw - 16
        _overlay_rect(frame, tx_pos - 2, right_y - th - 2, tw + 10, th + 8, alpha=0.5)
        _text(frame, temp_text, tx_pos + 2, right_y + 2,
              color=color, scale=0.45, thickness=1)
        right_y += 18

    # ── Detection time – rolling 30-frame average ────────────────────────────
    if det_ms > 0:
        det_text = f"DET: {det_ms:.1f}ms"
        (dw, dh), _ = cv2.getTextSize(det_text, HUD_FONT, 0.45, 1)
        dx_pos = w - dw - 16
        _overlay_rect(frame, dx_pos - 2, right_y - dh - 2, dw + 10, dh + 8, alpha=0.5)
        _text(frame, det_text, dx_pos + 2, right_y + 2,
              color=HUD_COLOR_WHITE, scale=0.45, thickness=1)

    # ── Hands detected count – top left ─────────────────────────────────────
    n     = len(detections)
    title = f"HANDS: {n}"
    _overlay_rect(frame, 6, 6, 160, 30, alpha=0.5)
    _text(frame, title, 10, 26, color=HUD_COLOR_PRIMARY, scale=0.65, thickness=2)

    # ── Per-hand summary rows ────────────────────────────────────────────────
    for i, d in enumerate(detections):
        row_y   = 44 + i * 22
        summary = (f"  {d['handedness']}: "
                   f"{GESTURE_EMOJI.get(d['gesture'], d['gesture'].upper())} "
                   f"| {d['finger_count']} fingers")
        _overlay_rect(frame, 6, row_y - 14, 310, 20, alpha=0.45)
        _text(frame, summary, 10, row_y,
              color=HUD_COLOR_WHITE, scale=0.48, thickness=1)

    # ── No-hand message ─────────────────────────────────────────────────────
    if n == 0:
        msg = "No hand detected"
        (mw, mh), _ = cv2.getTextSize(msg, HUD_FONT, 0.7, 2)
        mx = (w - mw) // 2
        my = (h + mh) // 2
        _overlay_rect(frame, mx - 10, my - mh - 10, mw + 20, mh + 20, alpha=0.55)
        _text(frame, msg, mx, my, color=HUD_COLOR_WARN, scale=0.7, thickness=2)
