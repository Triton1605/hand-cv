"""
coral_pipeline.py — Two-stage hand detection pipeline

Stage 1: Palm detection on Coral Edge TPU (fast, runs on every frame)
Stage 2: Hand landmark inference on CPU via TFLite (runs per detected palm)

This replaces MediaPipe's Hands solution entirely when PIPELINE_MODE = "coral".
The gesture logic in detector.py is reused unchanged — it only needs a list of
landmark objects in the same format MediaPipe produces.

Pipeline architecture
---------------------
1. Resize full BGR frame to 192×192, normalise to [0,1]
2. Run palm detection model on Coral Edge TPU
   - Output: (1, 2016, 18) raw boxes + (1, 2016, 1) scores
   - Decode with SSD anchor scheme (4 layers, strides 8/16/16/16)
   - Apply sigmoid to scores, NMS to boxes
   - Returns up to MAX_HANDS palm ROIs
3. For each palm ROI:
   - Compute a square crop with padding + rotation alignment
   - Resize to 224×224, normalise to [0,1]
   - Run landmark model on CPU TFLite
   - Output: (1, 63) landmarks + (1,1) presence score + (1,1) handedness
   - Project landmarks back to full frame coordinates
4. Build detection dicts matching the format from detector.py

Anchor scheme (from mediapipe/modules/palm_detection/palm_detection_cpu.pbtxt)
  num_layers: 4
  min_scale: 0.1484375
  max_scale: 0.75
  input_size: 192×192
  anchor_offset: 0.5
  strides: [8, 16, 16, 16]
  aspect_ratios: [1.0]
  fixed_anchor_size: true
"""

import math
import os
import numpy as np
import cv2

# ── Model paths ───────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_HERE)

PALM_MODEL_PATH = os.path.join(
    _PROJ_ROOT, "models",
    "palm_detection_builtin_256_integer_quant_edgetpu.tflite",
)
LANDMARK_MODEL_PATH = os.path.join(
    _PROJ_ROOT, "venv", "lib", "python3.11", "site-packages",
    "mediapipe", "modules", "hand_landmark", "hand_landmark_lite.tflite",
)

# ── Palm detector constants ───────────────────────────────────────────────────
PALM_INPUT_SIZE   = 192
PALM_SCORE_THRESH = 0.3 #0.5
PALM_NMS_THRESH   = 0.3

# SSD anchor parameters (from palm_detection_cpu.pbtxt)
_ANCHOR_PARAMS = dict(
    num_layers=4,
    min_scale=0.1484375,
    max_scale=0.75,
    input_size=PALM_INPUT_SIZE,
    anchor_offset=0.5,
    strides=[8, 16, 16, 16],
    aspect_ratios=[1.0],
)

# ── Landmark constants ────────────────────────────────────────────────────────
LANDMARK_INPUT_SIZE   = 224
LANDMARK_SCORE_THRESH = 0.5

# How much to expand the palm crop before sending to landmark model
# MediaPipe uses ~2.9 to ensure fingertips stay inside the crop
CROP_EXPAND_FACTOR = 3.5 #2.9
CROP_SHIFT_X       = 0.0   # normalised shift of crop centre
CROP_SHIFT_Y       = -0.5  # shift upward so fingers are included


# ── Anchor generation ─────────────────────────────────────────────────────────

def _generate_anchors(params: dict) -> np.ndarray:
    """
    Generate SSD anchor boxes matching MediaPipe's SsdAnchorsCalculator.
    Returns array of shape (num_anchors, 4): [x_center, y_center, w, h]
    all normalised to [0, 1].
    """
    anchors = []
    num_layers  = params['num_layers']
    min_scale   = params['min_scale']
    max_scale   = params['max_scale']
    size        = params['input_size']
    offset      = params['anchor_offset']
    strides     = params['strides']
    aspect_ratios = params['aspect_ratios']

    layer_idx = 0
    for stride in strides:
        feature_map_size = math.ceil(size / stride)
        for y in range(feature_map_size):
            for x in range(feature_map_size):
                x_center = (x + offset) / feature_map_size
                y_center = (y + offset) / feature_map_size
                for aspect_ratio in aspect_ratios:
                    scale = min_scale + (max_scale - min_scale) * layer_idx / (num_layers - 1)
                    ratio_sqrt = math.sqrt(aspect_ratio)
                    anchors.append([x_center, y_center,
                                    scale / ratio_sqrt, scale * ratio_sqrt])
                    # Interpolated scale anchor
                    if layer_idx + 1 < num_layers:
                        scale_next = min_scale + (max_scale - min_scale) * (layer_idx + 1) / (num_layers - 1)
                    else:
                        scale_next = 1.0
                    anchors.append([x_center, y_center,
                                    math.sqrt(scale * scale_next),
                                    math.sqrt(scale * scale_next)])
        layer_idx += 1

    return np.array(anchors, dtype=np.float32)


# Pre-generate anchors once at import time
_ANCHORS = _generate_anchors(_ANCHOR_PARAMS)


# ── Box decoding & NMS ────────────────────────────────────────────────────────

def _decode_boxes(raw_boxes: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """
    Decode raw SSD regression outputs into [x_center, y_center, w, h]
    normalised to [0, 1] relative to the input image.

    raw_boxes: (N, 18) — first 4 are box deltas, remaining 14 are keypoints
    anchors:   (N, 4)  — [x_center, y_center, w, h]
    """
    scale = PALM_INPUT_SIZE
    boxes = np.zeros_like(raw_boxes[:, :4])

    # x_center, y_center decoded relative to anchor centre
    boxes[:, 0] = raw_boxes[:, 0] / scale * anchors[:, 2] + anchors[:, 0]
    boxes[:, 1] = raw_boxes[:, 1] / scale * anchors[:, 3] + anchors[:, 1]
    # width, height
    boxes[:, 2] = raw_boxes[:, 2] / scale * anchors[:, 2]
    boxes[:, 3] = raw_boxes[:, 3] / scale * anchors[:, 3]

    return boxes


def _nms(boxes_cx_cy_wh: np.ndarray, scores: np.ndarray,
         score_thresh: float, nms_thresh: float,
         max_detections: int) -> list[int]:
    """
    Non-maximum suppression. Returns list of kept indices.
    boxes are [cx, cy, w, h] normalised.
    """
    # Filter by score threshold
    mask = scores > score_thresh
    if not np.any(mask):
        return []

    filtered_idx    = np.where(mask)[0]
    filtered_scores = scores[filtered_idx]
    filtered_boxes  = boxes_cx_cy_wh[filtered_idx]

    # Convert to [x1, y1, x2, y2]
    x1 = filtered_boxes[:, 0] - filtered_boxes[:, 2] / 2
    y1 = filtered_boxes[:, 1] - filtered_boxes[:, 3] / 2
    x2 = filtered_boxes[:, 0] + filtered_boxes[:, 2] / 2
    y2 = filtered_boxes[:, 1] + filtered_boxes[:, 3] / 2
    xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

    # cv2.dnn.NMSBoxes expects list of [x, y, w, h]
    cv_boxes = [[float(x1[i]), float(y1[i]),
                 float(x2[i] - x1[i]), float(y2[i] - y1[i])]
                for i in range(len(filtered_idx))]
    indices = cv2.dnn.NMSBoxes(
        cv_boxes,
        filtered_scores.tolist(),
        score_thresh,
        nms_thresh,
    )
    if len(indices) == 0:
        return []
    kept = [filtered_idx[i] for i in indices.flatten()[:max_detections]]
    return kept


# ── Crop utilities ────────────────────────────────────────────────────────────

def _palm_box_to_crop(box_cx_cy_wh: np.ndarray,
                      raw_kps: np.ndarray,
                      anchor: np.ndarray,
                      frame_w: int, frame_h: int):
    """
    Convert a decoded palm box into a square crop for the landmark model.
    Simple padded square with upward shift — no rotation.
    Returns (M, M_inv, cx_px, cy_px, side_px, angle=0).
    """
    cx, cy, w, h = box_cx_cy_wh

    # Square side with generous padding to include fingers
    side = max(w, h) * CROP_EXPAND_FACTOR

    # Shift crop centre upward toward fingers
    cy_shifted = cy + (CROP_SHIFT_Y * side)

    # Convert to pixels
    cx_px   = cx         * frame_w
    cy_px   = cy_shifted * frame_h
    side_px = side       * max(frame_w, frame_h)
    half    = side_px / 2.0

    # Source corners: top-left, top-right, bottom-left
    src = np.array([
        [cx_px - half, cy_px - half],
        [cx_px + half, cy_px - half],
        [cx_px - half, cy_px + half],
    ], dtype=np.float32)

    dst = np.array([
        [0,                   0                  ],
        [LANDMARK_INPUT_SIZE, 0                  ],
        [0,                   LANDMARK_INPUT_SIZE],
    ], dtype=np.float32)

    M     = cv2.getAffineTransform(src, dst)
    M_inv = cv2.getAffineTransform(dst, src)

    return M, M_inv, cx_px, cy_px, int(side_px), 0.0


def _warp_frame(frame_bgr: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply affine warp to extract the landmark crop."""
    h, w = frame_bgr.shape[:2]
    crop_rgb = cv2.warpAffine(
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
        M, (LANDMARK_INPUT_SIZE, LANDMARK_INPUT_SIZE),
    )
    return crop_rgb


def _project_landmarks(raw_lm: np.ndarray, M_inv: np.ndarray,
                        frame_w: int, frame_h: int):
    """
    Project 21 landmarks from crop space back to full frame pixel space.
    raw_lm: (63,) float — x,y,z interleaved, normalised to [0,1] in crop
    Returns: list of (x_px, y_px, z) tuples
    """
    pts = raw_lm.reshape(21, 3)
    # Scale from [0,1] to crop pixels
    pts_px = pts[:, :2] * LANDMARK_INPUT_SIZE

    # Apply inverse affine
    ones   = np.ones((21, 1), dtype=np.float32)
    pts_h  = np.hstack([pts_px, ones])   # (21, 3)
    proj   = (M_inv @ pts_h.T).T         # (21, 2)

    # Normalise to [0,1] in full frame
    proj[:, 0] /= frame_w
    proj[:, 1] /= frame_h

    return proj, pts[:, 2]   # normalised xy, raw z


# ── Fake landmark container (mimics MediaPipe NormalizedLandmark) ─────────────

class _FakeLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

class _FakeLandmarkList:
    def __init__(self, landmarks):
        self.landmark = landmarks


# ── Main pipeline class ───────────────────────────────────────────────────────

class CoralHandPipeline:
    """
    Full two-stage hand detection pipeline:
      Stage 1: Coral Edge TPU palm detection
      Stage 2: CPU TFLite landmark estimation

    Produces the same detection dict format as HandDetector.process().
    """

    def __init__(self, max_hands: int = 2):
        from tflite_runtime.interpreter import Interpreter, load_delegate
        import ctypes.util

        self.max_hands = max_hands
        self.available = False

        # ── Palm detector on Coral ─────────────────────────────────────────
        lib = ctypes.util.find_library("edgetpu")
        if lib is None:
            raise RuntimeError("libedgetpu not found")
        if not os.path.isfile(PALM_MODEL_PATH):
            raise RuntimeError(f"Palm model not found: {PALM_MODEL_PATH}")

        delegate = load_delegate(lib)
        self._palm = Interpreter(
            model_path=PALM_MODEL_PATH,
            experimental_delegates=[delegate],
        )
        self._palm.allocate_tensors()
        self._palm_in  = self._palm.get_input_details()[0]['index']
        self._palm_out = self._palm.get_output_details()

        # ── Landmark model on CPU ──────────────────────────────────────────
        if not os.path.isfile(LANDMARK_MODEL_PATH):
            raise RuntimeError(f"Landmark model not found: {LANDMARK_MODEL_PATH}")

        self._lm = Interpreter(model_path=LANDMARK_MODEL_PATH)
        self._lm.allocate_tensors()
        self._lm_in  = self._lm.get_input_details()[0]['index']
        self._lm_out = self._lm.get_output_details()

        self.available = True
        print("[coral_pipeline] Palm (Coral) + Landmark (CPU) pipeline ready.")

    def process(self, bgr_frame: np.ndarray) -> list[dict]:
        """
        Process a BGR frame. Returns list of detection dicts matching the
        format produced by HandDetector.process() in detector.py.
        """
        if not self.available:
            return []

        h, w = bgr_frame.shape[:2]

        # ── Stage 1: palm detection on Coral ──────────────────────────────
        resized = cv2.resize(bgr_frame, (PALM_INPUT_SIZE, PALM_INPUT_SIZE))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        inp     = (rgb.astype(np.float32) / 255.0)[np.newaxis]

        self._palm.set_tensor(self._palm_in, inp)
        self._palm.invoke()

        # Output 0 = boxes (1,2016,18), Output 1 = scores (1,2016,1)
        raw_boxes  = self._palm.get_tensor(self._palm_out[0]['index'])[0]  # (2016,18)
        raw_scores = self._palm.get_tensor(self._palm_out[1]['index'])[0, :, 0]  # (2016,)

        # Sigmoid scores
        scores = 1.0 / (1.0 + np.exp(-raw_scores))

        # Decode boxes
        boxes = _decode_boxes(raw_boxes, _ANCHORS)  # (2016, 4) cx,cy,w,h norm

        # NMS
        kept = _nms(boxes, scores, PALM_SCORE_THRESH, PALM_NMS_THRESH,
                    self.max_hands)
        if not kept:
            return []

        # ── Stage 2: landmarks per palm ────────────────────────────────────
        detections = []
        for idx in kept:
            box = boxes[idx]        # cx, cy, w, h normalised
            kps = raw_boxes[idx, 4:]  # 14 keypoint deltas
            anchor = _ANCHORS[idx]

            # Build affine crop
            M, M_inv, cx_px, cy_px, side_px, angle = _palm_box_to_crop(
                box, kps, anchor, w, h
            )
            crop_rgb = _warp_frame(bgr_frame, M)

            # Run landmark model
            lm_inp = (crop_rgb.astype(np.float32) / 255.0)[np.newaxis]
            self._lm.set_tensor(self._lm_in, lm_inp)
            self._lm.invoke()

            # Output 0 = landmarks (1,63), Output 1 = presence (1,1),
            # Output 2 = handedness (1,1)
            lm_raw      = self._lm.get_tensor(self._lm_out[0]['index'])[0]  # (63,)
            presence    = float(self._lm.get_tensor(self._lm_out[1]['index'])[0, 0])
            handedness  = float(self._lm.get_tensor(self._lm_out[2]['index'])[0, 0])

            if presence < LANDMARK_SCORE_THRESH:
                continue

            # Project landmarks back to full frame
            lm_norm, lm_z = _project_landmarks(lm_raw, M_inv, w, h)

            # Build fake landmark list mimicking MediaPipe's format
            fake_lms = _FakeLandmarkList([
                _FakeLandmark(lm_norm[i, 0], lm_norm[i, 1], lm_z[i])
                for i in range(21)
            ])

            # handedness: >0.5 = right hand (from model's perspective)
            label = "Right" if handedness > 0.5 else "Left"

            # Bounding box in pixels
            xs = [int(lm_norm[i, 0] * w) for i in range(21)]
            ys = [int(lm_norm[i, 1] * h) for i in range(21)]
            x1 = max(0, min(xs) - 15)
            y1 = max(0, min(ys) - 15)
            x2 = min(w, max(xs) + 15)
            y2 = min(h, max(ys) + 15)

            cx_lm = int(sum(xs) / 21)
            cy_lm = int(sum(ys) / 21)

            # Wrist normalised coords
            nx = round(float(lm_norm[0, 0]), 3)
            ny = round(float(lm_norm[0, 1]), 3)

            # Import gesture/finger logic from detector.py
            from detector import count_fingers, detect_gesture
            fingers = count_fingers(fake_lms, w, h, label)
            gesture = detect_gesture(fake_lms, fingers, w, h)

            detections.append({
                'landmarks':    fake_lms,
                'handedness':   label,
                'finger_count': fingers,
                'gesture':      gesture,
                'bbox':         (x1, y1, x2 - x1, y2 - y1),
                'center':       (cx_lm, cy_lm),
                'coords_norm':  (nx, ny),
            })

        return detections

    def draw_landmarks(self, frame, landmarks):
        """Draw landmark connections on frame (mimics HandDetector.draw_landmarks)."""
        h, w = frame.shape[:2]
        lms  = landmarks.landmark

        # MediaPipe hand connection pairs
        connections = [
            (0,1),(1,2),(2,3),(3,4),          # thumb
            (0,5),(5,6),(6,7),(7,8),           # index
            (0,9),(9,10),(10,11),(11,12),      # middle
            (0,13),(13,14),(14,15),(15,16),    # ring
            (0,17),(17,18),(18,19),(19,20),    # pinky
            (5,9),(9,13),(13,17),              # palm
        ]
        for a, b in connections:
            x1 = int(lms[a].x * w); y1 = int(lms[a].y * h)
            x2 = int(lms[b].x * w); y2 = int(lms[b].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (80, 80, 80), 1, cv2.LINE_AA)
        for lm in lms:
            cx = int(lm.x * w); cy = int(lm.y * h)
            cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1, cv2.LINE_AA)

    def close(self):
        self._palm = None
        self._lm   = None
        self.available = False
