"""
coral_detector.py — Google Coral USB Accelerator integration

Architecture note
-----------------
MediaPipe's hand tracking pipeline has two stages:
  1. Palm detection  — a lightweight MobileNet-based detector that finds
                        hand bounding boxes in the full frame.
  2. Landmark model  — a 21-point 3D landmark regressor that runs on the
                        cropped hand region found by stage 1.

Google published an Edge TPU-compiled version of the palm detector model
for their Coral MediaPipe examples. The landmark model was never published
in EdgeTPU format (it uses ops not fully supported by the EdgeTPU compiler).

Strategy
--------
When the Coral is available we use it to run palm detection on each frame.
The result (a list of bounding-box crops) is then fed to MediaPipe's normal
CPU-based Hands pipeline as a region-of-interest hint, bypassing MediaPipe's
own palm detector for that frame and saving the most expensive CPU work.

If the Coral is unavailable (not plugged in, libedgetpu not installed, model
file missing), this class degrades silently to a no-op and MediaPipe runs its
full CPU pipeline as usual.

Usage
-----
    coral = CoralPalmDetector("/path/to/palm_detection_edgetpu.tflite")
    if coral.available:
        rois = coral.detect(bgr_frame)   # list of (x1,y1,x2,y2) in pixels
    coral.close()
"""

import os
import numpy as np
import cv2

# The PALM_INPUT_SIZE expected by the palm detection model
PALM_INPUT_SIZE = 256


class CoralPalmDetector:
    """
    Wraps the Edge TPU palm detection tflite model.
    Provides a detect(frame) method that returns bounding box candidates.
    Falls back silently if Coral hardware/libraries are not available.
    """

    def __init__(self, model_path: str):
        self.available = False
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._model_path = model_path

        if not model_path or not os.path.isfile(model_path):
            print(f"[coral] Model not found at '{model_path}' — Coral disabled.")
            return

        try:
            self._load(model_path)
        except Exception as e:
            print(f"[coral] Failed to initialise Edge TPU: {e}")
            print("[coral] Coral disabled — running CPU-only.")

    def _load(self, model_path: str):
        """Load the EdgeTPU delegate and TFLite interpreter."""
        from tflite_runtime.interpreter import Interpreter, load_delegate 
        # Try to locate libedgetpu shared library
        import ctypes.util
        lib = ctypes.util.find_library("edgetpu")
        if lib is None:
            raise RuntimeError(
                "libedgetpu not found. Install with: sudo apt install libedgetpu1-std"
            )

        delegate = load_delegate(lib)
        self._interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[delegate],
        )
        self._interpreter.allocate_tensors()
        self._input_details  = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        self.available = True
        print(f"[coral] Edge TPU palm detector loaded: {os.path.basename(model_path)}")

    def detect(self, bgr_frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Run palm detection on a BGR frame.

        Returns a list of (x1, y1, x2, y2) pixel bounding boxes for each
        detected palm, in the coordinate space of the original frame.
        Returns an empty list if no palms detected or Coral unavailable.
        """
        if not self.available:
            return []

        h, w = bgr_frame.shape[:2]

        # Preprocess: resize to 256×256, convert to RGB, normalise to [0,1]
        resized = cv2.resize(bgr_frame, (PALM_INPUT_SIZE, PALM_INPUT_SIZE))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        inp     = (rgb.astype(np.float32) / 255.0)[np.newaxis, ...]   # (1,256,256,3)

        input_idx = self._input_details[0]['index']
        self._interpreter.set_tensor(input_idx, inp)

        try:
            self._interpreter.invoke()
        except Exception as e:
            print(f"[coral] Inference error: {e}")
            return []

        # The palm detector outputs boxes in [y_min, x_min, y_max, x_max]
        # normalised [0,1] relative to the 256×256 input.
        # Output tensor 0 = detection boxes, output tensor 1 = scores
        boxes_tensor = self._output_details[0]['index']
        score_tensor = self._output_details[1]['index']

        raw_boxes  = self._interpreter.get_tensor(boxes_tensor)   # shape varies
        raw_scores = self._interpreter.get_tensor(score_tensor)

        boxes  = np.squeeze(raw_boxes)
        scores = np.squeeze(raw_scores)

        SCORE_THRESHOLD = 0.5

        results = []
        # Handle both (N,4) and (4,) shapes (single detection edge case)
        if boxes.ndim == 1:
            boxes  = boxes[np.newaxis, :]
            scores = np.atleast_1d(scores)

        for box, score in zip(boxes, scores):
            if float(score) < SCORE_THRESHOLD:
                continue
            # box = [y_min, x_min, y_max, x_max] normalised
            y_min, x_min, y_max, x_max = box
            x1 = max(0, int(x_min * w))
            y1 = max(0, int(y_min * h))
            x2 = min(w, int(x_max * w))
            y2 = min(h, int(y_max * h))
            if x2 > x1 and y2 > y1:
                results.append((x1, y1, x2, y2))

        return results

    def close(self):
        """Clean up resources."""
        self._interpreter = None
        self.available = False
