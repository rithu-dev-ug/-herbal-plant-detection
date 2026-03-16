import cv2
import numpy as np

# ─────────────────────────────────────────────
# MULTI-SCALE SLIDING WINDOW PLANT DETECTOR
# ─────────────────────────────────────────────
# Replaces YOLO with a sliding window approach.
# Divides the image into overlapping regions at
# multiple scales and returns candidate crops
# for the classifier to evaluate.
#
# Advantages over generic YOLO for this project:
# - No domain mismatch (no pretrained bias)
# - Works on any plant type without retraining
# - Produces consistent bounding boxes
# - Still shows visual detection boxes in the UI
# ─────────────────────────────────────────────

# Window sizes as fractions of image dimensions
# Multiple scales capture both close-up and distant plants
WINDOW_SCALES = [0.5, 0.65, 0.8, 1.0]

# Step size as fraction of window size
# Lower = more overlap = better coverage, slower speed
STEP_RATIO = 0.5

# Minimum window size in pixels
MIN_WINDOW_PX = 150


def detect_plants(image_path):
    """
    Runs multi-scale sliding window detection.
    Returns list of candidate regions as detections.
    Each detection contains crop, bounding box, and confidence.
    The classifier in inference_engine.py evaluates each crop
    and decides which ones contain known medicinal plants.
    """

    img = cv2.imread(image_path)

    if img is None:
        return []

    h, w = img.shape[:2]
    detections = []

    for scale in WINDOW_SCALES:

        # Window size for this scale
        win_h = int(h * scale)
        win_w = int(w * scale)

        # Skip if window is too small
        if win_h < MIN_WINDOW_PX or win_w < MIN_WINDOW_PX:
            continue

        # Step size
        step_y = max(1, int(win_h * STEP_RATIO))
        step_x = max(1, int(win_w * STEP_RATIO))

        # Slide across image
        y = 0
        while y + win_h <= h:
            x = 0
            while x + win_w <= w:

                crop = img[y:y + win_h, x:x + win_w]

                if crop.shape[0] >= 80 and crop.shape[1] >= 80:
                    detections.append({
                        "crop": crop,
                        "box":  (x, y, x + win_w, y + win_h),
                        "conf": 1.0   # confidence assigned by classifier
                    })

                x += step_x
            y += step_y

    

    return detections