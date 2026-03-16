import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import json
import os
import uuid
from plant_detector import detect_plants

MODEL_PATH         = "plant_classifier.h5"
CLASS_INDICES_PATH = "class_indices.json"
IMG_SIZE           = 224
RESULTS_DIR        = "static/results"

# ─────────────────────────────────────────────
# PER-SPECIES CONFIDENCE THRESHOLDS
# Each species has its own threshold based on
# how visually distinctive it is
# ─────────────────────────────────────────────
SPECIES_THRESHOLDS = {
    "aloe_vera": 0.92,   # strict — very distinct, avoid false positives
    "brahmi":    0.82,   # lenient — harder to detect, small round leaves
    "centella":  0.88,   # balanced
    "turmeric":  0.88,   # balanced
}

# Minimum margin between top and second prediction
MIN_MARGIN = 0.20

# ─────────────────────────────────────────────
# COLOR MAP — different color per species
# ─────────────────────────────────────────────
SPECIES_COLORS = {
    "aloe_vera": (0,   200,  80),   # green
    "brahmi":    (255, 140,   0),   # orange
    "centella":  (0,   180, 255),   # blue
    "turmeric":  (0,    60, 255),   # dark blue
}
DEFAULT_COLOR = (180, 180, 180)

# ─────────────────────────────────────────────
# ENSURE RESULTS DIRECTORY EXISTS
# ─────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"[OK] Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Could not load model: {e}")
    model = None

# ─────────────────────────────────────────────
# LOAD CLASS INDICES
# ─────────────────────────────────────────────
if os.path.exists(CLASS_INDICES_PATH):
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
    classes = {v: k for k, v in class_indices.items()}
    print(f"[OK] Classes loaded: {classes}")
else:
    print(f"[WARN] class_indices.json not found. Using fallback.")
    classes = {0: "aloe_vera", 1: "brahmi", 2: "centella", 3: "turmeric", 4: "unknown"}

# ─────────────────────────────────────────────
# LOAD MEDICINAL DATA
# ─────────────────────────────────────────────
try:
    data_df        = pd.read_csv("medicinal_data.csv")
    medicinal_data = data_df.set_index("plant").to_dict("index")
    print(f"[OK] Medicinal data loaded")
except Exception as e:
    print(f"[ERROR] Could not load medicinal_data.csv: {e}")
    medicinal_data = {}


# ─────────────────────────────────────────────
# PREPROCESS SINGLE CROP
# ─────────────────────────────────────────────
def preprocess(crop):
    resized    = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0)


# ─────────────────────────────────────────────
# CLASSIFY SINGLE CROP
# Uses per-species threshold — no global MIN_CONFIDENCE
# ─────────────────────────────────────────────
def classify_crop(crop):
    if crop.shape[0] < 40 or crop.shape[1] < 40:
        return None

    processed  = preprocess(crop)
    prediction = model.predict(processed, verbose=0)

    sorted_probs       = np.sort(prediction[0])
    top_probability    = sorted_probs[-1]
    second_probability = sorted_probs[-2]
    class_index        = np.argmax(prediction[0])

    # Get plant name first
    plant_name = classes.get(class_index, "unknown")

    # Reject unknown class
    if plant_name == "unknown":
        return None

    # Per-species threshold — brahmi uses 0.82, aloe uses 0.92
    threshold = SPECIES_THRESHOLDS.get(plant_name, 0.88)
    if top_probability < threshold:
        return None

    # Reject ambiguous predictions where margin is too small
    if (top_probability - second_probability) < MIN_MARGIN:
        return None

    return (plant_name, float(top_probability))


# ─────────────────────────────────────────────
# NON-MAXIMUM SUPPRESSION
# Merges overlapping boxes keeping highest confidence
# ─────────────────────────────────────────────
def apply_nms(candidates, iou_threshold=0.3):
    if not candidates:
        return []

    by_plant = {}
    for c in candidates:
        p = c["plant"]
        if p not in by_plant:
            by_plant[p] = []
        by_plant[p].append(c)

    final = []

    for plant, items in by_plant.items():
        items = sorted(items, key=lambda x: x["confidence"], reverse=True)

        kept = []
        while items:
            best = items.pop(0)
            kept.append(best)

            x1b, y1b, x2b, y2b = best["box"]
            area_b = (x2b - x1b) * (y2b - y1b)

            remaining = []
            for item in items:
                x1i, y1i, x2i, y2i = item["box"]
                area_i = (x2i - x1i) * (y2i - y1i)

                ix1 = max(x1b, x1i)
                iy1 = max(y1b, y1i)
                ix2 = min(x2b, x2i)
                iy2 = min(y2b, y2i)

                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                intersection = iw * ih

                union = area_b + area_i - intersection
                iou   = intersection / union if union > 0 else 0

                if iou < iou_threshold:
                    remaining.append(item)

            items = remaining

        final.append(kept[0])

    return final


# ─────────────────────────────────────────────
# DRAW BOUNDING BOXES — different color per species
# ─────────────────────────────────────────────
def draw_boxes(img, results):
    annotated = img.copy()

    for r in results:
        x1, y1, x2, y2 = r["box"]
        plant_name      = r["plant"].replace("_", " ").title()
        confidence      = r["confidence"]
        color           = SPECIES_COLORS.get(r["plant"], DEFAULT_COLOR)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        label      = f"{plant_name} {confidence}%"
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness  = 2

        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

        cv2.rectangle(
            annotated,
            (x1, y1 - th - 12),
            (x1 + tw + 8, y1),
            color,
            -1
        )

        cv2.putText(
            annotated,
            label,
            (x1 + 4, y1 - 6),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )

    return annotated


# ─────────────────────────────────────────────
# MAIN PREDICTION FUNCTION
# Returns: (results, annotated_image_path, error)
# ─────────────────────────────────────────────
def predict_plant(image_path):

    if model is None:
        return [], None, "Model could not be loaded."

    img = cv2.imread(image_path)
    if img is None:
        return [], None, "Image could not be loaded."

    img_area = img.shape[0] * img.shape[1]

    # Get sliding window candidates
    detections = detect_plants(image_path)

    # Fallback only when sliding window finds nothing
    if not detections:
        h, w = img.shape[:2]
        detections = [{"crop": img, "box": (0, 0, w, h), "conf": 1.0}]

    # Classify every candidate region
    candidates = []
    for det in detections:
        crop            = det["crop"]
        box             = det["box"]
        x1, y1, x2, y2 = box
        box_area        = (x2 - x1) * (y2 - y1)

        # Reject crops covering more than 60% of image
        # Prevents full-scene crops from causing false predictions
        if box_area > 0.60 * img_area:
            continue

        prediction = classify_crop(crop)
        if prediction is None:
            continue

        plant_name, confidence = prediction

        candidates.append({
            "plant":      plant_name,
            "confidence": round(confidence * 100, 2),
            "box":        box
        })

    # Apply NMS to collapse overlapping detections
    final_detections = apply_nms(candidates, iou_threshold=0.3)

    # Build results with medicinal info
    results     = []
    seen_plants = set()

    for det in final_detections:
        plant_name = det["plant"]

        if plant_name in seen_plants:
            continue
        seen_plants.add(plant_name)

        info        = medicinal_data.get(plant_name, {})
        uses        = info.get("medicinal_uses", "Not available")
        precautions = info.get("precautions", "Not available")

        results.append({
            "plant":       plant_name,
            "confidence":  det["confidence"],
            "uses":        uses,
            "precautions": precautions,
            "box":         det["box"]
        })

    # Sort by confidence — highest first
    results.sort(key=lambda x: x["confidence"], reverse=True)

    # If top result is very high confidence keep only that one
    # Suppresses false secondary detections in complex scenes
    if results and results[0]["confidence"] >= 92.0:
        results = results[:1]

    # Draw bounding boxes and save annotated image
    result_image_path = None
    if results:
        annotated         = draw_boxes(img, results)
        unique_name       = f"result_{uuid.uuid4().hex}.jpg"
        save_path         = os.path.join(RESULTS_DIR, unique_name)
        cv2.imwrite(save_path, annotated)
        result_image_path = f"static/results/{unique_name}"

    if not results:
        return [], None, None

    return results, result_image_path, None