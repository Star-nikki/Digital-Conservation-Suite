"""
Digital Conservation Suite — Core Algorithm Pipeline
=====================================================
Four-algorithm pipeline for art authentication and conservation analysis.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
import torch
import torchvision.transforms as transforms
from torchvision import models
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# ALGORITHM 1 — YOLOv8 Compositional Layout Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_composition(img: np.ndarray) -> dict:
    """
    Uses YOLOv8 for compositional layout analysis.
    Detects objects/figures and derives geometric structure.

    Args:
        img: BGR numpy array (from OpenCV)

    Returns:
        dict with keys: detections, keypoints, rule_of_thirds_score,
                        annotated_img, composition_type
    """
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")  # nano model — fastest, smallest
        results = model(img, verbose=False)

        detections = []
        annotated = img.copy()
        h, w = img.shape[:2]

        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls  = int(box.cls[0])
                    name = model.names[cls]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    detections.append({
                        "label": name,
                        "confidence": round(conf, 3),
                        "bbox": [x1, y1, x2, y2],
                        "center": [cx, cy],
                    })

                    # Draw bounding box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 180), 2)
                    cv2.putText(
                        annotated, f"{name} {conf:.2f}",
                        (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 180), 1,
                    )

        # Draw rule-of-thirds grid
        for i in [1, 2]:
            cv2.line(annotated, (w * i // 3, 0), (w * i // 3, h), (255, 180, 0), 1)
            cv2.line(annotated, (0, h * i // 3), (w, h * i // 3), (255, 180, 0), 1)

        # Rule-of-thirds score: reward centres near grid intersections
        thirds_pts = [
            (w // 3, h // 3), (2 * w // 3, h // 3),
            (w // 3, 2 * h // 3), (2 * w // 3, 2 * h // 3),
        ]
        rot_score = 0.0
        if detections:
            for det in detections:
                cx, cy = det["center"]
                min_dist = min(
                    np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                    for px, py in thirds_pts
                )
                rot_score += max(0, 1 - min_dist / (w * 0.2))
            rot_score = min(1.0, rot_score / len(detections))

        # Derive composition type heuristically
        composition_type = _infer_composition_type(detections, w, h)

        return {
            "detections": detections,
            "detection_count": len(detections),
            "rule_of_thirds_score": round(rot_score, 3),
            "composition_type": composition_type,
            "annotated_img": annotated,
        }

    except Exception as e:
        return {
            "detections": [],
            "detection_count": 0,
            "rule_of_thirds_score": 0.0,
            "composition_type": "Analysis unavailable",
            "annotated_img": img,
            "error": str(e),
        }


def _infer_composition_type(detections, w, h):
    if not detections:
        return "Abstract / Landscape (no figures)"
    centers = [d["center"] for d in detections]
    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    x_spread = (max(xs) - min(xs)) / w if len(xs) > 1 else 0
    y_spread = (max(ys) - min(ys)) / h if len(ys) > 1 else 0
    if x_spread < 0.2 and y_spread < 0.2:
        return "Focal / Single-point composition"
    elif x_spread > 0.6:
        return "Horizontal / Panoramic composition"
    elif y_spread > 0.6:
        return "Vertical / Hierarchical composition"
    else:
        return "Balanced / Multi-figure composition"


# ─────────────────────────────────────────────────────────────────────────────
# ALGORITHM 2 — GLCM Texture & Brushstroke Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_texture(img: np.ndarray) -> dict:
    """
    Gray-Level Co-occurrence Matrix (GLCM) for brushstroke texture analysis.
    Calculates contrast, dissimilarity, homogeneity, energy, correlation,
    and entropy. Infers likely artistic movement from texture properties.

    Args:
        img: BGR numpy array

    Returns:
        dict with GLCM metrics and style inference
    """
    try:
        gray = rgb2gray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # Quantise to 64 grey levels for speed
        gray_uint = (gray * 63).astype(np.uint8)

        distances  = [1, 3, 5]
        angles     = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        glcm       = graycomatrix(gray_uint, distances=distances, angles=angles,
                                  levels=64, symmetric=True, normed=True)

        props = {}
        for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation"):
            vals = graycoprops(glcm, prop).flatten()
            props[prop] = {
                "mean": round(float(np.mean(vals)), 5),
                "std":  round(float(np.std(vals)), 5),
            }

        # Shannon entropy on the full GLCM
        glcm_sum = glcm.sum(axis=(2, 3))
        glcm_norm = glcm_sum / glcm_sum.sum()
        entropy_val = float(-np.sum(glcm_norm * np.log2(glcm_norm + 1e-10)))

        # Infer style from contrast + entropy thresholds
        contrast_mean = props["contrast"]["mean"]
        style_label, style_desc = _infer_style_from_glcm(contrast_mean, entropy_val)

        # Directional dominance (0°, 45°, 90°, 135°)
        angle_contrasts = [
            float(graycoprops(glcm[:, :, :, i:i+1], "contrast").mean())
            for i in range(len(angles))
        ]
        dominant_angle_idx = int(np.argmax(angle_contrasts))
        angle_labels = ["Horizontal (0°)", "Diagonal (45°)",
                        "Vertical (90°)", "Anti-diagonal (135°)"]

        return {
            "glcm_properties": props,
            "entropy": round(entropy_val, 4),
            "inferred_style": style_label,
            "style_description": style_desc,
            "dominant_stroke_direction": angle_labels[dominant_angle_idx],
            "roughness_index": round(contrast_mean / (entropy_val + 1e-6), 4),
        }

    except Exception as e:
        return {"error": str(e), "entropy": 0.0, "inferred_style": "Unknown"}


def _infer_style_from_glcm(contrast, entropy):
    if contrast > 150 and entropy > 4.5:
        return "Impressionist / Post-Impressionist", (
            "High contrast and entropy indicate thick, varied brushwork — "
            "hallmark of Monet, Van Gogh, and Cézanne."
        )
    elif contrast > 80 and entropy > 3.5:
        return "Baroque / Romantic", (
            "Moderate-high contrast suggests chiaroscuro-style tonal drama, "
            "consistent with Rembrandt or Delacroix."
        )
    elif contrast < 40 and entropy < 3.0:
        return "Renaissance / Neoclassical", (
            "Low contrast and smooth texture signal sfumato or glazing techniques, "
            "common in Leonardo or Raphael."
        )
    elif entropy > 5.0:
        return "Abstract Expressionist", (
            "Extremely high entropy reflects chaotic, gestural mark-making "
            "as seen in Pollock or de Kooning."
        )
    else:
        return "Academic / Realist", (
            "Balanced texture metrics suggest controlled, measured brushwork "
            "typical of academic realism."
        )


# ─────────────────────────────────────────────────────────────────────────────
# ALGORITHM 3 — K-Means Pigment & Palette Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_palette(img: np.ndarray, k: int = 5) -> dict:
    """
    K-Means clustering on the pixel colour space to extract dominant pigments.

    Args:
        img: BGR numpy array
        k:   number of pigment clusters

    Returns:
        dict with hex codes, RGB values, percentages, and pigment identifications
    """
    try:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        # Subsample for performance (max 50 k pixels)
        pixels = rgb.reshape(-1, 3).astype(np.float32)
        if len(pixels) > 50_000:
            idx = np.random.choice(len(pixels), 50_000, replace=False)
            pixels = pixels[idx]

        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(pixels)

        centres   = km.cluster_centers_.astype(int)
        labels    = km.labels_
        counts    = np.bincount(labels, minlength=k)
        total     = counts.sum()
        order     = np.argsort(-counts)  # descending by frequency

        palette = []
        for rank, idx in enumerate(order):
            r, g, b  = centres[idx]
            pct      = round(100 * counts[idx] / total, 2)
            hex_code = f"#{r:02X}{g:02X}{b:02X}"
            pigment  = _identify_pigment(r, g, b)
            palette.append({
                "rank": rank + 1,
                "hex": hex_code,
                "rgb": [int(r), int(g), int(b)],
                "percentage": pct,
                "pigment_id": pigment,
            })

        # Build a small swatch visualisation (100-px tall strips)
        swatch_w = w
        swatch   = np.zeros((100, swatch_w, 3), dtype=np.uint8)
        x = 0
        for entry in palette:
            strip_w = max(1, int(swatch_w * entry["percentage"] / 100))
            r, g, b = entry["rgb"]
            swatch[:, x : x + strip_w] = [b, g, r]  # BGR
            x += strip_w

        warmth   = _compute_warmth(palette)
        luminance = _compute_luminance(palette)

        return {
            "palette": palette,
            "palette_swatch": swatch,
            "dominant_hex": palette[0]["hex"],
            "palette_warmth": warmth,
            "average_luminance": luminance,
            "historical_pigments": [p["pigment_id"] for p in palette],
        }

    except Exception as e:
        return {"error": str(e), "palette": []}


def _identify_pigment(r, g, b):
    """Heuristic mapping from RGB to historical pigment names."""
    h, s, v = _rgb_to_hsv(r, g, b)
    if v < 30:
        return "Ivory Black / Lamp Black"
    if v > 220 and s < 30:
        return "Lead White / Titanium White"
    if h < 15 or h > 345:
        return "Vermillion / Cadmium Red"
    if 15 <= h < 40:
        return "Yellow Ochre / Raw Sienna"
    if 40 <= h < 65:
        return "Naples Yellow / Cadmium Yellow"
    if 65 <= h < 150:
        return "Viridian Green / Terre Verte"
    if 150 <= h < 200 and s > 80:
        return "Cobalt Blue / Cerulean"
    if 200 <= h < 260 and s > 100:
        return "Lapis Lazuli / Ultramarine"
    if 260 <= h < 300:
        return "Manganese Violet / Dioxazine"
    if 300 <= h < 345:
        return "Rose Madder / Alizarin Crimson"
    if s < 40:
        return "Raw Umber / Burnt Sienna"
    return "Mixed Compound Pigment"


def _rgb_to_hsv(r, g, b):
    arr = np.array([[[r, g, b]]], dtype=np.uint8)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)[0, 0]
    return int(hsv[0] * 2), int(hsv[1]), int(hsv[2])


def _compute_warmth(palette):
    warm = sum(p["percentage"] for p in palette
               if _is_warm(*p["rgb"]))
    cool = 100 - warm
    if warm > 60:
        return "Predominantly Warm"
    elif cool > 60:
        return "Predominantly Cool"
    else:
        return "Balanced / Neutral"


def _is_warm(r, g, b):
    return r > b  # simplistic but effective heuristic


def _compute_luminance(palette):
    lum = sum(
        (0.2126 * p["rgb"][0] + 0.7152 * p["rgb"][1] + 0.0722 * p["rgb"][2])
        * p["percentage"] / 100
        for p in palette
    )
    return round(lum, 2)


# ─────────────────────────────────────────────────────────────────────────────
# ALGORITHM 4 — ViT / ResNet Style & Era Classifier
# ─────────────────────────────────────────────────────────────────────────────

# Map ImageNet categories to artistic eras (best-effort heuristic)
_ERA_MAP = {
    "mask":            ("Medieval / Byzantine",     "Flat gold leaf style with symbolic figures."),
    "shield":          ("Medieval / Byzantine",     "Heraldic imagery prevalent before the Renaissance."),
    "robe":            ("Renaissance",              "Classical drapery; linear perspective mastered."),
    "toga":            ("Neoclassical",             "Revival of Greco-Roman ideals in the 18th century."),
    "kimono":          ("Japonism / Orientalism",   "Eastern influence on Western art circa 1860–1900."),
    "comic_book":      ("Pop Art",                  "Mass-culture imagery celebrated by Warhol and Lichtenstein."),
    "wool":            ("Impressionism",            "Texture and light over precise line; plein-air painting."),
    "bucket":          ("Abstract",                 "Form and colour decoupled from representational reality."),
    "envelope":        ("Realism",                  "Trompe-l'œil tradition in 19th-century academic painting."),
}

_DEFAULT_ERAS = [
    ("Baroque",              "Dramatic light, rich colour, and grandeur (17th century)."),
    ("Impressionism",        "Broken brushwork capturing transient light (1860–1900)."),
    ("Post-Impressionism",   "Structural colour and personal vision beyond Impressionism."),
    ("Renaissance",          "Harmony, perspective, and humanist ideals (15th–16th c.)."),
    ("Romanticism",          "Emotion, nature, and the sublime (early 19th century)."),
    ("Modernism",            "Radical experimentation breaking from academic tradition."),
    ("Neoclassicism",        "Rational clarity inspired by antiquity (18th century)."),
    ("Abstract Expressionism","Raw emotional gestural abstraction (mid-20th century)."),
]


def classify_style(img: np.ndarray) -> dict:
    """
    Uses a pre-trained ResNet-50 as a feature extractor + heuristic mapping
    to identify artistic era and style.

    Args:
        img: BGR numpy array

    Returns:
        dict with era label, description, confidence, and top-5 features
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model  = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.eval().to(device)

        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor  = preprocess(rgb_img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]

        top5_vals, top5_idx = torch.topk(probs, 5)
        top5 = [
            {
                "imagenet_class": models.ResNet50_Weights.IMAGENET1K_V2.meta[
                    "categories"][idx],
                "probability": round(float(val), 4),
            }
            for idx, val in zip(top5_idx.tolist(), top5_vals.tolist())
        ]

        # Map to artistic era
        era_label, era_desc = _map_features_to_era(top5, img)
        confidence = round(float(top5_vals[0]), 3)

        return {
            "era_label": era_label,
            "era_description": era_desc,
            "confidence": confidence,
            "top5_imagenet_features": top5,
            "device_used": device,
        }

    except Exception as e:
        return {
            "era_label": "Analysis unavailable",
            "era_description": str(e),
            "confidence": 0.0,
            "top5_imagenet_features": [],
        }


def _map_features_to_era(top5, img):
    for feat in top5:
        cls_lower = feat["imagenet_class"].lower().replace(" ", "_")
        for key, (era, desc) in _ERA_MAP.items():
            if key in cls_lower:
                return era, desc

    # Fallback: use colour + texture heuristics
    h, w = img.shape[:2]
    brightness = img.mean()
    saturation = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1].mean()

    if brightness < 80:
        return _DEFAULT_ERAS[0]  # Baroque
    if saturation > 130:
        return _DEFAULT_ERAS[1]  # Impressionism
    if brightness > 180 and saturation < 80:
        return _DEFAULT_ERAS[3]  # Renaissance
    return _DEFAULT_ERAS[np.random.randint(0, len(_DEFAULT_ERAS))]
