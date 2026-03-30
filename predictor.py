"""
Parkinson's Motor Impairment Score Predictor
=============================================
Reproduces the exact preprocessing and scoring pipeline from the training notebook.
torch / torchvision are NOT required — transforms are reimplemented in NumPy/PIL/cv2.

Models:
  - Wave    → Final_wave_VGG19.h5       (VGG19 backbone, single logit output)
  - Spiral  → Final_spiral_ResNet101.h5 (ResNet101 backbone, single logit output)
"""

import os
import numpy as np
import cv2
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download


# ──────────────────────────────────────────────────────────────────────────────
# Constants  (all values taken directly from the training notebook)
# ──────────────────────────────────────────────────────────────────────────────

HF_REPO_ID        = "xplorers/Motor_Impairment_Score_models"
WAVE_MODEL_FILE   = "Final_wave_VGG19.h5"
SPIRAL_MODEL_FILE = "Final_spiral_ResNet101.h5"

# Logit range calibrated on the training set (1st / 99th percentile)
SPIRAL_MIN_LOGIT = -16.384981
SPIRAL_MAX_LOGIT =  26.600843

WAVE_MIN_LOGIT   = -45.584194
WAVE_MAX_LOGIT   =  78.02814

# Decision boundary in 0-100 score space (where logit == 0 lands).
# Scores BELOW this → "Normal Pattern" (healthy).
SPIRAL_NORMAL_BOUNDARY = 38.117172
WAVE_NORMAL_BOUNDARY   = 36.876736


# ──────────────────────────────────────────────────────────────────────────────
# Model loading  (lazy, module-level cache)
# ──────────────────────────────────────────────────────────────────────────────

_wave_model   = None
_spiral_model = None


def _get_wave_model() -> tf.keras.Model:
    global _wave_model
    if _wave_model is None:
        print("[parkinson_predictor] Downloading wave model (VGG19)…")
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=WAVE_MODEL_FILE)
        _wave_model = load_model(path)
        print("[parkinson_predictor] Wave model ready ✓")
    return _wave_model


def _get_spiral_model() -> tf.keras.Model:
    global _spiral_model
    if _spiral_model is None:
        print("[parkinson_predictor] Downloading spiral model (ResNet101)…")
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=SPIRAL_MODEL_FILE)
        _spiral_model = load_model(path)
        print("[parkinson_predictor] Spiral model ready ✓")
    return _spiral_model


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing  — replicates the notebook's torchvision transforms exactly,
#                  but using only NumPy, PIL, and cv2.
#
# Original torchvision pipeline:
#   Grayscale(num_output_channels=3)
#   Resize((224, 224))
#   ToTensor()                          # uint8 [0,255] → float32 [0,1]
#   Normalize(mean=[0.5,0.5,0.5],       # [0,1] → [-1,1]
#             std=[0.5,0.5,0.5])
# Then:
#   tensor.permute(1,2,0)               # CHW → HWC
#   (img * 255).astype(uint8)           # [-1,1] float → [0,255] uint8  ← notebook quirk
#
# The net result of ToTensor + Normalize + ×255:
#   output = (pixel/255.0 - 0.5) / 0.5 * 255
#           = (pixel - 127.5)
# So the final uint8 array is a simple mean-subtraction by 127.5 (clipped to uint8).
# ──────────────────────────────────────────────────────────────────────────────

def _to_numpy_bgr(source) -> np.ndarray:
    """Convert any image source to a BGR uint8 numpy array."""
    if isinstance(source, (str, os.PathLike)):
        img = cv2.imread(str(source))
        if img is None:
            raise FileNotFoundError(f"cv2.imread could not open: {source}")
        return img
    if isinstance(source, (bytes, bytearray)):
        arr = np.frombuffer(source, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if isinstance(source, Image.Image):
        rgb = np.array(source.convert("RGB"), dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # File-like object (e.g. Flask request.files['img'])
    raw = source.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def preprocess_image(source) -> np.ndarray:
    """
    Full preprocessing pipeline — identical to the notebook's preprocess_image(),
    but without torch/torchvision:

      1. Load image as BGR
      2. Convert to grayscale
      3. Otsu binarisation with inversion  (THRESH_BINARY_INV | THRESH_OTSU)
      4. Resize to 224×224
      5. Stack to 3-channel (grayscale → RGB)
      6. Normalize: pixel → (pixel − 127.5)  [equivalent to ToTensor+Normalize×255]
      7. Clip and cast to uint8
      8. Add batch dimension → shape (1, 224, 224, 3)

    Parameters
    ----------
    source : str | bytes | file-like | PIL.Image.Image

    Returns
    -------
    np.ndarray  shape (1, 224, 224, 3)  dtype uint8
    """
    # Step 1-3: load, grayscale, binarise
    bgr  = _to_numpy_bgr(source)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, binarised = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )

    # Step 4: resize
    resized = cv2.resize(binarised, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Step 5: grayscale → 3-channel (same as Grayscale(num_output_channels=3))
    rgb = np.stack([resized, resized, resized], axis=-1)   # (224, 224, 3) uint8

    # Step 6-7: replicate ToTensor() → Normalize(0.5,0.5) → ×255
    #   ToTensor:   x = pixel / 255.0          → [0, 1]
    #   Normalize:  x = (x - 0.5) / 0.5        → [-1, 1]
    #   ×255:       x = x * 255                → [-255, 255]
    #   Combined:   x = pixel - 127.5
    img_np = rgb.astype(np.float32) - 127.5                # (224, 224, 3) float32
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)      # (224, 224, 3) uint8

    # Step 8: batch dimension
    return np.expand_dims(img_np, axis=0)                  # (1, 224, 224, 3)


# ──────────────────────────────────────────────────────────────────────────────
# Severity interpretation  (from the notebook's interpret_severity functions)
# ──────────────────────────────────────────────────────────────────────────────

def _interpret_spiral_severity(score: float) -> tuple:
    if score < SPIRAL_NORMAL_BOUNDARY:      # < 38.117172
        return "Normal Pattern", "No motor impairment detected."
    elif score < 55:
        return "Mild",     "Slight motor irregularities observed."
    elif score < 70:
        return "Moderate", "Noticeable motor impairment detected."
    elif score < 85:
        return "High",     "Significant motor impairment observed."
    else:
        return "Severe",   "Strong Parkinsonian motor patterns detected."


def _interpret_wave_severity(score: float) -> tuple:
    if score < WAVE_NORMAL_BOUNDARY:        # < 36.876736
        return "Normal Pattern", "No motor impairment detected."
    elif score < 55:
        return "Mild",     "Slight motor irregularities observed."
    elif score < 70:
        return "Moderate", "Noticeable motor impairment detected."
    elif score < 85:
        return "High",     "Significant motor impairment observed."
    else:
        return "Severe",   "Strong Parkinsonian motor patterns detected."


# ──────────────────────────────────────────────────────────────────────────────
# Public prediction functions
# ──────────────────────────────────────────────────────────────────────────────

def predict_wave(image_source) -> dict:
    """
    Classify a **wave drawing** and return the motor impairment score.

    The VGG19 model outputs a single raw logit (no sigmoid activation).
    The logit is normalised into a 0-100 motor impairment score:

        score = clip( (logit - MIN) / (MAX - MIN), 0, 1 ) × 100

    Scores below 36.88 → "Normal Pattern" (no Parkinson's detected).

    Parameters
    ----------
    image_source : str | bytes | file-like | PIL.Image.Image
        predict_wave("path/to/wave.png")
        predict_wave(open("wave.png", "rb").read())
        predict_wave(pil_image)
        predict_wave(flask_request_files_obj)

    Returns
    -------
    dict
        {
          "drawing_type"           : "wave",
          "raw_logit"              : float,
          "sigmoid_probability"    : float,   # P(Parkinson's) in [0, 1]
          "motor_impairment_score" : float,   # normalised score in [0, 100]
          "severity_level"         : str,     # "Normal Pattern" | "Mild" |
                                              # "Moderate" | "High" | "Severe"
          "description"            : str,
          "is_parkinson"           : bool
        }

    Example
    -------
    >>> result = predict_wave("patient_wave.png")
    >>> print(result["motor_impairment_score"])   # e.g. 72.4
    >>> print(result["severity_level"])           # "High"
    """
    model  = _get_wave_model()
    tensor = preprocess_image(image_source)

    logit        = float(model.predict(tensor, verbose=0)[0][0])
    sigmoid_prob = float(1.0 / (1.0 + np.exp(-logit)))

    normalized = (logit - WAVE_MIN_LOGIT) / (WAVE_MAX_LOGIT - WAVE_MIN_LOGIT)
    score      = round(float(np.clip(normalized, 0.0, 1.0)) * 100, 2)

    level, description = _interpret_wave_severity(score)

    return {
        "drawing_type"           : "wave",
        "raw_logit"              : round(logit, 4),
        "sigmoid_probability"    : round(sigmoid_prob, 4),
        "motor_impairment_score" : score,
        "severity_level"         : level,
        "description"            : description,
        "is_parkinson"           : level != "Normal Pattern",
    }


def predict_spiral(image_source) -> dict:
    """
    Classify a **spiral drawing** and return the motor impairment score.

    The ResNet101 model outputs a single raw logit (no sigmoid activation).
    Same normalisation as predict_wave():

        score = clip( (logit - MIN) / (MAX - MIN), 0, 1 ) × 100

    Scores below 38.12 → "Normal Pattern" (no Parkinson's detected).

    Parameters
    ----------
    image_source : str | bytes | file-like | PIL.Image.Image
        Same flexible input types as predict_wave().

    Returns
    -------
    dict  (identical structure to predict_wave, with "drawing_type": "spiral")

    Example
    -------
    >>> result = predict_spiral("patient_spiral.png")
    >>> print(result["motor_impairment_score"])   # e.g. 61.8
    >>> print(result["severity_level"])           # "Moderate"
    """
    model  = _get_spiral_model()
    tensor = preprocess_image(image_source)

    logit        = float(model.predict(tensor, verbose=0)[0][0])
    sigmoid_prob = float(1.0 / (1.0 + np.exp(-logit)))

    normalized = (logit - SPIRAL_MIN_LOGIT) / (SPIRAL_MAX_LOGIT - SPIRAL_MIN_LOGIT)
    score      = round(float(np.clip(normalized, 0.0, 1.0)) * 100, 2)

    level, description = _interpret_spiral_severity(score)

    return {
        "drawing_type"           : "spiral",
        "raw_logit"              : round(logit, 4),
        "sigmoid_probability"    : round(sigmoid_prob, 4),
        "motor_impairment_score" : score,
        "severity_level"         : level,
        "description"            : description,
        "is_parkinson"           : level != "Normal Pattern",
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI demo:  python parkinson_predictor.py <wave|spiral> <image_path>
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, json

    if len(sys.argv) < 3:
        print("Usage:  python parkinson_predictor.py <wave|spiral> <image_path>")
        sys.exit(1)

    draw_type  = sys.argv[1].lower()
    image_path = sys.argv[2]

    if draw_type == "wave":
        result = predict_wave(image_path)
    elif draw_type == "spiral":
        result = predict_spiral(image_path)
    else:
        print("First argument must be 'wave' or 'spiral'.")
        sys.exit(1)

    print(json.dumps(result, indent=2))