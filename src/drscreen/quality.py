"""Heuristic image-quality gate: good / enhance / reject, with a reason.

IMPORTANT: this is a rule-based gate built from classical image statistics (blur, illumination, contrast,
field-of-view). It is NOT a model trained on genuine quality labels (e.g. EyeQ), and its thresholds are
distribution-relative (fitted on training-set percentiles) rather than clinically validated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .preprocess import ImageError, circle_mask, retina_mask, validate_array

QUALITY_SIZE = 512  # all measurements are taken at this size so they do not depend on camera resolution


@dataclass
class QualityThresholds:
    min_fov_fraction: float = 0.30      # retina must cover this fraction of the (cropped, square) frame
    min_fov_completeness: float = 0.60  # retina area / area of the full circle its largest side implies; low => retina badly cut off
    reject_blur: float = 8.0            # variance of Laplacian (green channel, inside FOV)
    reject_brightness_low: float = 25.0
    enhance_brightness_low: float = 45.0
    reject_brightness_high: float = 230.0
    enhance_contrast_low: float = 22.0
    reject_contrast_low: float = 10.0
    max_saturated_fraction: float = 0.25
    fitted_from: str = "defaults (not fitted)"

    def to_dict(self):
        return asdict(self)


@dataclass
class QualityResult:
    status: str                 # good | enhance | reject
    score: float                # 0 (unusable) .. 1 (clearly fine); heuristic, not a calibrated probability
    reason: str
    metrics: dict = field(default_factory=dict)
    method: str = "heuristic rules (not a trained quality model)"

    def to_dict(self):
        return asdict(self)


def compute_quality_metrics(img: np.ndarray) -> dict:
    """Blur / illumination / contrast / field-of-view measurements on a size-normalised copy of the image."""
    validate_array(img)
    h, w = img.shape[:2]
    s = QUALITY_SIZE / max(h, w)
    small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    mask = retina_mask(small, thr=15)
    fov_fraction = float(mask.mean())
    if mask.sum() < 100:
        return {"fov_fraction": fov_fraction, "fov_completeness": 0.0, "blur": 0.0, "brightness": float(small.mean()), "contrast": 0.0,
                "saturated_fraction": 0.0, "retina_found": False}
    ys, xs = np.where(mask)
    bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    completeness = float(mask.sum() / (np.pi * (max(bw, bh) / 2.0) ** 2))  # 1.0 for a full circle
    fov_in_frame = float(mask.sum() / (max(bw, bh) ** 2))  # retina area / area of its square crop
    inner = cv2.erode(mask.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)  # avoid the strong border edge
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    green = small[..., 1]
    lap = cv2.Laplacian(cv2.GaussianBlur(green, (0, 0), 1.0), cv2.CV_64F)
    sel = inner if inner.sum() > 100 else mask
    return {
        "fov_fraction": fov_in_frame, "fov_completeness": completeness, "retina_found": True,
        "blur": float(lap[sel].var()), "brightness": float(gray[sel].mean()), "contrast": float(gray[sel].std()),
        "saturated_fraction": float((gray[sel] > 245).mean()),
    }


def assess_quality(img, thr: QualityThresholds | None = None) -> QualityResult:
    """Return status good/enhance/reject with a reason. Never raises on bad input: invalid images are rejected."""
    thr = thr or QualityThresholds()
    try:
        m = compute_quality_metrics(img)
    except (ImageError, cv2.error, ValueError) as e:
        return QualityResult("reject", 0.0, f"image is missing or invalid ({e}); please recapture or re-upload", {})
    if not m["retina_found"] or m["fov_fraction"] < thr.min_fov_fraction:
        return QualityResult("reject", 0.0, "no retina field of view found (image mostly black or not a fundus photograph)", m)
    if m["fov_completeness"] < thr.min_fov_completeness:
        return QualityResult("reject", 0.1, "retina field of view is heavily cut off; please recapture with the eye centred", m)
    if m["blur"] < thr.reject_blur:
        return QualityResult("reject", 0.1, f"image is too blurry / out of focus (sharpness {m['blur']:.1f} < {thr.reject_blur:.1f}); please recapture", m)
    if m["brightness"] < thr.reject_brightness_low:
        return QualityResult("reject", 0.1, "image is far too dark to assess; please recapture with better illumination", m)
    if m["brightness"] > thr.reject_brightness_high or m["saturated_fraction"] > thr.max_saturated_fraction:
        return QualityResult("reject", 0.1, "image is over-exposed / saturated; please recapture", m)
    if m["contrast"] < thr.reject_contrast_low:
        return QualityResult("reject", 0.15, "image has almost no contrast (hazy); please recapture", m)
    # sub-scores in [0,1], 1 = comfortably fine
    sub = {
        "blur": np.clip(m["blur"] / (3 * thr.reject_blur), 0, 1),
        "brightness": np.clip((m["brightness"] - thr.reject_brightness_low) / (thr.enhance_brightness_low - thr.reject_brightness_low + 1e-6), 0, 1),
        "contrast": np.clip((m["contrast"] - thr.reject_contrast_low) / (thr.enhance_contrast_low - thr.reject_contrast_low + 1e-6), 0, 1),
    }
    score = float(min(sub.values()))
    problems = []
    if m["brightness"] < thr.enhance_brightness_low:
        problems.append("low illumination")
    if m["contrast"] < thr.enhance_contrast_low:
        problems.append("low contrast")
    if problems:
        return QualityResult("enhance", min(score, 0.6), " and ".join(problems) + "; contrast enhancement (CLAHE) applied before analysis", m)
    return QualityResult("good", max(score, 0.6), "sharpness, illumination, contrast and field of view are acceptable", m)


def enhance_image(img: np.ndarray, clip: float = 2.0) -> np.ndarray:
    """CLAHE on the L channel of LAB (local contrast / illumination correction)."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def fit_thresholds(metrics_list: list[dict], reject_pct: float = 1.0, enhance_pct: float = 5.0) -> QualityThresholds:
    """Set distribution-relative thresholds from TRAINING-set quality metrics only (never from val/test).

    reject_blur = reject_pct-th percentile of sharpness; brightness/contrast 'enhance' at enhance_pct-th percentile and
    'reject' at the reject_pct-th percentile / 2. Everything else keeps the defaults.
    """
    ok = [m for m in metrics_list if m.get("retina_found")]
    t = QualityThresholds()
    if len(ok) < 50:
        return t
    q = lambda k, p: float(np.percentile([m[k] for m in ok], p))
    t.reject_blur = max(2.0, q("blur", reject_pct))
    t.enhance_brightness_low = q("brightness", enhance_pct)
    t.reject_brightness_low = min(t.enhance_brightness_low * 0.6, q("brightness", reject_pct / 2))
    t.enhance_contrast_low = q("contrast", enhance_pct)
    t.reject_contrast_low = min(t.enhance_contrast_low * 0.5, q("contrast", reject_pct / 2))
    t.fitted_from = f"{len(ok)} training images; reject at p{reject_pct:g}, enhance at p{enhance_pct:g}"
    return t


# --------------------------------------------------------------------------- synthetic degradations (evaluation only)
DEGRADATIONS = ("blur", "dark", "overexposed", "low_contrast", "jpeg", "cropped_fov")


def degrade(img: np.ndarray, kind: str, severity: int = 2, seed: int = 0) -> np.ndarray:
    """Apply a synthetic degradation (severity 1..3) to an RGB uint8 image. Used to test the gate and robustness."""
    sev = {1: 0, 2: 1, 3: 2}[severity]
    h, w = img.shape[:2]
    if kind == "blur":
        sigma = (3.0, 6.0, 10.0)[sev] * max(h, w) / 512
        return cv2.GaussianBlur(img, (0, 0), sigma)
    if kind == "dark":
        f = (0.55, 0.35, 0.18)[sev]
        return np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)
    if kind == "overexposed":
        return np.clip(img.astype(np.float32) * (1.6, 2.2, 3.0)[sev] + (10, 30, 60)[sev], 0, 255).astype(np.uint8)
    if kind == "low_contrast":
        f = (0.55, 0.35, 0.18)[sev]
        return np.clip((img.astype(np.float32) - 128) * f + 128, 0, 255).astype(np.uint8)
    if kind == "jpeg":
        q = (30, 12, 5)[sev]
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
        return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if kind == "cropped_fov":
        out = img.copy()
        cut = (0.15, 0.30, 0.45)[sev]
        out[: int(h * cut)] = 0  # simulate the retina being cut off at the top
        return out
    raise ValueError(f"unknown degradation {kind}")
