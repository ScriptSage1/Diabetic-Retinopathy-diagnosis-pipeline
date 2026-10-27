"""Fundus image reading, validation, border cropping, circular masking, resizing, normalisation.

Pipeline used everywhere (training cache, quality gate, app):

    read -> validate -> crop black border -> pad to square -> resize -> circular FOV mask   ("plain")
    plain -> (optional) Ben Graham colour normalisation                                     ("model input")
    model input -> float32 CHW, ImageNet mean/std                                           ("tensor")
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


class ImageError(ValueError):
    """Raised when an image cannot be read or is unusable."""


def read_image(path: str | Path) -> np.ndarray:
    """Read an image file as RGB uint8. Raises ImageError for missing/corrupt/unsupported files."""
    p = Path(path)
    if not p.is_file():
        raise ImageError(f"file not found: {p}")
    try:
        data = np.fromfile(str(p), dtype=np.uint8)  # np.fromfile handles unicode paths on Windows
    except OSError as e:
        raise ImageError(f"cannot read {p}: {e}") from e
    if data.size == 0:
        raise ImageError(f"empty file: {p}")
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ImageError(f"cannot decode image (corrupt or unsupported format): {p}")
    return validate_array(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def validate_array(img: np.ndarray, min_side: int = 64) -> np.ndarray:
    """Check an RGB uint8 array is plausible as a photo; return it unchanged or raise ImageError."""
    if not isinstance(img, np.ndarray):
        raise ImageError("image is not a numpy array")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ImageError(f"expected HxWx3 image, got shape {img.shape}")
    if img.dtype != np.uint8:
        raise ImageError(f"expected uint8 image, got {img.dtype}")
    if min(img.shape[:2]) < min_side:
        raise ImageError(f"image too small: {img.shape[1]}x{img.shape[0]} (min side {min_side})")
    if int(img.max()) == int(img.min()):
        raise ImageError("image is a constant colour")
    return img


def to_rgb_uint8(obj) -> np.ndarray:
    """Coerce a PIL image / array (gray, RGBA, float 0-1, uint16) to RGB uint8; used for app inputs."""
    if obj is None:
        raise ImageError("no image provided")
    arr = np.asarray(obj)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ImageError(f"unsupported image shape {arr.shape}")
    if arr.dtype != np.uint8:
        a = arr.astype(np.float64)
        if np.issubdtype(arr.dtype, np.floating) and a.max() <= 1.0:
            a = a * 255.0
        elif arr.dtype == np.uint16:
            a = a / 257.0
        arr = np.clip(a, 0, 255).astype(np.uint8)
    return validate_array(np.ascontiguousarray(arr))


def retina_mask(img: np.ndarray, thr: int = 10) -> np.ndarray:
    """Boolean mask of non-black (retina) pixels, cleaned with a morphological opening."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = (gray > thr).astype(np.uint8)
    k = max(3, int(min(img.shape[:2]) / 100) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    return mask.astype(bool)


def crop_black_borders(img: np.ndarray, thr: int = 10, min_area_frac: float = 0.05) -> np.ndarray:
    """Crop to the bounding box of the retina. Returns the input unchanged if no retina is found."""
    mask = retina_mask(img, thr)
    if mask.mean() < min_area_frac:
        return img
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return img[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def pad_to_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    return cv2.copyMakeBorder(img, top, s - h - top, left, s - w - left, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def circle_mask(size: int, radius_frac: float = 0.98) -> np.ndarray:
    """Boolean size x size mask of a centred disc."""
    yy, xx = np.ogrid[:size, :size]
    c = (size - 1) / 2.0
    return (yy - c) ** 2 + (xx - c) ** 2 <= (radius_frac * size / 2.0) ** 2


def preprocess_plain(img: np.ndarray, size: int = 384, radius_frac: float = 0.98) -> np.ndarray:
    """crop black border -> pad square -> resize -> circular FOV mask. Returns size x size x 3 uint8."""
    validate_array(img)
    out = pad_to_square(crop_black_borders(img))
    interp = cv2.INTER_AREA if out.shape[0] >= size else cv2.INTER_CUBIC
    out = cv2.resize(out, (size, size), interpolation=interp)
    out = out.copy()
    out[~circle_mask(size, radius_frac)] = 0
    return out


def ben_graham(img: np.ndarray, sigma_frac: float = 1 / 30, radius_frac: float = 0.94) -> np.ndarray:
    """Ben Graham local-colour normalisation: 4*I - 4*GaussianBlur(I) + 128, outside the FOV set to 128.

    sigma = sigma_frac * image size (so the same relative scale at any resolution).
    """
    size = img.shape[0]
    sigma = max(1.0, sigma_frac * size)
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    out = cv2.addWeighted(img, 4, blur, -4, 128)
    out[~circle_mask(size, radius_frac)] = 128
    return out


def to_model_input(plain: np.ndarray, use_ben_graham: bool = True) -> np.ndarray:
    """Apply the model-input colour normalisation to a plain preprocessed image."""
    return ben_graham(plain) if use_ben_graham else plain


def to_tensor_array(arr: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 -> float32 3xHxW normalised with ImageNet statistics."""
    x = arr.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def preprocess_for_model(img: np.ndarray, size: int = 384, use_ben_graham: bool = True):
    """Full inference-time pipeline. Returns (plain_uint8, model_input_uint8, tensor_float32_CHW)."""
    plain = preprocess_plain(img, size)
    inp = to_model_input(plain, use_ben_graham)
    return plain, inp, to_tensor_array(inp)
