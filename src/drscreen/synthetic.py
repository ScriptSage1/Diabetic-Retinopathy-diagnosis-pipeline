"""Synthetic fundus-like images for unit tests and local smoke runs ONLY.

These are procedurally generated (reddish disc, vessel-like curves, bright optic disc, grade-correlated
dots). They are NOT real retinal images and no result obtained on them is ever reported as a
performance result. Their sole purpose is to exercise code paths without needing the real dataset.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def make_fundus(grade: int = 0, size: int = 320, seed: int = 0, border: bool = True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = size
    w = int(size * (1.25 if border else 1.0))
    img = np.zeros((h, w, 3), np.uint8)
    cx, cy, r = w // 2, h // 2, int(h * 0.46)
    base = np.array([170, 70, 40], np.float32) * rng.uniform(0.7, 1.15)
    yy, xx = np.mgrid[:h, :w]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
    shade = np.clip(1.15 - 0.45 * d, 0, 1.2)[..., None]
    retina = (base[None, None] * shade).astype(np.float32)
    retina += rng.normal(0, 4, retina.shape)
    disc = d <= 1.0
    img[disc] = np.clip(retina[disc], 0, 255).astype(np.uint8)
    # vessels
    for _ in range(14):
        ang = rng.uniform(0, 2 * np.pi)
        pts = []
        x0, y0 = cx + r * 0.35, cy - 0.1 * r
        for t in np.linspace(0, 1, 12):
            a = ang + 0.5 * np.sin(6 * t + rng.uniform(0, 3))
            pts.append((int(cx + (0.1 + 0.9 * t) * r * np.cos(a)), int(cy + (0.1 + 0.9 * t) * r * np.sin(a))))
        cv2.polylines(img, [np.array(pts, np.int32)], False, (110, 25, 20), int(rng.integers(1, 3)))
    # optic disc
    cv2.circle(img, (int(cx + 0.45 * r), cy), max(3, int(0.12 * r)), (235, 215, 150), -1)
    # grade-correlated lesions: red dots and bright exudates
    n_red = [0, 3, 12, 30, 45][grade]
    n_ex = [0, 0, 5, 14, 20][grade]
    for _ in range(n_red):
        a, rad = rng.uniform(0, 2 * np.pi), rng.uniform(0, 0.85) * r
        cv2.circle(img, (int(cx + rad * np.cos(a)), int(cy + rad * np.sin(a))), int(rng.integers(1, 4)), (90, 10, 10), -1)
    for _ in range(n_ex):
        a, rad = rng.uniform(0, 2 * np.pi), rng.uniform(0, 0.8) * r
        cv2.circle(img, (int(cx + rad * np.cos(a)), int(cy + rad * np.sin(a))), int(rng.integers(2, 5)), (240, 225, 140), -1)
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    return img


def make_synthetic_dataset(root: str | Path, n_per_grade=(12, 6, 10, 6, 6), size: int = 224, seed: int = 0) -> pd.DataFrame:
    """Write PNGs + a train.csv (id_code, diagnosis) mimicking the APTOS layout. Returns the label table."""
    root = Path(root)
    (root / "train_images").mkdir(parents=True, exist_ok=True)
    rows = []
    k = 0
    for g, n in enumerate(n_per_grade):
        for _ in range(n):
            code = f"syn{k:04d}"
            im = make_fundus(g, size=size, seed=seed + k)
            cv2.imwrite(str(root / "train_images" / f"{code}.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            rows.append({"id_code": code, "diagnosis": g})
            k += 1
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.to_csv(root / "train.csv", index=False)
    return df
