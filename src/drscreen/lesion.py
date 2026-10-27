"""Lesion evidence: IDRiD segmentation (MA, HE, EX, SE, optic disc), optic-disc/fovea locator, quadrant features.

Everything here is trained on IDRiD (CC BY 4.0, Porwal et al. 2018) and evaluated on IDRiD's official test split.
Applying the models to APTOS (other cameras, resolutions, populations) is an unvalidated domain transfer: APTOS has no
lesion annotations, so lesion outputs on APTOS are NEVER reported as lesion-detection performance.

Coordinate frames: ``canvas`` = the retina cropped to its bounding box, padded square and resized to CANVAS px
(so scale is normalised by retina size, not by camera resolution). ``Fov`` maps between original pixels and canvas.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocess import ben_graham, circle_mask, retina_mask, to_tensor_array

CANVAS = 1280
LESIONS = ["MA", "HE", "EX", "SE", "OD"]  # output channels of the segmentation network
FINDINGS = ["MA", "HE", "EX", "SE"]       # lesions that feed the rule engine
FOLDER = {"MA": "microaneurysms", "HE": "haemorrhages", "EX": "hard exudates", "SE": "soft exudates", "OD": "optic disc"}
SUFFIX = {"MA": "_MA", "HE": "_HE", "EX": "_EX", "SE": "_SE", "OD": "_OD"}
QUADRANTS = ["temporal-superior", "temporal-inferior", "nasal-superior", "nasal-inferior"]


# ----------------------------------------------------------------------------- geometry
@dataclass
class Fov:
    """Crop box -> square padding -> resize, remembered so points/masks can be mapped both ways."""
    r0: int
    c0: int
    ch: int      # crop height
    cw: int      # crop width
    side: int    # padded square side (= max(ch, cw))
    size: int = CANVAS

    @property
    def top(self) -> int:
        return (self.side - self.ch) // 2

    @property
    def left(self) -> int:
        return (self.side - self.cw) // 2

    @property
    def scale(self) -> float:
        return self.size / self.side

    def to_canvas(self, img: np.ndarray, mask: bool = False) -> np.ndarray:
        crop = img[self.r0 : self.r0 + self.ch, self.c0 : self.c0 + self.cw]
        pad = cv2.copyMakeBorder(crop, self.top, self.side - self.ch - self.top, self.left, self.side - self.cw - self.left,
                                 cv2.BORDER_CONSTANT, value=0)
        if mask:  # area interpolation keeps thin/small lesions; keep pixels with >= 25% coverage
            m = cv2.resize(pad.astype(np.uint8) * 255, (self.size, self.size), interpolation=cv2.INTER_AREA)
            return (m >= 64).astype(np.uint8)
        interp = cv2.INTER_AREA if self.side >= self.size else cv2.INTER_CUBIC
        return cv2.resize(pad, (self.size, self.size), interpolation=interp)

    def pts_to_canvas(self, xy) -> np.ndarray:
        xy = np.asarray(xy, float).reshape(-1, 2)
        return np.stack([(xy[:, 0] - self.c0 + self.left) * self.scale, (xy[:, 1] - self.r0 + self.top) * self.scale], 1)

    def pts_to_orig(self, xy) -> np.ndarray:
        xy = np.asarray(xy, float).reshape(-1, 2)
        return np.stack([xy[:, 0] / self.scale - self.left + self.c0, xy[:, 1] / self.scale - self.top + self.r0], 1)

    def as_array(self) -> np.ndarray:
        return np.array([self.r0, self.c0, self.ch, self.cw, self.side, self.size], np.int64)

    @staticmethod
    def from_array(a) -> "Fov":
        return Fov(*[int(v) for v in a])


def make_fov(img: np.ndarray, size: int = CANVAS) -> Fov:
    """Bounding box of the retina in `img`, padded to a square, to be resized to `size`."""
    h, w = img.shape[:2]
    m = retina_mask(img)
    if m.mean() < 0.05:
        r0, r1, c0, c1 = 0, h, 0, w
    else:
        rows, cols = np.where(m.any(1))[0], np.where(m.any(0))[0]
        r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    ch, cw = r1 - r0, c1 - c0
    side = max(ch, cw)
    return Fov(int(r0), int(c0), int(ch), int(cw), int(side), size)


def canvas_for_model(img: np.ndarray, size: int = CANVAS) -> tuple[np.ndarray, Fov]:
    """Retina-normalised canvas with Ben Graham colour normalisation (model input, uint8) and its Fov."""
    fov = make_fov(img, size)
    plain = fov.to_canvas(img)
    plain[~circle_mask(size, 0.98)] = 0
    return ben_graham(plain), fov


# ----------------------------------------------------------------------------- model
def _cbr(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class LesionUNet(nn.Module):
    """U-Net with an ImageNet-pretrained EfficientNet-B0 encoder (timm features_only) and a full-resolution output stem."""

    def __init__(self, n_out: int = len(LESIONS), encoder: str = "efficientnet_b0", pretrained: bool = True, pretrained_file: str | None = None):
        super().__init__()
        import timm

        kw = dict(features_only=True, pretrained=pretrained)
        if pretrained and pretrained_file:
            kw["pretrained_cfg_overlay"] = dict(file=str(pretrained_file))
        self.enc = timm.create_model(encoder, **kw)
        ch = self.enc.feature_info.channels()
        d = [128, 64, 48, 32]
        self.d4 = nn.Sequential(_cbr(ch[4] + ch[3], d[0]), _cbr(d[0], d[0]))
        self.d3 = nn.Sequential(_cbr(d[0] + ch[2], d[1]), _cbr(d[1], d[1]))
        self.d2 = nn.Sequential(_cbr(d[1] + ch[1], d[2]), _cbr(d[2], d[2]))
        self.d1 = nn.Sequential(_cbr(d[2] + ch[0], d[3]), _cbr(d[3], d[3]))
        self.stem = _cbr(3, 16)
        self.d0 = nn.Sequential(_cbr(d[3] + 16, 32), _cbr(32, 32))
        self.head = nn.Conv2d(32, n_out, 1)

    @staticmethod
    def _up(t, ref):
        return F.interpolate(t, size=ref.shape[2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        f = self.enc(x)
        y = self.d4(torch.cat([self._up(f[4], f[3]), f[3]], 1))
        y = self.d3(torch.cat([self._up(y, f[2]), f[2]], 1))
        y = self.d2(torch.cat([self._up(y, f[1]), f[1]], 1))
        y = self.d1(torch.cat([self._up(y, f[0]), f[0]], 1))
        s = self.stem(x)
        return self.head(self.d0(torch.cat([self._up(y, s), s], 1)))


class PointRegressor(nn.Module):
    """EfficientNet-B0 regressing (OD x, OD y, fovea x, fovea y) as fractions of the canvas."""

    def __init__(self, pretrained: bool = True, pretrained_file: str | None = None):
        super().__init__()
        import timm

        kw = dict(pretrained=pretrained, num_classes=4)
        if pretrained and pretrained_file:
            kw["pretrained_cfg_overlay"] = dict(file=str(pretrained_file))
        self.net = timm.create_model("efficientnet_b0", **kw)

    def forward(self, x):
        return torch.sigmoid(self.net(x))


# ----------------------------------------------------------------------------- IDRiD indexing
def _mask_read(path: str | None, shape_hw=None) -> np.ndarray | None:
    if not path:
        return None
    data = np.fromfile(path, np.uint8)
    m = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = m.max(axis=2)
    return (m > 0).astype(np.uint8)


def index_idrid(root: str | Path) -> dict:
    """Locate IDRiD parts under root by folder names. Returns dict with 'seg' {train,test}: {stem: {image, MA, HE, EX, SE, OD}},
    'grading' {train,test}: DataFrame(image_id, grade, dme, path), 'loc' {train,test}: DataFrame(image_id, od_x, od_y, fo_x, fo_y)."""
    seg = {"train": {}, "test": {}}
    images = {"train": {}, "test": {}}
    csvs = []
    for dp, _, files in os.walk(root):
        low = dp.replace("\\", "/").lower()
        split = "train" if "training set" in low or "training_set" in low else ("test" if "testing set" in low or "testing_set" in low else None)
        for f in files:
            p = str(Path(dp) / f)
            if f.lower().endswith(".csv"):
                csvs.append(p)
                continue
            if not f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")) or split is None:
                continue
            stem = Path(f).stem
            if "segmentation" in low and "original images" in low:
                seg[split].setdefault(stem, {})["image"] = p
            elif "segmentation" in low and "groundtruth" in low:
                for k, folder in FOLDER.items():
                    if folder in low:
                        base = stem[: -len(SUFFIX[k])] if stem.endswith(SUFFIX[k]) else stem
                        seg[split].setdefault(base, {})[k] = p
            elif "grading" in low and "original images" in low:
                images[split]["grading", stem] = p
            elif "localization" in low and "original images" in low:
                images[split]["loc", stem] = p
    if not csvs:
        top = sorted(x.name for x in Path(root).iterdir())[:20] if Path(root).is_dir() else []
        print(f"[idrid] WARNING: no CSV files found under {root}; top-level entries: {top}", flush=True)
    grading, loc = {}, {}
    for p in csvs:
        low = p.replace("\\", "/").lower()
        split = "train" if "training" in Path(p).name.lower() else ("test" if "testing" in Path(p).name.lower() else None)
        if split is None:
            continue
        df = pd.read_csv(p)
        df.columns = [str(c).strip() for c in df.columns]
        low_cols = {c.lower(): c for c in df.columns}
        print(f"[idrid] {Path(p).name}: columns={list(df.columns)} rows={len(df)}", flush=True)
        if "grading" in low:
            idc = next(c for l, c in low_cols.items() if "image" in l)
            gc = next(c for l, c in low_cols.items() if "retinopathy" in l)
            dc = next((c for l, c in low_cols.items() if "macular" in l), None)
            t = pd.DataFrame({"image_id": df[idc].astype(str).str.strip(), "grade": df[gc].astype(int), "dme": df[dc].astype(int) if dc else -1})
            t["path"] = t["image_id"].map(lambda s: images[split].get(("grading", s)))
            grading[split] = t
        elif "fovea" in low or "optic disc" in low or "od_center" in low:
            kind = "fo" if "fovea" in low else "od"
            xc = next(c for l, c in low_cols.items() if l.startswith("x"))
            yc = next(c for l, c in low_cols.items() if l.startswith("y"))
            idc = list(df.columns)[0]
            t = pd.DataFrame({"image_id": df[idc].astype(str).str.strip(), f"{kind}_x": pd.to_numeric(df[xc], errors="coerce"), f"{kind}_y": pd.to_numeric(df[yc], errors="coerce")})
            loc.setdefault(split, []).append(t)
    locd = {}
    for split, parts in loc.items():
        t = parts[0]
        for q in parts[1:]:
            t = t.merge(q, on="image_id", how="outer")
        t["path"] = t["image_id"].map(lambda s: images[split].get(("loc", s)))
        locd[split] = t.dropna()
    seg = {s: {k: v for k, v in d.items() if "image" in v} for s, d in seg.items()}
    return {"seg": seg, "grading": grading, "loc": locd}


def load_seg_item(item: dict, size: int = CANVAS):
    """(canvas_bg uint8 HxWx3, masks uint8 [5,H,W], fov) for one IDRiD segmentation image."""
    from .preprocess import read_image

    img = read_image(item["image"])
    canvas, fov = canvas_for_model(img, size)
    h, w = img.shape[:2]
    masks = np.zeros((len(LESIONS), size, size), np.uint8)
    for i, k in enumerate(LESIONS):
        m = _mask_read(item.get(k))
        if m is not None:
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            masks[i] = fov.to_canvas(m, mask=True)
    return canvas, masks, fov


# ----------------------------------------------------------------------------- training
POS_WEIGHT = torch.tensor([20.0, 4.0, 4.0, 6.0, 1.0])


def seg_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    """BCE (class-weighted positives) + soft Dice computed per class over the batch. target float [B,5,H,W].
    valid [B,5] (1 = this class is annotated for this image) enables partial-label training: unannotated classes contribute nothing."""
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=POS_WEIGHT.to(logits.device).view(1, -1, 1, 1), reduction="none")
    p = torch.sigmoid(logits)
    if valid is not None:
        v = valid.to(logits.device).float().view(valid.shape[0], valid.shape[1], 1, 1)
        bce = (bce * v).sum() / (v.sum() * logits.shape[2] * logits.shape[3]).clamp_min(1.0)
        p, target = p * v, target * v
    else:
        bce = bce.mean()
    inter = (p * target).sum((0, 2, 3))
    den = p.sum((0, 2, 3)) + target.sum((0, 2, 3))
    dice = 1 - (2 * inter + 1.0) / (den + 1.0)
    present = target.sum((0, 2, 3)) > 0  # only penalise Dice for classes that occur (and are annotated) in the batch
    dl = dice[present].mean() if present.any() else logits.sum() * 0
    return bce + dl


def sample_batch(images, masks, batch: int, patch: int, rng: np.random.Generator, p_lesion: float = 0.6):
    """Random crops (patch x patch); with probability p_lesion the crop is centred on a random lesion pixel."""
    xs, ys = [], []
    n = len(images)
    for _ in range(batch):
        i = int(rng.integers(n))
        H = images[i].shape[0]
        cy = cx = None
        if rng.random() < p_lesion:
            cls = [k for k in range(4) if masks[i][k].any()]
            if cls:
                k = cls[int(rng.integers(len(cls)))]
                yy, xx = np.nonzero(masks[i][k])
                j = int(rng.integers(len(yy)))
                cy, cx = int(yy[j]), int(xx[j])
        if cy is None:
            cy, cx = int(rng.integers(patch // 2, H - patch // 2)), int(rng.integers(patch // 2, H - patch // 2))
        y0 = int(np.clip(cy - patch // 2, 0, H - patch))
        x0 = int(np.clip(cx - patch // 2, 0, H - patch))
        im = images[i][y0 : y0 + patch, x0 : x0 + patch]
        mk = masks[i][:, y0 : y0 + patch, x0 : x0 + patch]
        k = int(rng.integers(4))
        im, mk = np.rot90(im, k, (0, 1)), np.rot90(mk, k, (1, 2))
        if rng.random() < 0.5:
            im, mk = im[:, ::-1], mk[:, :, ::-1]
        f = im.astype(np.float32) * rng.uniform(0.9, 1.1) + rng.uniform(-8, 8)
        xs.append(to_tensor_array(np.clip(f, 0, 255).astype(np.uint8)))
        ys.append(np.ascontiguousarray(mk))
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)).float()


@torch.no_grad()
def predict_probs(model: nn.Module, canvas: np.ndarray, device: str, amp: bool = True) -> np.ndarray:
    """Sigmoid probability maps [5,H,W] (float16) for one canvas image."""
    model.eval()
    x = torch.from_numpy(to_tensor_array(canvas))[None].to(device)
    with torch.autocast(device_type="cuda", enabled=amp and device == "cuda"):
        out = model(x)
    return torch.sigmoid(out.float())[0].cpu().numpy().astype(np.float16)


class PixelStats:
    """Accumulates per-class pixel histograms so AUPR / best-F1 can be computed over many large images without keeping them."""

    def __init__(self, n_bins: int = 200):
        self.n = n_bins
        self.pos = np.zeros((len(LESIONS), n_bins), np.int64)
        self.neg = np.zeros((len(LESIONS), n_bins), np.int64)

    def add(self, probs: np.ndarray, masks: np.ndarray, valid: np.ndarray | None = None, channels=None):
        for k in (range(len(LESIONS)) if channels is None else channels):
            p = probs[k].astype(np.float32)
            idx = np.minimum((p * self.n).astype(np.int64), self.n - 1)
            m = masks[k] > 0
            if valid is not None:
                p_idx, m_ = idx[valid], m[valid]
            else:
                p_idx, m_ = idx, m
            self.pos[k] += np.bincount(p_idx[m_], minlength=self.n)
            self.neg[k] += np.bincount(p_idx[~m_], minlength=self.n)

    def curves(self, k: int):
        tp = np.cumsum(self.pos[k][::-1])[::-1].astype(float)  # pixels with prob bin >= b
        fp = np.cumsum(self.neg[k][::-1])[::-1].astype(float)
        P = self.pos[k].sum()
        prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 1.0)
        rec = tp / P if P else np.zeros_like(tp)
        return prec, rec, tp, fp, P

    def aupr(self, k: int) -> float:
        """Average precision: sum over thresholds (high -> low) of (recall step) x precision, as in scikit-learn."""
        prec, rec, _, _, P = self.curves(k)
        if P == 0:
            return float("nan")
        r, p = rec[::-1], prec[::-1]  # bins are ascending thresholds; reverse = descending threshold = increasing recall
        return float(np.sum(np.diff(np.concatenate([[0.0], r])) * p))

    def best_threshold(self, k: int) -> tuple[float, float]:
        """(threshold, F1/Dice at it) maximising pixel F1; bins are right-edge thresholds."""
        prec, rec, tp, fp, P = self.curves(k)
        if P == 0:
            return 0.5, float("nan")
        f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
        b = int(np.argmax(f1))
        return float(b / self.n), float(f1[b])

    def at_threshold(self, k: int, thr: float) -> dict:
        b = int(round(thr * self.n))
        tp = float(self.pos[k][b:].sum())
        fp = float(self.neg[k][b:].sum())
        fn = float(self.pos[k][:b].sum())
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
        return {"dice": dice, "iou": iou, "precision": tp / (tp + fp) if tp + fp else float("nan"), "recall": tp / (tp + fn) if tp + fn else float("nan"),
                "threshold": thr}


# ----------------------------------------------------------------------------- post-processing + features
DEFAULT_MIN_AREA = {"MA": 3, "HE": 25, "EX": 12, "SE": 40}  # canvas pixels (canvas = 1280 px across the retina)


def components(prob: np.ndarray, thr: float, min_area: int) -> list[dict]:
    """Connected components of prob>thr: list of {cx, cy, area}. Coordinates in canvas px."""
    b = (prob.astype(np.float32) > thr).astype(np.uint8)
    n, _, stats, cent = cv2.connectedComponentsWithStats(b, connectivity=8)
    return [{"cx": float(cent[i][0]), "cy": float(cent[i][1]), "area": int(stats[i, cv2.CC_STAT_AREA])} for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]


def quadrant_of(pt, fovea, od) -> int:
    """Quadrant index (see QUADRANTS) of canvas point `pt` in the frame centred on the fovea whose horizontal axis is the
    OD->fovea line (temporal side = beyond the fovea). 'superior' = the perpendicular direction pointing up in the image.
    Assumes an upright photograph; an approximation of the clinical quadrants, not a validated definition."""
    u = np.asarray(fovea, float) - np.asarray(od, float)
    nu = np.linalg.norm(u)
    u = u / nu if nu > 1e-6 else np.array([1.0, 0.0])
    v = np.array([-u[1], u[0]])
    if v[1] > 0:
        v = -v  # make v point up (negative y)
    d = np.asarray(pt, float) - np.asarray(fovea, float)
    temporal = float(d @ u) > 0
    superior = float(d @ v) > 0
    return {(True, True): 0, (True, False): 1, (False, True): 2, (False, False): 3}[(temporal, superior)]


FEATURE_NAMES = (
    [f"{k}_count" for k in FINDINGS] + [f"{k}_large" for k in FINDINGS] + [f"{k}_area_frac" for k in FINDINGS]
    + [f"{k}_q{q}" for k in FINDINGS for q in range(4)]
    + ["HE_quadrants_ge1", "HE_quadrants_ge5", "HE_min_quadrant_count", "MA_quadrants_ge1",
       "EX_within_1DD", "EX_within_2DD", "HE_within_1DD", "EX_area_within_1DD_frac", "od_diameter_px", "fovea_od_dist_px", "loc_ok", "loc_fallback"]
)


def od_component(prob_od: np.ndarray, thr: float, min_area: int = 2500):
    """Largest connected component of the optic-disc probability map above `thr` -> (cx, cy, area) or None."""
    b = (prob_od.astype(np.float32) > thr).astype(np.uint8)
    n, _, stats, cent = cv2.connectedComponentsWithStats(b, connectivity=8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[i, cv2.CC_STAT_AREA] < min_area:
        return None
    return float(cent[i][0]), float(cent[i][1]), float(stats[i, cv2.CC_STAT_AREA])


def locate_od_fovea(probs: np.ndarray, reg_pts: np.ndarray, thr_od: float, prior_dd: float = 2.5):
    """Optic-disc and fovea positions (canvas px) from the segmentation network and the point regressor.

    The disc position comes from the segmented disc (much more reliable on unseen cameras than the regressor). The regressor's
    fovea is used only if its disc estimate agrees with the segmented disc (within one disc diameter); otherwise the fovea is placed by
    an anatomical prior: `prior_dd` disc diameters from the disc toward the centre of the retina. Returns (od, fovea, info)."""
    H = probs.shape[1]
    reg_od, reg_fo = np.asarray(reg_pts[0], float), np.asarray(reg_pts[1], float)
    comp = od_component(probs[LESIONS.index("OD")], thr_od)
    if comp is None:  # no disc segmented: nothing to cross-check against, keep the regressor and say so
        return reg_od, reg_fo, {"od_source": "regressor", "fovea_source": "regressor", "fallback": 1, "od_area": None}
    od = np.array(comp[:2])
    dd = 2.0 * np.sqrt(comp[2] / np.pi)
    if np.linalg.norm(od - reg_od) <= 1.0 * dd:
        return od, reg_fo, {"od_source": "segmentation", "fovea_source": "regressor", "fallback": 0, "od_area": comp[2]}
    v = np.array([H / 2.0, H / 2.0]) - od
    n = np.linalg.norm(v)
    if n < 0.5 * dd:  # disc near the centre: direction is ambiguous, point away from the nearer horizontal edge
        v, n = np.array([1.0 if od[0] < H / 2 else -1.0, 0.0]), 1.0
    return od, od + prior_dd * dd * v / n, {"od_source": "segmentation", "fovea_source": "anatomical prior", "fallback": 1, "od_area": comp[2]}


def extract_features(probs: np.ndarray, thresholds: dict, od_xy, fovea_xy, min_area: dict | None = None, od_mask_area: float | None = None, fallback: int = 0,
                     max_area: dict | None = None) -> tuple[dict, dict]:
    """Per-image lesion features + the component lists. probs [5,H,W]; od_xy/fovea_xy in canvas px."""
    min_area = min_area or DEFAULT_MIN_AREA
    H = probs.shape[1]
    fov_area = float(circle_mask(H, 0.98).sum())
    comps = {k: components(probs[LESIONS.index(k)], thresholds[k], min_area[k]) for k in FINDINGS}
    large = {k: 0 for k in FINDINGS}
    if max_area:  # regions far larger than any annotated lesion (e.g. illumination artefacts) are excluded from the counts and tallied separately
        for k in FINDINGS:
            keep = [c for c in comps[k] if c["area"] <= max_area[k]]
            large[k] = len(comps[k]) - len(keep)
            comps[k] = keep
    dd = float(2 * np.sqrt(od_mask_area / np.pi)) if od_mask_area and od_mask_area > 50 else 0.09 * H  # disc diameter in canvas px
    f: dict = {}
    qcount = {k: np.zeros(4, int) for k in FINDINGS}
    near = {k: {1: 0, 2: 0} for k in ("EX", "HE")}
    ex_area_near = 0
    for k in FINDINGS:
        f[f"{k}_count"] = len(comps[k])
        f[f"{k}_large"] = large[k]
        f[f"{k}_area_frac"] = sum(c["area"] for c in comps[k]) / fov_area
        for c in comps[k]:
            qcount[k][quadrant_of((c["cx"], c["cy"]), fovea_xy, od_xy)] += 1
            dist = float(np.hypot(c["cx"] - fovea_xy[0], c["cy"] - fovea_xy[1]))
            if k in near:
                for r in (1, 2):
                    near[k][r] += int(dist <= r * dd)
            if k == "EX" and dist <= dd:
                ex_area_near += c["area"]
        for q in range(4):
            f[f"{k}_q{q}"] = int(qcount[k][q])
    f["HE_quadrants_ge1"] = int((qcount["HE"] >= 1).sum())
    f["HE_quadrants_ge5"] = int((qcount["HE"] >= 5).sum())
    f["HE_min_quadrant_count"] = int(qcount["HE"].min())
    f["MA_quadrants_ge1"] = int((qcount["MA"] >= 1).sum())
    f["EX_within_1DD"], f["EX_within_2DD"] = near["EX"][1], near["EX"][2]
    f["HE_within_1DD"] = near["HE"][1]
    f["EX_area_within_1DD_frac"] = ex_area_near / fov_area
    f["od_diameter_px"] = dd
    f["fovea_od_dist_px"] = float(np.hypot(fovea_xy[0] - od_xy[0], fovea_xy[1] - od_xy[1]))
    f["loc_ok"] = 1
    f["loc_fallback"] = int(fallback)
    return f, comps


# ----------------------------------------------------------------------------- analyzer (used by the app / predictor)
LESION_COLORS = {"MA": (255, 0, 0), "HE": (255, 140, 0), "EX": (255, 255, 0), "SE": (0, 220, 255), "OD": (60, 60, 255)}


class LesionAnalyzer:
    """Loads lesion_unet.pt + locator.pt + lesion_config.json (written by the Kaggle lesion run) and analyses one image."""

    def __init__(self, directory: str | Path, device: str | None = None):
        import json

        d = Path(directory)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg = json.loads((d / "lesion_config.json").read_text())
        self.cfg = cfg
        self.seg = LesionUNet(pretrained=False)
        self.seg.load_state_dict(torch.load(d / "lesion_unet.pt", map_location="cpu", weights_only=False)["model_state"])
        self.seg.to(self.device).eval()
        self.loc = PointRegressor(pretrained=False)
        self.loc.load_state_dict(torch.load(d / "locator.pt", map_location="cpu", weights_only=False)["model_state"])
        self.loc.to(self.device).eval()
        self.thresholds = cfg["thresholds"]
        self.min_area = cfg.get("min_area", DEFAULT_MIN_AREA)
        self.max_area = cfg.get("max_area")
        self.loc_size = cfg.get("locator_size", 512)

    @torch.no_grad()
    def locate(self, canvas: np.ndarray) -> np.ndarray:
        small = cv2.resize(canvas, (self.loc_size, self.loc_size), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(to_tensor_array(small))[None].to(self.device)
        return self.loc(x).float().cpu().numpy()[0].reshape(2, 2) * canvas.shape[0]  # [[od_x, od_y], [fo_x, fo_y]]

    def analyze_canvas(self, canvas: np.ndarray) -> dict:
        probs = predict_probs(self.seg, canvas, self.device)
        reg = self.locate(canvas)
        od, fo, info = locate_od_fovea(probs, reg, self.thresholds.get("OD", 0.5))
        od_area = info["od_area"] if info["od_area"] else float((probs[LESIONS.index("OD")].astype(np.float32) > self.thresholds.get("OD", 0.5)).sum())
        feats, comps = extract_features(probs, self.thresholds, od, fo, self.min_area, od_area, info["fallback"], self.max_area)
        return {"features": feats, "components": comps, "od_xy": od, "fovea_xy": fo, "probs": probs, "location": info, "regressor_xy": reg}

    def analyze(self, img: np.ndarray) -> dict:
        canvas, fov = canvas_for_model(img, CANVAS)
        out = self.analyze_canvas(canvas)
        base = fov.to_canvas(img)
        base[~circle_mask(CANVAS, 0.98)] = 0
        out["overlay"] = lesion_overlay(base, out["probs"], self.thresholds, self.min_area, out["od_xy"], out["fovea_xy"], self.max_area)
        out["canvas_plain"] = base
        return out


def lesion_overlay(plain: np.ndarray, probs: np.ndarray, thresholds: dict, min_area: dict, od_xy, fovea_xy, max_area: dict | None = None) -> np.ndarray:
    """Colour the detected lesion regions on the plain canvas image and mark the estimated OD / fovea / quadrant axes."""
    out = plain.copy()
    for k in ("SE", "EX", "HE", "MA"):
        b = (probs[LESIONS.index(k)].astype(np.float32) > thresholds[k]).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
        keep = np.zeros(n, bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area[k]
        if max_area:
            keep[1:] &= stats[1:, cv2.CC_STAT_AREA] <= max_area[k]
        m = keep[lab]
        if k == "MA":
            m = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)  # make tiny dots visible
        out[m] = (0.35 * out[m] + 0.65 * np.array(LESION_COLORS[k])).astype(np.uint8)
    u = np.asarray(fovea_xy, float) - np.asarray(od_xy, float)
    nu = np.linalg.norm(u) or 1.0
    u = u / nu
    v = np.array([-u[1], u[0]])
    fx, fy = fovea_xy
    L = out.shape[0]
    for d in (u, v):
        cv2.line(out, (int(fx - d[0] * L), int(fy - d[1] * L)), (int(fx + d[0] * L), int(fy + d[1] * L)), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(out, (int(od_xy[0]), int(od_xy[1])), 12, LESION_COLORS["OD"], 3)
    cv2.drawMarker(out, (int(fx), int(fy)), (0, 255, 0), cv2.MARKER_CROSS, 30, 3)
    return out


# ===================================================================================== v2: more data, partial labels, strong augmentation
def index_eophtha(root: str | Path) -> list[dict]:
    """e-ophtha (EX and MA sets, MIT-declared Kaggle copy). Images are merged across the two sets by (patient, image name), masks are binary PNGs.
    Returns items {id, patient, image, masks: {MA, EX paths or None}, known: [class names whose annotation exists for this image], healthy}.
    'healthy' images carry no lesion of the annotated type(s): they are negatives for the classes of the set they come from (HE/SE/OD stay unknown)."""
    items: dict = {}
    for dp, _, files in os.walk(root):
        parts = [p.lower() for p in Path(dp).parts]
        for f in files:
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = str(Path(dp) / f)
            stem = Path(f).stem
            in_ex, in_ma = any("e_optha_ex" in p or "e_ophtha_ex" in p for p in parts), any("e_optha_ma" in p or "e_ophtha_ma" in p for p in parts)
            annotation = any(p.startswith("annotation") for p in parts)
            healthy = "healthy" in parts
            if not (in_ex or in_ma):
                continue
            kind = "EX" if in_ex else "MA"
            patient = Path(dp).name if Path(dp).name.upper().startswith("E") and not healthy else stem
            if annotation:
                base = stem[: -len("_" + kind)] if stem.endswith("_" + kind) else stem
                it = items.setdefault((patient, base), {"id": f"{patient}/{base}", "patient": patient, "masks": {"MA": None, "EX": None}, "known": set(), "healthy": False})
                it["masks"][kind] = path
            else:
                it = items.setdefault((patient, stem), {"id": f"{patient}/{stem}", "patient": patient, "masks": {"MA": None, "EX": None}, "known": set(), "healthy": False})
                it["image"] = path
                it["known"].add(kind)
                it["healthy"] = it["healthy"] or healthy
    out = []
    for it in items.values():
        if "image" in it:
            it["known"] = sorted(it["known"])
            out.append(it)
    return sorted(out, key=lambda d: d["id"])


def split_by_patient(items: list[dict], seed: int = 0, fracs=(0.8, 0.1, 0.1)) -> dict:
    """Patient-level train/val/test split (a patient's images never straddle splits; healthy images are their own group)."""
    pats = sorted({it["patient"] for it in items})
    rng = np.random.default_rng(seed)
    rng.shuffle(pats)
    n = len(pats)
    cut1, cut2 = int(fracs[0] * n), int((fracs[0] + fracs[1]) * n)
    where = {p: ("train" if i < cut1 else "val" if i < cut2 else "test") for i, p in enumerate(pats)}
    out = {"train": [], "val": [], "test": []}
    for it in items:
        out[where[it["patient"]]].append(it)
    return out


def load_plain_idrid(item: dict, size: int = CANVAS):
    """(plain masked canvas uint8, masks uint8 [5,H,W], valid bool [5], fov) for one IDRiD segmentation image (all classes annotated)."""
    from .preprocess import read_image

    img = read_image(item["image"])
    fov = make_fov(img, size)
    plain = fov.to_canvas(img)
    plain[~circle_mask(size, 0.98)] = 0
    h, w = img.shape[:2]
    masks = np.zeros((len(LESIONS), size, size), np.uint8)
    for i, k in enumerate(LESIONS):
        m = _mask_read(item.get(k))
        if m is not None:
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            masks[i] = fov.to_canvas(m, mask=True)
    return plain, masks, np.ones(len(LESIONS), bool), fov


def load_plain_eophtha(item: dict, size: int = CANVAS):
    """Same for an e-ophtha item: only MA / EX can be supervised, and only the channels whose set this image belongs to."""
    from .preprocess import read_image

    img = read_image(item["image"])
    fov = make_fov(img, size)
    plain = fov.to_canvas(img)
    plain[~circle_mask(size, 0.98)] = 0
    h, w = img.shape[:2]
    masks = np.zeros((len(LESIONS), size, size), np.uint8)
    valid = np.zeros(len(LESIONS), bool)
    for k in item["known"]:
        i = LESIONS.index(k)
        valid[i] = True
        m = _mask_read(item["masks"].get(k))
        if m is not None:
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            masks[i] = fov.to_canvas(m, mask=True)
    return plain, masks, valid, fov


def bg_from_plain(plain: np.ndarray) -> np.ndarray:
    """Model-input canvas (Ben Graham normalised) from a plain masked canvas: identical to canvas_for_model()."""
    return ben_graham(plain)


_CM98, _CM94 = {}, {}


def _circle(size: int, frac: float) -> np.ndarray:
    d = _CM98 if frac == 0.98 else _CM94
    if size not in d:
        d[size] = circle_mask(size, frac)
    return d[size]


def augment_acquisition(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Acquisition-style augmentation of a plain RGB patch: smooth shading field, per-channel gain (colour cast), gamma, contrast,
    blur, noise, JPEG. Applied BEFORE the local-contrast normalisation so the network sees realistic exposure and colour variation."""
    h, w = img.shape[:2]
    f = img.astype(np.float32) / 255.0
    if rng.random() < 0.6:  # uneven illumination: low-frequency multiplicative field
        g = rng.uniform(0.45, 1.5, (3, 3)).astype(np.float32)
        f = f * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)[..., None]
    f = f * rng.uniform(0.85, 1.15, (1, 1, 3)).astype(np.float32)  # colour cast
    f = np.clip(f, 0, 1) ** float(rng.uniform(0.6, 1.8))
    f = (f - 0.5) * float(rng.uniform(0.6, 1.3)) + 0.5 + float(rng.uniform(-0.08, 0.08))
    out = np.clip(f * 255.0, 0, 255).astype(np.uint8)
    if rng.random() < 0.2:
        out = cv2.GaussianBlur(out, (0, 0), float(rng.uniform(0.5, 2.5)))
    if rng.random() < 0.2:
        out = np.clip(out.astype(np.float32) + rng.normal(0, rng.uniform(1, 6), out.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        ok, enc = cv2.imencode(".jpg", out[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(20, 91))])
        if ok:
            out = np.ascontiguousarray(cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1])
    return out


def sample_batch_v2(plains, masks, valids, weights, batch: int, patch: int, rng: np.random.Generator, p_lesion: float = 0.6, margin: int = 64, augment: bool = True):
    """Patches with acquisition augmentation applied to the PLAIN image, then Ben Graham with the canvas-scale sigma, then a centre crop (the margin
    absorbs blur edge effects). weights: sampling probability per image. Returns x [B,3,P,P], y [B,5,P,P], valid [B,5]."""
    xs, ys, vs = [], [], []
    n = len(plains)
    big = patch + 2 * margin
    sigma = max(1.0, CANVAS / 30.0)
    for _ in range(batch):
        i = int(rng.choice(n, p=weights))
        H = plains[i].shape[0]
        cy = cx = None
        if rng.random() < p_lesion:
            cls = [k for k in range(4) if valids[i][k] and masks[i][k].any()]
            if cls:
                k = cls[int(rng.integers(len(cls)))]
                yy, xx = np.nonzero(masks[i][k])
                j = int(rng.integers(len(yy)))
                cy, cx = int(yy[j]), int(xx[j])
        if cy is None:
            cy, cx = int(rng.integers(big // 2, H - big // 2)), int(rng.integers(big // 2, H - big // 2))
        y0, x0 = int(np.clip(cy - big // 2, 0, H - big)), int(np.clip(cx - big // 2, 0, H - big))
        im = plains[i][y0 : y0 + big, x0 : x0 + big]
        mk = masks[i][:, y0 : y0 + big, x0 : x0 + big]
        k90 = int(rng.integers(4))
        im, mk = np.rot90(im, k90, (0, 1)), np.rot90(mk, k90, (1, 2))
        cm98 = np.rot90(_circle(H, 0.98)[y0 : y0 + big, x0 : x0 + big], k90)
        cm94 = np.rot90(_circle(H, 0.94)[y0 : y0 + big, x0 : x0 + big], k90)
        if rng.random() < 0.5:
            im, mk, cm98, cm94 = im[:, ::-1], mk[:, :, ::-1], cm98[:, ::-1], cm94[:, ::-1]
        im = np.ascontiguousarray(im)
        if augment:
            im = augment_acquisition(im, rng)
        im[~cm98] = 0
        blur = cv2.GaussianBlur(im, (0, 0), sigma)
        out = cv2.addWeighted(im, 4, blur, -4, 128)
        out[~cm94] = 128
        c = margin
        xs.append(to_tensor_array(np.ascontiguousarray(out[c : c + patch, c : c + patch])))
        ys.append(np.ascontiguousarray(mk[:, c : c + patch, c : c + patch]))
        vs.append(valids[i])
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)).float(), torch.from_numpy(np.stack(vs)).float()


def component_area_limits(masks_list, valids, percentile: float = 99.5, factor: float = 2.0) -> dict:
    """Maximum plausible region area per finding class, from the TRAINING annotations (percentile of annotated component areas x factor)."""
    areas = {k: [] for k in FINDINGS}
    for m, v in zip(masks_list, valids):
        for k in FINDINGS:
            i = LESIONS.index(k)
            if v[i] and m[i].any():
                n, _, st, _ = cv2.connectedComponentsWithStats(m[i].astype(np.uint8), connectivity=8)
                areas[k] += [int(a) for a in st[1:, cv2.CC_STAT_AREA]]
    return {k: int(np.percentile(a, percentile) * factor) if a else 10**9 for k, a in areas.items()}


def find_idrid_root(base: str | Path = "/kaggle/input") -> str | None:
    """Locate the IDRiD dataset root by content (Kaggle mounts may name folders literally 'A.%20Segmentation'): climb from a segmentation folder to the
    ancestor that also holds the localization and grading parts."""
    base = Path(base)
    if not base.exists():
        return None
    for p in base.rglob("*egmentation*"):
        if not p.is_dir():
            continue
        for cand in [p, *p.parents]:
            if cand == base.parent:
                break
            try:
                names = " ".join(x.name.lower() for x in cand.iterdir())
            except OSError:
                continue
            if "localization" in names and "grading" in names:
                return str(cand)
        return str(p)
    return None
