"""Dataset discovery, validation, leakage-aware splits, preprocessing cache and the PyTorch Dataset."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .preprocess import IMAGE_EXTENSIONS, ImageError, preprocess_plain, read_image, to_model_input, to_tensor_array

ID_ALIASES = ("id_code", "image", "image_id", "id", "image name", "image_name")
LABEL_ALIASES = ("diagnosis", "level", "label", "retinopathy grade", "dr_grade", "grade")


# --------------------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------------------
def _norm(c: str) -> str:
    return str(c).strip().lower()


NOT_GROUND_TRUTH = ("submission", "sample_sub", "predictions", "metrics", "manifest")  # CSVs with id+label-like columns that are NOT labels


def find_labelled_tables(root: str | Path):
    """Yield (path, id_col, label_col) for every CSV under root that has an id column and a DR-grade column.
    Submission templates (e.g. Kaggle's sample_submission.csv, whose 'diagnosis' is a dummy 0) and our own outputs are skipped."""
    for dp, _, files in os.walk(root):
        for f in files:
            if not f.lower().endswith(".csv") or any(w in f.lower() for w in NOT_GROUND_TRUTH):
                continue
            p = Path(dp) / f
            try:
                head = pd.read_csv(p, nrows=5)
            except Exception:
                continue
            cols = {_norm(c): c for c in head.columns}
            idc = next((cols[a] for a in ID_ALIASES if a in cols), None)
            lc = next((cols[a] for a in LABEL_ALIASES if a in cols), None)
            if idc and lc:
                yield p, idc, lc


def index_images(root: str | Path) -> dict[str, Path]:
    """Map image stem -> path for all images under root (first occurrence wins)."""
    idx: dict[str, Path] = {}
    for dp, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMAGE_EXTENSIONS):
                idx.setdefault(Path(f).stem, Path(dp) / f)
    return idx


def load_labelled_dataset(root: str | Path, name: str = "dataset", only_csv_stems: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Build a table (image_id, grade, path, source_csv) from every labelled CSV under root.

    Handles the original APTOS competition layout (train.csv + train_images/) and common Kaggle mirrors.
    Rows are de-duplicated by image_id; rows with an invalid label or no matching image are reported, not silently used.
    """
    root = Path(root)
    tables = []
    for p, idc, lc in find_labelled_tables(root):
        if only_csv_stems and p.stem not in only_csv_stems:
            continue
        t = pd.read_csv(p, usecols=[idc, lc]).rename(columns={idc: "image_id", lc: "grade"})
        if len(t) > 50 and t["grade"].nunique() <= 1:  # a constant 'label' column is a placeholder, not ground truth
            print(f"[datasets] skipping {p.name}: constant label column ({len(t)} rows) is not ground truth", flush=True)
            continue
        t["source_csv"] = p.name
        tables.append(t)
    if not tables:
        raise FileNotFoundError(f"no labelled CSV (id + DR grade columns) found under {root}")
    df = pd.concat(tables, ignore_index=True)
    df["image_id"] = df["image_id"].astype(str).str.replace(r"\.(png|jpe?g|tiff?|bmp)$", "", regex=True, case=False)
    df = df.drop_duplicates("image_id").reset_index(drop=True)
    idx = index_images(root)
    df["path"] = df["image_id"].map(lambda s: str(idx[s]) if s in idx else None)
    n0 = len(df)
    bad_label = ~pd.to_numeric(df["grade"], errors="coerce").isin([0, 1, 2, 3, 4])
    missing = df["path"].isna()
    df.attrs["dropped"] = {"invalid_label": int(bad_label.sum()), "missing_image": int(missing.sum()), "total_rows": n0}
    df = df[~bad_label & ~missing].copy()
    df["grade"] = pd.to_numeric(df["grade"]).astype(int)
    df["dataset"] = name
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# validation + hashing + cache
# --------------------------------------------------------------------------------------
def file_md5(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(img: np.ndarray, hash_size: int = 16) -> np.ndarray:
    """Difference hash (hash_size*hash_size bits) of the fundus; used to find near-duplicate photos."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    return (g[:, 1:] > g[:, :-1]).flatten()


def build_cache(df: pd.DataFrame, size: int, cache_dir: str | Path, progress: bool = True, workers: int | None = None, reader=None):
    """Preprocess every image once (crop/mask/resize) into a uint8 .npy array and record per-image info.

    Returns (df_ok, cache_path): df_ok keeps only readable images, with columns cache_idx, md5, dhash (hex),
    orig_w, orig_h. Unreadable images are written to <cache_dir>/rejected_images.csv (never silently dropped).
    `reader(path) -> (rgb image, md5)` replaces reading from disk (used for images inside a split archive).
    """
    workers = workers or min(4, os.cpu_count() or 1)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    arr_path = cache_dir / f"plain_{size}.npy"
    n = len(df)
    arr = np.lib.format.open_memmap(arr_path, mode="w+", dtype=np.uint8, shape=(n, size, size, 3))
    kept, rejected, md5s, dh, ow, oh = [], [], [], [], [], []

    def work(pos_row):
        pos, row = pos_row
        try:
            img, md5 = reader(row.path) if reader else (read_image(row.path), file_md5(row.path))
            return pos, row, preprocess_plain(img, size), md5, np.packbits(dhash(img)).tobytes().hex(), img.shape[1], img.shape[0], None
        except (ImageError, cv2.error, ValueError) as e:
            return pos, row, None, None, None, None, None, str(e)

    from concurrent.futures import ThreadPoolExecutor

    it = ThreadPoolExecutor(max_workers=workers).map(work, enumerate(df[["image_id", "path"]].itertuples(index=False)))  # ordered; OpenCV releases the GIL
    if progress:
        try:
            from tqdm import tqdm

            it = tqdm(it, total=n, desc=f"preprocess@{size}")
        except Exception:
            pass
    j = 0
    for pos, row, arr_j, md5, hx, w_, h_, err in it:
        if err is not None:
            rejected.append({"image_id": row.image_id, "path": row.path, "error": err})
            continue
        arr[j] = arr_j
        md5s.append(md5)
        dh.append(hx)
        ow.append(w_)
        oh.append(h_)
        kept.append(pos)
        j += 1
    arr.flush()
    ok = df.iloc[kept].reset_index(drop=True)
    ok["cache_idx"] = np.arange(j)
    ok["md5"], ok["dhash"], ok["orig_w"], ok["orig_h"] = md5s, dh, ow, oh
    if rejected:
        pd.DataFrame(rejected).to_csv(cache_dir / "rejected_images.csv", index=False)
    if j < n:  # shrink the array to the readable images
        tmp = cache_dir / f"plain_{size}_tmp.npy"
        small = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(j, size, size, 3))
        small[:] = arr[:j]
        small.flush()
        del arr, small
        os.replace(tmp, arr_path)
    return ok, arr_path


# --------------------------------------------------------------------------------------
# leakage-aware splits
# --------------------------------------------------------------------------------------
def duplicate_groups(df: pd.DataFrame, max_hamming: int = 6) -> np.ndarray:
    """Group ids so that exact duplicates (md5) and near-duplicates (dHash Hamming <= max_hamming) share a group.

    No patient/eye identifiers exist in APTOS, so this is the only available guard against the same photo
    appearing in train and test. It is NOT patient-level separation.
    """
    n = len(df)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    first = {}
    for i, m in enumerate(df["md5"].tolist()):
        if m in first:
            union(first[m], i)
        else:
            first[m] = i
    if max_hamming >= 0 and "dhash" in df:
        bits = np.unpackbits(np.frombuffer(b"".join(bytes.fromhex(h) for h in df["dhash"]), dtype=np.uint8).reshape(n, -1), axis=1)
        for i in range(n):
            dist = (bits[i + 1 :] != bits[i]).sum(axis=1)
            for j in np.where(dist <= max_hamming)[0]:
                union(i, i + 1 + int(j))
    roots = np.array([find(i) for i in range(n)])
    _, groups = np.unique(roots, return_inverse=True)
    return groups


def make_splits(df: pd.DataFrame, seed: int = 42, val_frac: float = 0.15, test_frac: float = 0.15, max_hamming: int = 6) -> pd.DataFrame:
    """Stratified (on grade), group-aware (duplicates stay together) train/val/test split. Adds 'group' and 'split'."""
    from sklearn.model_selection import StratifiedGroupKFold

    df = df.copy()
    df["group"] = duplicate_groups(df, max_hamming)
    y, g = df["grade"].to_numpy(), df["group"].to_numpy()
    k_test = max(2, round(1 / test_frac))
    sgkf = StratifiedGroupKFold(n_splits=k_test, shuffle=True, random_state=seed)
    rest, test = next(sgkf.split(df, y, g))
    k_val = max(2, round((1 - test_frac) / val_frac))
    sgkf2 = StratifiedGroupKFold(n_splits=k_val, shuffle=True, random_state=seed + 1)
    tr, va = next(sgkf2.split(df.iloc[rest], y[rest], g[rest]))
    df["split"] = ""
    df.loc[df.index[rest[tr]], "split"] = "train"
    df.loc[df.index[rest[va]], "split"] = "val"
    df.loc[df.index[test], "split"] = "test"
    assert (df["split"] != "").all()
    check_no_leakage(df)
    return df


def check_no_leakage(df: pd.DataFrame) -> None:
    """Raise if any duplicate group or md5 spans more than one split."""
    for col in ("group", "md5"):
        n_splits = df.groupby(col)["split"].nunique()
        if (n_splits > 1).any():
            raise AssertionError(f"leakage: {int((n_splits > 1).sum())} {col} values span multiple splits")


def split_summary(df: pd.DataFrame) -> dict:
    out = {}
    for s in ("train", "val", "test"):
        sub = df[df["split"] == s]
        out[s] = {
            "n": int(len(sub)),
            "grade_counts": {str(g): int((sub["grade"] == g).sum()) for g in range(5)},
            "referable_prevalence": float((sub["grade"] >= 2).mean()) if len(sub) else float("nan"),
        }
    return out


def save_manifest(df: pd.DataFrame, path: str | Path, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = [c for c in ("image_id", "grade", "split", "group", "md5", "orig_w", "orig_h", "dataset", "cache_idx") if c in df]
    df[keep].to_csv(path, index=False)
    if meta is not None:
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


# --------------------------------------------------------------------------------------
# augmentation + torch dataset
# --------------------------------------------------------------------------------------
def augment_plain(img: np.ndarray, rng: np.random.Generator, robust: bool = False) -> np.ndarray:
    """Training-only augmentation on a plain (masked) fundus image: rotation, flips, scale, brightness/contrast, gamma.
    robust=True additionally simulates acquisition/transmission defects: Gaussian blur, JPEG re-compression, wider
    gamma/contrast changes and over-exposure (each applied independently with its own probability)."""
    s = img.shape[0]
    angle = rng.uniform(0, 360)
    scale = rng.uniform(0.92, 1.08)
    M = cv2.getRotationMatrix2D(((s - 1) / 2, (s - 1) / 2), angle, scale)
    out = cv2.warpAffine(img, M, (s, s), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    if rng.random() < 0.5:
        out = out[:, ::-1]
    if rng.random() < 0.5:
        out = out[::-1]
    f = out.astype(np.float32)
    f = f * rng.uniform(0.85, 1.15) + rng.uniform(-12, 12)  # contrast, brightness
    f = 255.0 * np.clip(f / 255.0, 0, 1) ** rng.uniform(0.85, 1.2)  # gamma
    out = np.ascontiguousarray(np.clip(f, 0, 255).astype(np.uint8))
    if robust:
        if rng.random() < 0.25:  # defocus / motion blur
            out = cv2.GaussianBlur(out, (0, 0), float(rng.uniform(0.5, 4.0)))
        if rng.random() < 0.30:  # JPEG re-compression (blocking, colour banding)
            ok, enc = cv2.imencode(".jpg", out[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(8, 61))])
            if ok:
                out = np.ascontiguousarray(cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1])
        if rng.random() < 0.30:  # dark / low-contrast / washed-out illumination
            g, c = float(rng.uniform(0.6, 1.8)), float(rng.uniform(0.5, 1.2))
            f = ((np.clip(out.astype(np.float32) / 255.0, 0, 1) ** g) - 0.5) * c + 0.5
            out = np.clip(f * 255.0, 0, 255).astype(np.uint8)
        if rng.random() < 0.15:  # over-exposure
            out = np.clip(out.astype(np.float32) * float(rng.uniform(1.3, 2.2)) + float(rng.uniform(5, 40)), 0, 255).astype(np.uint8)
    return out


class DRDataset:
    """Torch-style dataset over the preprocessing cache. Augmentation only when train=True."""

    def __init__(self, cache: np.ndarray, labels: np.ndarray, indices: np.ndarray, train: bool = False, use_ben_graham: bool = True, seed: int = 0,
                 robust: bool = False):
        self.cache, self.labels, self.indices = cache, np.asarray(labels), np.asarray(indices)
        self.train, self.bg, self.seed, self.robust = train, use_ben_graham, seed, robust

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        import torch

        k = int(self.indices[i])
        img = np.asarray(self.cache[k])
        if self.train:
            rng = np.random.default_rng((torch.initial_seed() + 1_000_003 * i + self.seed) % (2**32))
            img = augment_plain(img, rng, self.robust)
        x = to_tensor_array(to_model_input(img, self.bg))
        return torch.from_numpy(x), int(self.labels[k])
