"""EyePACS (Kaggle 'Diabetic Retinopathy Detection', 2015) access: labels, patient-level split and image reading from either an extracted
`train/` folder or the split archive parts (`train.zip.001`, ...) without extracting 34 GB.

Terms: the data belong to EyePACS / CHF and are used here under the Kaggle competition rules the account holder accepted; nothing is redistributed.
"""
from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .preprocess import ImageError, validate_array


class ConcatFile(io.RawIOBase):
    """Read-only, seekable view of several files concatenated (a raw split archive: train.zip.001, train.zip.002, ...)."""

    def __init__(self, parts):
        self.parts = [Path(p) for p in parts]
        self.sizes = [p.stat().st_size for p in self.parts]
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)])
        self.total = int(self.offsets[-1])
        self.pos = 0
        self._fh: dict[int, io.BufferedReader] = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: off, io.SEEK_CUR: self.pos + off, io.SEEK_END: self.total + off}[whence]
        self.pos = max(0, min(self.pos, self.total))
        return self.pos

    def readinto(self, b):
        n, want = 0, len(b)
        while n < want and self.pos < self.total:
            i = int(np.searchsorted(self.offsets, self.pos, side="right") - 1)
            fh = self._fh.get(i) or self._fh.setdefault(i, open(self.parts[i], "rb"))
            fh.seek(self.pos - int(self.offsets[i]))
            chunk = fh.read(min(want - n, int(self.offsets[i + 1]) - self.pos))
            if not chunk:
                break
            b[n : n + len(chunk)] = chunk
            n += len(chunk)
            self.pos += len(chunk)
        return n

    def close(self):
        for fh in self._fh.values():
            fh.close()
        super().close()


def find_eyepacs(root: str | Path) -> dict:
    """Locate the pieces under `root` (searches recursively): labels (csv or zipped csv), an extracted train folder, or split archive parts."""
    root = Path(root)
    out = {"labels": None, "train_dir": None, "train_parts": []}
    for dp, dn, files in os.walk(root):
        for f in files:
            fl = f.lower()
            if fl in ("trainlabels.csv", "trainlabels.csv.zip") and out["labels"] is None:
                out["labels"] = Path(dp) / f
            if fl.startswith("train.zip.") or fl == "train.zip":
                out["train_parts"].append(Path(dp) / f)
        for d in dn:
            if d.lower() == "train" and out["train_dir"] is None and any((Path(dp) / d).glob("*.jpeg")) :
                out["train_dir"] = Path(dp) / d
    out["train_parts"] = sorted(out["train_parts"])
    return out


def load_labels(labels_path: str | Path) -> pd.DataFrame:
    p = Path(labels_path)
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            df = pd.read_csv(z.open(name))
    else:
        df = pd.read_csv(p)
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    df["patient"] = df["image"].str.split("_").str[0]
    df["eye"] = df["image"].str.split("_").str[1]
    return df.rename(columns={"level": "grade"})[["image", "grade", "patient", "eye"]]


def patient_split(df: pd.DataFrame, seed: int = 0, fracs=(0.85, 0.05, 0.10)) -> pd.DataFrame:
    """Patient-level train / val / test assignment (both eyes of a patient always land in the same split)."""
    pats = np.array(sorted(df["patient"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(pats)
    n = len(pats)
    c1, c2 = int(fracs[0] * n), int((fracs[0] + fracs[1]) * n)
    where = {p: ("train" if i < c1 else "val" if i < c2 else "test") for i, p in enumerate(pats)}
    out = df.copy()
    out["ep_split"] = out["patient"].map(where)
    return out


def build_pool(df: pd.DataFrame, n_grade0_train: int = 7000, n_test: int = 3500, seed: int = 0) -> pd.DataFrame:
    """Training pool: every grade >= 1 image of the train patients plus a random sample of grade-0 images; test: a random sample of test-patient images
    with their NATURAL grade mix (so EyePACS-internal numbers are not distorted by the pool design)."""
    rng = np.random.default_rng(seed)
    tr = df[df.ep_split == "train"]
    pos = tr[tr.grade >= 1]
    neg = tr[tr.grade == 0]
    neg = neg.sample(min(len(neg), n_grade0_train), random_state=int(rng.integers(1 << 31)))
    te = df[df.ep_split == "test"]
    te = te.sample(min(len(te), n_test), random_state=int(rng.integers(1 << 31)))
    pool = pd.concat([pos.assign(split="train"), neg.assign(split="train"), te.assign(split="ep_test")], ignore_index=True)
    pool["image_id"] = "ep_" + pool["image"]
    pool["dataset"] = "eyepacs"
    return pool


class EyePacsReader:
    """Thread-safe reader returning (RGB uint8 image, md5-of-bytes) for an EyePACS image name, from a train folder or from split archive parts."""

    def __init__(self, found: dict):
        import hashlib
        import threading

        self._md5 = lambda b: hashlib.md5(b).hexdigest()
        self.train_dir = found.get("train_dir")
        self.zf = None
        self.lock = threading.Lock()
        self.members: dict[str, str] = {}
        if self.train_dir is None:
            if not found.get("train_parts"):
                raise FileNotFoundError("no EyePACS train images found (neither a train folder nor train.zip.* parts)")
            self.zf = zipfile.ZipFile(ConcatFile(found["train_parts"]))
            self.members = {Path(n).stem: n for n in self.zf.namelist() if n.lower().endswith((".jpeg", ".jpg", ".png"))}

    def read(self, image: str):
        if self.train_dir is not None:
            cand = list(Path(self.train_dir).glob(f"{image}.*"))
            if not cand:
                raise ImageError(f"missing EyePACS image {image}")
            data = cand[0].read_bytes()
        else:
            if image not in self.members:
                raise ImageError(f"missing EyePACS image {image}")
            with self.lock:  # one reader at a time on the shared archive handle; decoding happens outside the lock
                data = self.zf.read(self.members[image])
        flag = cv2.IMREAD_REDUCED_COLOR_2 if len(data) > 150_000 else cv2.IMREAD_COLOR  # large photographs: decode at half size (still >= 800 px)
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), flag)
        if bgr is None:
            raise ImageError(f"cannot decode EyePACS image {image}")
        return validate_array(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), self._md5(data)


def hash_bits(hexes) -> np.ndarray:
    """dHash hex strings -> [n, bits] uint8 array."""
    hexes = list(hexes)
    if not hexes:
        return np.zeros((0, 256), np.uint8)
    return np.unpackbits(np.frombuffer(b"".join(bytes.fromhex(h) for h in hexes), dtype=np.uint8).reshape(len(hexes), -1), axis=1)


def near_duplicates(bits_a: np.ndarray, bits_b: np.ndarray, max_hamming: int = 6, chunk: int = 1000) -> np.ndarray:
    """Boolean mask over rows of `bits_a`: True if some row of `bits_b` is within `max_hamming` bits (guard against a photo being in two datasets)."""
    out = np.zeros(len(bits_a), bool)
    if len(bits_b) == 0:
        return out
    b = bits_b.astype(np.float32)
    bs = b.sum(1)
    for i in range(0, len(bits_a), chunk):
        a = bits_a[i : i + chunk].astype(np.float32)
        dist = a.sum(1)[:, None] + bs[None, :] - 2 * (a @ b.T)
        out[i : i + chunk] = (dist <= max_hamming + 0.5).any(1)
    return out
