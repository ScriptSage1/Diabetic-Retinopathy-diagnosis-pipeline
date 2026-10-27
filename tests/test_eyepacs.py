import io
import zipfile

import cv2
import numpy as np
import pandas as pd

from drscreen import eyepacs as E


def _labels(n_pat=200):
    rows = []
    rng = np.random.default_rng(0)
    for p in range(n_pat):
        for eye in ("left", "right"):
            rows.append({"image": f"{p + 10}_{eye}", "level": int(rng.choice(5, p=[0.7, 0.07, 0.15, 0.04, 0.04]))})
    return pd.DataFrame(rows)


def test_labels_from_zip_and_patient_split_has_no_patient_overlap(tmp_path):
    df = _labels()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("trainLabels.csv", df.to_csv(index=False))
    (tmp_path / "trainLabels.csv.zip").write_bytes(buf.getvalue())
    lab = E.load_labels(tmp_path / "trainLabels.csv.zip")
    assert list(lab.columns) == ["image", "grade", "patient", "eye"] and len(lab) == 400 and lab.patient.nunique() == 200
    sp = E.patient_split(lab, seed=1)
    assert sp.groupby("patient")["ep_split"].nunique().max() == 1  # both eyes always together
    assert set(sp.ep_split) == {"train", "val", "test"}
    pool = E.build_pool(sp, n_grade0_train=30, n_test=25)
    tr = pool[pool.split == "train"]
    assert (tr.grade >= 1).sum() == ((sp.ep_split == "train") & (sp.grade >= 1)).sum()  # every positive kept
    assert (tr.grade == 0).sum() <= 30 and (pool.split == "ep_test").sum() <= 25
    assert set(pool[pool.split == "ep_test"].patient).isdisjoint(set(tr.patient))
    assert pool.image_id.str.startswith("ep_").all()


def test_reader_over_split_archive_matches_plain_files(tmp_path):
    rng = np.random.default_rng(0)
    imgs = {}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for name in ("10_left", "10_right", "11_left"):
            im = rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)
            ok, enc = cv2.imencode(".jpeg", im)
            imgs[name] = enc.tobytes()
            z.writestr(f"train/{name}.jpeg", enc.tobytes())
    raw = buf.getvalue()
    cuts = [0, len(raw) // 3, 2 * len(raw) // 3, len(raw)]
    for i in range(3):  # an arbitrary byte-wise split, as Kaggle's train.zip.001 ... parts
        (tmp_path / f"train.zip.00{i + 1}").write_bytes(raw[cuts[i]:cuts[i + 1]])
    found = E.find_eyepacs(tmp_path)
    assert found["train_dir"] is None and len(found["train_parts"]) == 3
    rd = E.EyePacsReader(found)
    assert set(rd.members) == {"10_left", "10_right", "11_left"}
    img, md5 = rd.read("10_right")
    import hashlib

    assert img.shape == (300, 400, 3) and md5 == hashlib.md5(imgs["10_right"]).hexdigest()


def test_near_duplicates_mask():
    rng = np.random.default_rng(1)
    ref = rng.integers(0, 2, (20, 256), dtype=np.uint8)
    a = rng.integers(0, 2, (50, 256), dtype=np.uint8)
    a[3] = ref[7]
    a[10] = ref[2].copy()
    a[10, :4] ^= 1  # 4 bits flipped: still a near duplicate
    a[11] = ref[5].copy()
    a[11, :30] ^= 1  # 30 bits flipped: different image
    m = E.near_duplicates(a, ref, 6)
    assert m[3] and m[10] and not m[11] and m.sum() == 2
    assert not E.near_duplicates(a, ref[:0], 6).any()
    assert E.hash_bits(["ff" * 32]).shape == (1, 256)
