import shutil

import numpy as np
import pandas as pd

from drscreen import datasets as D


def test_label_mapping_and_discovery(synth_root):
    df = D.load_labelled_dataset(synth_root, "synthetic")
    assert len(df) == 90
    assert set(df["grade"]) == {0, 1, 2, 3, 4}
    assert df["path"].notna().all()
    truth = pd.read_csv(synth_root / "train.csv").set_index("id_code")["diagnosis"]
    assert (df.set_index("image_id")["grade"] == truth.loc[df["image_id"]]).all()


def test_invalid_and_missing_rows_reported(tmp_path, synth_root):
    shutil.copytree(synth_root, tmp_path / "d")
    t = pd.read_csv(tmp_path / "d" / "train.csv")
    t.loc[0, "diagnosis"] = 9
    t.loc[1, "id_code"] = "no_such_image"
    t.to_csv(tmp_path / "d" / "train.csv", index=False)
    df = D.load_labelled_dataset(tmp_path / "d")
    assert df.attrs["dropped"]["invalid_label"] == 1 and df.attrs["dropped"]["missing_image"] == 1
    assert len(df) == 88


def test_cache_handles_corrupt_image(tmp_path, synth_root):
    shutil.copytree(synth_root, tmp_path / "d")
    df = D.load_labelled_dataset(tmp_path / "d")
    (tmp_path / "d" / "train_images" / f"{df['image_id'][3]}.png").write_bytes(b"garbage")
    ok, p = D.build_cache(df, 48, tmp_path / "c", progress=False)
    assert len(ok) == len(df) - 1
    arr = np.load(p, mmap_mode="r")
    assert arr.shape == (len(ok), 48, 48, 3)
    assert (tmp_path / "c" / "rejected_images.csv").exists()


def test_splits_stratified_and_leak_free(prepared):
    df, _ = prepared
    assert set(df["split"]) == {"train", "val", "test"}
    D.check_no_leakage(df)
    frac = df["split"].value_counts(normalize=True)
    assert 0.55 < frac["train"] < 0.8 and 0.08 < frac["val"] < 0.25 and 0.08 < frac["test"] < 0.25
    for s in ("train", "val", "test"):
        assert df[df.split == s]["grade"].nunique() == 5  # every grade present in every split


def test_duplicates_stay_in_one_split(synth_root, tmp_path):
    shutil.copytree(synth_root, tmp_path / "d")
    t = pd.read_csv(tmp_path / "d" / "train.csv")
    extra = []
    for i in range(10):  # exact duplicates under new ids must not straddle splits
        shutil.copy(tmp_path / "d" / "train_images" / f"{t['id_code'][i]}.png", tmp_path / "d" / "train_images" / f"dup{i}.png")
        extra.append({"id_code": f"dup{i}", "diagnosis": t["diagnosis"][i]})
    pd.concat([t, pd.DataFrame(extra)]).to_csv(tmp_path / "d" / "train.csv", index=False)
    df = D.load_labelled_dataset(tmp_path / "d")
    ok, _ = D.build_cache(df, 48, tmp_path / "c", progress=False)
    ok = D.make_splits(ok, seed=1)
    D.check_no_leakage(ok)
    for i in range(10):
        a = ok.loc[ok.image_id == t["id_code"][i], "split"].iloc[0]
        b = ok.loc[ok.image_id == f"dup{i}", "split"].iloc[0]
        assert a == b


def test_dataset_item_shapes_and_eval_is_deterministic(prepared):
    df, cache = prepared
    labels = np.zeros(len(cache), int)
    labels[df.cache_idx.values] = df.grade.values
    idx = df.cache_idx.values[:4]
    ev = D.DRDataset(cache, labels, idx, train=False)
    x1, y1 = ev[0]
    x2, _ = ev[0]
    assert tuple(x1.shape) == (3, 64, 64) and (x1 == x2).all() and y1 == labels[idx[0]]
    tr = D.DRDataset(cache, labels, idx, train=True)
    assert tuple(tr[0][0].shape) == (3, 64, 64)
    a = np.asarray(cache[0])
    aug = D.augment_plain(a, np.random.default_rng(0))
    assert aug.shape == a.shape and aug.dtype == np.uint8


def test_submission_templates_are_never_used_as_labels(tmp_path, synth_root):
    """Regression: Kaggle's sample_submission.csv has id_code,diagnosis (all 0) for the UNLABELLED test images; using it
    silently gave 1,928 test images a fake grade-0 label in the first full run."""
    shutil.copytree(synth_root, tmp_path / "d")
    (tmp_path / "d" / "test_images").mkdir()
    ids = []
    for i in range(60):
        shutil.copy(next((tmp_path / "d" / "train_images").glob("*.png")), tmp_path / "d" / "test_images" / f"hidden{i}.png")
        ids.append(f"hidden{i}")
    pd.DataFrame({"id_code": ids, "diagnosis": 0}).to_csv(tmp_path / "d" / "sample_submission.csv", index=False)
    pd.DataFrame({"id_code": ids}).to_csv(tmp_path / "d" / "test.csv", index=False)
    pd.DataFrame({"id_code": ids, "diagnosis": 0}).to_csv(tmp_path / "d" / "dummy_labels.csv", index=False)  # constant label table
    df = D.load_labelled_dataset(tmp_path / "d")
    assert len(df) == 90 and not df["image_id"].str.startswith("hidden").any()
    assert set(df["source_csv"]) == {"train.csv"}


def test_robust_augmentation_is_training_only_valid_and_changes_images(prepared):
    df, cache = prepared
    a = np.asarray(cache[0])
    changed = 0
    for seed in range(40):
        r = D.augment_plain(a, np.random.default_rng(seed), robust=True)
        assert r.shape == a.shape and r.dtype == np.uint8
        b = D.augment_plain(a, np.random.default_rng(seed), robust=False)
        changed += int(np.abs(r.astype(int) - b.astype(int)).mean() > 0.5)
    assert changed > 5  # the extra blur/JPEG/illumination ops fire on a good share of draws
    labels = np.zeros(len(cache), int)
    labels[df.cache_idx.values] = df.grade.values
    idx = df.cache_idx.values[:4]
    ev = D.DRDataset(cache, labels, idx, train=False, robust=True)  # robust flag must not affect evaluation images
    assert (ev[0][0] == D.DRDataset(cache, labels, idx, train=False, robust=False)[0][0]).all()
    from drscreen.train import TrainConfig, make_loader

    cfg = TrainConfig(robust_aug=True, num_workers=0, batch_size=2)
    assert make_loader(cache, labels, idx, cfg, True, "cpu").dataset.robust is True
    assert make_loader(cache, labels, idx, cfg, False, "cpu").dataset.robust is False
