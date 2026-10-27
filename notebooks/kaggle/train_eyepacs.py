"""Kaggle entry point: does adding EyePACS (Kaggle 'Diabetic Retinopathy Detection', 2015) training data improve the APTOS classifier?

Inputs under /kaggle/input: the APTOS 2019 competition, the EyePACS competition (rules accepted by the account holder), the IDRiD dataset.
Design (fixed before any result):
  * APTOS split = exactly the split of the other classifiers (same code and seed); APTOS val is the ONLY set used for checkpoint choice,
    temperature and threshold; APTOS test, IDRiD grading test and a patient-level EyePACS test sample are evaluation only.
  * EyePACS is split by PATIENT (both eyes together). Training pool = all grade>=1 images of train patients + a random sample of grade-0 images.
    EyePACS images that are near-duplicates (dHash <= 6 bits) of any APTOS val/test or IDRiD test image are removed from training.
  * Phase A: mixed APTOS+EyePACS training (robustness augmentation as in the app's default model). Phase B: short low-LR fine-tune on APTOS only,
    starting from phase A's best checkpoint. Both models' logits are saved; the choice between them and against the current model is made
    locally by scripts/compare_eyepacs.py with a rule written beforehand.
`scripts/kaggle_submit.py` embeds the source and configuration.
"""
import base64
import io
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

try:
    _EMBEDDED_SRC_B64
except NameError:
    _EMBEDDED_SRC_B64 = None
try:
    _RUN_CONFIG
except NameError:
    _RUN_CONFIG = {}


def bootstrap_source():
    if _EMBEDDED_SRC_B64:
        dest = Path("/tmp/drsrc")
        dest.mkdir(parents=True, exist_ok=True)
        zipfile.ZipFile(io.BytesIO(base64.b64decode(_EMBEDDED_SRC_B64))).extractall(dest)
        sys.path.insert(0, str(dest))
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main():
    bootstrap_source()
    import numpy as np
    import pandas as pd
    import torch

    from drscreen import datasets as D
    from drscreen import eyepacs as E
    from drscreen import lesion as L
    from drscreen.pipeline import log
    from drscreen.preprocess import read_image
    from drscreen.train import TrainConfig, make_loader, predict_logits, train
    from drscreen.models import load_checkpoint

    c = {"smoke": False, "img_size": 384, "batch_size": 24, "epochs_a": 8, "epochs_b": 5, "lr_a": 3e-4, "lr_b": 5e-5, "n_grade0": 5000, "n_test": 3000,
         "max_minutes": 540, "seed": 42, "expected_n_labelled": 3662, **_RUN_CONFIG}
    smoke = bool(c["smoke"])
    on_kaggle = Path("/kaggle/working").exists()
    work = Path("/kaggle/working" if on_kaggle else "outputs_eyepacs")
    res, art = work / "results", work / "artifacts"
    res.mkdir(parents=True, exist_ok=True)
    cache_root = Path("/tmp/drcache_eyepacs")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path("/kaggle/input")
    t0 = time.time()
    size = int(c["img_size"])

    aptos = next((str(p.parent) for p in root.rglob("train.csv") if "aptos" in str(p).lower()), None)
    ep_root = next((p for p in root.rglob("*") if p.is_dir() and p.name == "diabetic-retinopathy-detection"), None)
    idrid = L.find_idrid_root(root)
    log("inputs:", {"aptos": aptos, "eyepacs": str(ep_root), "idrid": idrid, "device": device})
    if not (aptos and ep_root):
        raise SystemExit("missing inputs: need the APTOS and EyePACS competitions")

    # ------------------------------------------------------------------ APTOS (identical split to the other classifiers)
    df = D.load_labelled_dataset(aptos, "APTOS 2019")
    if smoke:
        df = df.groupby("grade", group_keys=False).apply(lambda g: g.sample(min(len(g), 12), random_state=c["seed"])).reset_index(drop=True)
    elif len(df) != c["expected_n_labelled"]:
        raise RuntimeError(f"APTOS label integrity check failed: {len(df)} != {c['expected_n_labelled']}")
    ap, ap_path = D.build_cache(df, size, cache_root / "aptos")
    ap = D.make_splits(ap, seed=c["seed"])
    n_ap = len(ap)
    ap["patient"] = ""

    # ------------------------------------------------------------------ EyePACS: patient-level split, pool, cache
    found = E.find_eyepacs(ep_root)
    lab = E.patient_split(E.load_labels(found["labels"]), seed=c["seed"])
    lab[["image", "grade", "patient", "eye", "ep_split"]].to_csv(res / "eyepacs_patient_split.csv", index=False)
    n0, nt = (60, 30) if smoke else (int(c["n_grade0"]), int(c["n_test"]))
    if smoke:
        lab = lab[lab.patient.isin(lab.patient.drop_duplicates().head(400))]
    pool = E.build_pool(lab, n_grade0_train=n0, n_test=nt, seed=c["seed"])
    if smoke:
        pool = pd.concat([pool[pool.split == "train"].groupby("grade", group_keys=False).head(8), pool[pool.split == "ep_test"].head(20)], ignore_index=True)
    pool["path"] = pool["image"]
    reader = E.EyePacsReader(found)
    log(f"EyePACS pool: {len(pool)} images ({(pool.split == 'train').sum()} train, {(pool.split == 'ep_test').sum()} test)")
    ep, ep_path = D.build_cache(pool, size, cache_root / "eyepacs", reader=reader.read, workers=4)
    log(f"EyePACS cached {len(ep)}/{len(pool)} in {(time.time() - t0) / 60:.1f} min")

    # ------------------------------------------------------------------ cross-dataset duplicate guard
    ref_hex = list(ap.loc[ap.split.isin(["val", "test"]), "dhash"])
    ref_md5 = set(ap.loc[ap.split.isin(["val", "test"]), "md5"])
    idrid_test = None
    if idrid:
        idrid_test = L.index_idrid(idrid)["grading"]["test"].dropna(subset=["path"]).reset_index(drop=True)
        if smoke:
            idrid_test = idrid_test.head(12)
        for p in idrid_test["path"]:
            img = read_image(p)
            ref_hex.append(np.packbits(D.dhash(img)).tobytes().hex())
            ref_md5.add(D.file_md5(p))
    dup = E.near_duplicates(E.hash_bits(ep["dhash"]), E.hash_bits(ref_hex), 6) | ep["md5"].isin(ref_md5).to_numpy()
    dedup = {"n_eyepacs_checked": int(len(ep)), "n_reference_images": len(ref_hex), "n_removed_near_duplicates": int(dup.sum()),
             "removed_in_train": int((dup & (ep["split"] == "train").to_numpy()).sum()), "removed_in_ep_test": int((dup & (ep["split"] == "ep_test").to_numpy()).sum())}
    log("duplicate guard:", dedup)
    ep = ep[~dup].copy()

    # ------------------------------------------------------------------ combined cache + table
    total = n_ap + len(np.load(ep_path, mmap_mode="r"))
    comb_path = cache_root / f"combined_{size}.npy"
    comb = np.lib.format.open_memmap(comb_path, mode="w+", dtype=np.uint8, shape=(total, size, size, 3))
    a = np.load(ap_path, mmap_mode="r")
    e = np.load(ep_path, mmap_mode="r")
    for off, src in ((0, a), (n_ap, e)):
        for i in range(0, len(src), 500):
            chunk = np.asarray(src[i : i + 500])
            comb[off + i : off + i + len(chunk)] = chunk
    comb.flush()
    del comb
    cache = np.load(comb_path, mmap_mode="r")
    ep["cache_idx"] = ep["cache_idx"] + n_ap
    keep = ["image_id", "grade", "split", "cache_idx", "dataset", "patient"]
    both = pd.concat([ap[keep], ep[keep]], ignore_index=True)
    labels = np.zeros(len(cache), int)
    labels[both["cache_idx"].to_numpy()] = both["grade"].to_numpy()
    summary = {"aptos": D.split_summary(ap),
               "eyepacs_train_grades": {str(g): int(((ep.split == "train") & (ep.grade == g)).sum()) for g in range(5)},
               "eyepacs_test_grades": {str(g): int(((ep.split == "ep_test") & (ep.grade == g)).sum()) for g in range(5)},
               "eyepacs_train_patients": int(ep[ep.split == "train"].patient.nunique()), "eyepacs_test_patients": int(ep[ep.split == "ep_test"].patient.nunique()),
               "dedup": dedup, "prep_minutes": round((time.time() - t0) / 60, 1)}
    (res / "eyepacs_data_summary.json").write_text(json.dumps(summary, indent=2))
    ap[["image_id", "grade", "split", "group", "md5"]].to_csv(res / "aptos_split_manifest.csv", index=False)
    log("data summary:", json.dumps(summary)[:600])

    # ------------------------------------------------------------------ phase A (mixed) and phase B (APTOS fine-tune)
    both_train = both[(both.split == "train") | (both.split == "val")]
    budget = float(c["max_minutes"])
    ea, eb = (1, 1) if smoke else (int(c["epochs_a"]), int(c["epochs_b"]))
    common = dict(arch="efficientnet_b0", img_size=size, batch_size=int(c["batch_size"]), num_workers=4, seed=int(c["seed"]), robust_aug=True,
                  out_dir=str(art), max_train_batches=2 if smoke else None, pretrained=True)
    cfg_a = TrainConfig(**common, epochs=ea, lr=float(c["lr_a"]), run_name="efficientnet_b0_eyepacs_mixed", patience=4,
                        max_minutes=max(30.0, budget - (time.time() - t0) / 60 - 40))
    log("=== phase A: mixed APTOS + EyePACS")
    sa = train(cfg_a, cache, both_train, device, log)
    cfg_b = TrainConfig(**{**common, "pretrained": False}, epochs=eb, lr=float(c["lr_b"]), warmup_epochs=0.5, run_name="efficientnet_b0_eyepacs", patience=5,
                        init_from=str(art / "efficientnet_b0_eyepacs_mixed" / "best.pt"), max_minutes=max(15.0, budget - (time.time() - t0) / 60 - 10))
    log("=== phase B: APTOS-only fine-tune")
    ap_only = both[(both.dataset == "APTOS 2019") & both.split.isin(["train", "val"])]
    sb = train(cfg_b, cache, ap_only, device, log)

    # ------------------------------------------------------------------ logits of both models on all evaluation sets
    def cached_logits(model, tcfg, idx):
        ld = make_loader(cache, labels, np.asarray(idx), tcfg, False, device)
        lg, y = predict_logits(model, ld, device, tcfg.amp)
        return lg, y

    sets = {"val": both[(both.dataset == "APTOS 2019") & (both.split == "val")], "test": both[(both.dataset == "APTOS 2019") & (both.split == "test")],
            "eyepacs_test": both[both.split == "ep_test"]}
    out_arrays = {}
    for run, tcfg in (("efficientnet_b0_eyepacs_mixed", cfg_a), ("efficientnet_b0_eyepacs", cfg_b)):
        model, ck = load_checkpoint(art / run / "best.pt", device)
        for name, d in sets.items():
            lg, y = cached_logits(model, tcfg, d["cache_idx"].to_numpy())
            out_arrays.setdefault(name, {"image_id": d["image_id"].to_numpy().astype(str), "grade": y})[run] = lg
        if idrid_test is not None:
            xs = []
            from drscreen.preprocess import preprocess_for_model

            for p in idrid_test["path"]:
                _, _, xin = preprocess_for_model(read_image(p), size, True)
                xs.append(xin)
            lgs = []
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=device == "cuda"):
                for i in range(0, len(xs), 16):
                    lgs.append(model(torch.from_numpy(np.stack(xs[i : i + 16])).to(device)).float().cpu().numpy())
            out_arrays.setdefault("idrid_test", {"image_id": idrid_test["image_id"].to_numpy().astype(str), "grade": idrid_test["grade"].to_numpy(int)})[run] = np.concatenate(lgs)
    # the current default (robust B0, trained on APTOS only) on the EyePACS test sample: an external-domain reference for it
    rob = next((p for p in root.rglob("best.pt") if p.parent.name == "efficientnet_b0_robust"), None)
    if rob is not None and "eyepacs_test" in out_arrays:
        rmodel, rck = load_checkpoint(rob, device)
        if int(rck["cfg"]["img_size"]) == size:
            out_arrays["eyepacs_test"]["efficientnet_b0_robust"] = cached_logits(rmodel, cfg_a, sets["eyepacs_test"]["cache_idx"].to_numpy())[0]
        else:
            log("robust reference skipped: different input size")
    for name, arrs in out_arrays.items():
        np.savez_compressed(res / f"logits_{name}.npz", **arrs)
        log("saved logits", name, len(arrs["grade"]))
    for run in ("efficientnet_b0_eyepacs_mixed", "efficientnet_b0_eyepacs"):
        shutil.copy(art / run / "history.csv", res / f"history_{run}.csv")
        for junk in ("last.pt",):  # keep the output small: only best.pt + config
            (art / run / junk).unlink(missing_ok=True)
    (res / "DONE_eyepacs.json").write_text(json.dumps({"finished": True, "phase_a": sa, "phase_b": sb, "sets": {k: len(v["grade"]) for k, v in out_arrays.items()},
                                                       "total_minutes": round((time.time() - t0) / 60, 1), "smoke": smoke}, default=str, indent=2))
    log("finished in", round((time.time() - t0) / 60, 1), "min")


if __name__ == "__main__":
    main()
