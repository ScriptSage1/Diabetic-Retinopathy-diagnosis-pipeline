"""Kaggle entry point: run several trained classifier checkpoints (with and without flip test-time augmentation) on the APTOS validation and test
images and on the IDRiD grading test images, and save the raw logits. Used to select and evaluate ensembles / TTA on CPU afterwards
(scripts/evaluate_ensemble.py): one image read per image, every model preprocesses it at its own resolution.

Inputs under /kaggle/input: the APTOS competition (rules accepted), the IDRiD dataset, and the output of each classifier kernel (best.pt +
calibration.json under artifacts/<run>/, and the split manifest). `scripts/kaggle_submit.py build --mode infer` embeds the source and configuration.
"""
import base64
import io
import json
import os
import sys
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

    from drscreen import lesion as L
    from drscreen.datasets import load_labelled_dataset
    from drscreen.models import load_checkpoint
    from drscreen.pipeline import log
    from drscreen.preprocess import preprocess_for_model, read_image

    on_kaggle = Path("/kaggle/working").exists()
    out = Path("/kaggle/working" if on_kaggle else "outputs_infer") / "results"
    out.mkdir(parents=True, exist_ok=True)
    root = Path("/kaggle/input")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    smoke = bool(_RUN_CONFIG.get("smoke"))

    aptos = next((str(p.parent) for p in root.rglob("train.csv") if "aptos" in str(p).lower()), None)
    idrid = L.find_idrid_root(root)
    manifest = next(root.rglob("split_manifest.csv"), None)
    models = {}
    for ck in sorted(root.rglob("best.pt")):
        run = ck.parent.name
        if run.startswith("efficientnet") and run not in models:
            models[run] = ck
    log("inputs:", {"aptos": aptos, "idrid": idrid, "manifest": str(manifest), "models": {k: str(v) for k, v in models.items()}})
    if not (aptos and manifest and models):
        raise SystemExit("missing inputs: need the APTOS competition, a split manifest and at least one classifier checkpoint")

    df = load_labelled_dataset(aptos, "APTOS 2019")
    man = pd.read_csv(manifest)[["image_id", "split"]]
    df = df.merge(man, on="image_id", how="inner")
    sets = {"val": df[df.split == "val"].reset_index(drop=True), "test": df[df.split == "test"].reset_index(drop=True)}
    if idrid:
        g = L.index_idrid(idrid)["grading"]["test"].dropna(subset=["path"]).reset_index(drop=True)
        sets["idrid_test"] = g.rename(columns={"grade": "grade"})
    if smoke:
        sets = {k: v.head(12) for k, v in sets.items()}
    loaded = {run: load_checkpoint(ck, device) for run, ck in models.items()}
    for run, (m, ck) in loaded.items():
        log(run, ck["cfg"]["img_size"], ck["cfg"]["arch"])

    def logits_for(model, x):
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=device == "cuda"):
            return model(x).float().cpu().numpy()

    for name, d in sets.items():
        res = {run: {"plain": [], "flip": []} for run in loaded}
        for i, p in enumerate(d["path"]):
            img = read_image(p)
            for run, (m, ck) in loaded.items():
                _, _, xin = preprocess_for_model(img, ck["cfg"]["img_size"], ck["cfg"].get("use_ben_graham", True))
                x = torch.from_numpy(xin)[None].to(device)
                res[run]["plain"].append(logits_for(m, x)[0])
                res[run]["flip"].append(np.mean([logits_for(m, torch.flip(x, [3]))[0], logits_for(m, torch.flip(x, [2]))[0]], axis=0))  # h-flip and v-flip views
            if i % 100 == 0:
                log(f"{name}: {i}/{len(d)}")
        arrays = {"image_id": d["image_id"].to_numpy().astype(str), "grade": d["grade"].to_numpy(int)}
        for run in loaded:
            arrays[f"{run}__plain"] = np.stack(res[run]["plain"])
            arrays[f"{run}__flip"] = np.stack(res[run]["flip"])
        np.savez_compressed(out / f"logits_{name}.npz", **arrays)
        log("saved", name, len(d))
    cal = {}
    for run, ck in models.items():
        f = ck.parent / "calibration.json"
        cal[run] = json.loads(f.read_text()) if f.is_file() else {}
    (out / "model_calibrations.json").write_text(json.dumps(cal, indent=2))
    (out / "DONE_infer.json").write_text(json.dumps({"finished": True, "models": list(models), "sets": {k: len(v) for k, v in sets.items()}}))


if __name__ == "__main__":
    main()
