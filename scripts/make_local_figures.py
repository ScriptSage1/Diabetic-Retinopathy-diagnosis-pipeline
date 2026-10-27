"""Regenerate the image-based report figures locally from the saved classifier outputs, downloading only the ~30 APTOS images needed.

    python scripts/make_local_figures.py

Uses results/predictions.csv (test split) to pick the same kinds of examples the remote run showed (a correct and a wrong prediction per true
grade; the most confident false negatives / false positives), downloads those single images from the competition into a cache folder OUTSIDE
the OneDrive project (LOCALAPPDATA/drscreen/aptos_examples), and recomputes Grad-CAM on CPU with the real checkpoint. Overwrites
report/figures/{gradcam_examples,error_examples,preprocessing_examples,quality_examples}.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _kaggle_common import COMPETITION, CACHE_DIR, run_kaggle  # noqa: E402
from drscreen import metrics as M  # noqa: E402
from drscreen import plots as P  # noqa: E402
from drscreen.gradcam import GradCAM, overlay_heatmap  # noqa: E402
from drscreen.models import load_checkpoint  # noqa: E402
from drscreen.preprocess import preprocess_plain, read_image, to_model_input, to_tensor_array  # noqa: E402
from drscreen.quality import DEGRADATIONS, QualityThresholds, assess_quality, degrade  # noqa: E402

EX_DIR = CACHE_DIR.parent / "aptos_examples"


def fetch(image_id: str) -> Path:
    EX_DIR.mkdir(parents=True, exist_ok=True)
    dest = EX_DIR / f"{image_id}.png"
    if not dest.exists():
        r = run_kaggle(["competitions", "download", COMPETITION, "-f", f"train_images/{image_id}.png", "-p", str(EX_DIR), "-q"], timeout=300)
        got = list(EX_DIR.rglob(f"{image_id}.png"))
        if not got:
            import zipfile

            for z in EX_DIR.rglob(f"{image_id}.png.zip"):  # single-file downloads arrive as <folder>/<name>.png.zip
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(EX_DIR)
                z.unlink()
            got = list(EX_DIR.rglob(f"{image_id}.png"))
        if not got:
            raise RuntimeError(f"could not download {image_id}: {(r.stdout + r.stderr)[-300:]}")
        if got[0] != dest:
            got[0].replace(dest)
    return dest


def main():
    figs = ROOT / "report" / "figures"
    pred = pd.read_csv(ROOT / "results" / "predictions.csv")
    te = pred[pred["split"] == "test"].reset_index(drop=True)
    met = json.loads((ROOT / "results" / "metrics.json").read_text())
    thr = met["validation_threshold_selection"]["threshold"]
    model, ck = load_checkpoint(ROOT / "artifacts" / "efficientnet_b0" / "best.pt", "cpu")
    size, bg = ck["cfg"]["img_size"], ck["cfg"].get("use_ben_graham", True)
    qthr = QualityThresholds(**json.loads((ROOT / "artifacts" / "efficientnet_b0" / "calibration.json").read_text())["quality_thresholds"])

    picks, tags = [], []
    for g in range(5):
        for cond, tag in (((te.true_grade == g) & (te.pred_grade == g), "correct"), ((te.true_grade == g) & (te.pred_grade != g), "wrong")):
            idx = te.index[cond]
            if len(idx):
                picks.append(int(idx[0]))
                tags.append(tag)
    pr = te["p_referable"].to_numpy()
    yb = te["true_referable"].to_numpy()
    fn = np.where((yb == 1) & (pr < thr))[0]
    fp = np.where((yb == 0) & (pr >= thr))[0]
    fn, fp = fn[np.argsort(pr[fn])][:5], fp[np.argsort(-pr[fp])][:5]
    first_per_grade = [int(te.index[te.true_grade == g][0]) for g in range(5)]
    need = sorted({te.image_id[i] for i in [*picks, *fn, *fp, *first_per_grade]})
    print(f"downloading {len(need)} example images to {EX_DIR} ...", flush=True)
    for iid in need:
        fetch(iid)

    def cam_overlay(iid):
        plain = preprocess_plain(read_image(EX_DIR / f"{iid}.png"), size)
        x = torch.from_numpy(to_tensor_array(to_model_input(plain, bg)))[None]
        gc = GradCAM(model)
        cam, _ = gc(x, "pred")
        gc.remove()
        return overlay_heatmap(plain, cam)

    ims, titles = [], []
    for i, tag in zip(picks, tags):
        r = te.loc[i]
        ims.append(cam_overlay(r.image_id))
        titles.append(f"{tag}: true {r.true_grade} / pred {r.pred_grade}\nP(ref)={r.p_referable:.2f}")
    P.plot_image_grid(ims, titles, figs / "gradcam_examples.png", ncols=5, figw=2.3,
                      suptitle="Grad-CAM overlays (classifier attention, not lesion detection): one correct and one wrong test prediction per true grade")
    ims, titles = [], []
    for i, tag in [(i, "FALSE NEGATIVE") for i in fn] + [(i, "FALSE POSITIVE") for i in fp]:
        r = te.loc[i]
        ims.append(cam_overlay(r.image_id))
        titles.append(f"{tag}: true {r.true_grade} / pred {r.pred_grade}\nP(ref)={r.p_referable:.2f}")
    P.plot_image_grid(ims, titles, figs / "error_examples.png", ncols=5, figw=2.3, suptitle="Most confident errors at the operating threshold (Grad-CAM overlay)")

    rows = [[], [], [], []]
    for g, i in enumerate(first_per_grade):
        raw = read_image(EX_DIR / f"{te.image_id[i]}.png")
        plain = preprocess_plain(raw, size)
        rows[0].append(cv2.resize(raw, (size, size), interpolation=cv2.INTER_AREA))
        rows[1].append(plain)
        rows[2].append(to_model_input(plain, True))
        rows[3].append(cv2.cvtColor(plain[..., 1], cv2.COLOR_GRAY2RGB))
    P.plot_image_grid([im for row in rows for im in row], [M.GRADE_NAMES[g] for g in range(5)] + [""] * 15, figs / "preprocessing_examples.png", ncols=5, figw=2.2,
                      suptitle="Rows: original (resized for display) | cropped + masked + resized | Ben Graham normalised (model input) | green channel")
    raw = None
    for iid in need:  # an illustration only makes sense if the ORIGINAL image passes the gate
        cand = read_image(EX_DIR / f"{iid}.png")
        if assess_quality(cand, qthr).status == "good":
            raw = cand
            break
    if raw is None:
        raise RuntimeError("no downloaded example passes the quality gate")
    imgs = [raw] + [degrade(raw, k, 2, 42) for k in DEGRADATIONS]
    P.plot_image_grid([cv2.resize(i, (256, 256)) for i in imgs], ["original\n" + assess_quality(raw, qthr).status] + [f"{k} (sev 2)\n{assess_quality(degrade(raw, k, 2, 42), qthr).status}" for k in DEGRADATIONS],
                      figs / "quality_examples.png", ncols=7, figw=1.9, suptitle="Synthetic degradations and the gate's verdict")
    print("wrote 4 figures")


if __name__ == "__main__":
    main()
