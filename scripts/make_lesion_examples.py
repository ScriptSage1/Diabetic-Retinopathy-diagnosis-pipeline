"""Regenerate the APTOS lesion-example figure and a worked example report with the DEPLOYED pipeline (CPU, ~10 images).

    python scripts/make_lesion_examples.py

Selection is not cherry-picked: the first two APTOS TEST images of each true grade (order of results/predictions.csv). Images are downloaded one
by one into LOCALAPPDATA/drscreen/aptos_examples (outside OneDrive). Writes report/figures/lesion_examples_aptos.png,
report/figures/example_report.png and results/example_report.md (the full text the app shows for one image).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_local_figures import EX_DIR, fetch  # noqa: E402
from drscreen import plots as P  # noqa: E402
from drscreen.predict import Predictor  # noqa: E402


def main():
    figs = ROOT / "report" / "figures"
    te = pd.read_csv(ROOT / "results" / "predictions.csv")
    te = te[te.split == "test"].reset_index(drop=True)
    picks = []
    for g in range(5):
        picks += [int(i) for i in te.index[te.true_grade == g][:2]]
    for i in picks:
        fetch(te.image_id[i])
    # standard checkpoint so the example matches the headline model's numbers; lesion models from artifacts/lesion
    p = Predictor(ROOT / "artifacts" / "efficientnet_b0" / "best.pt", device="cpu", lesion_dir=ROOT / "artifacts" / "lesion")
    assert p.lesions is not None, "lesion models not found in artifacts/lesion"
    ims, titles, results = [], [], {}
    for i in picks:
        r = te.loc[i]
        res = p.predict(EX_DIR / f"{r.image_id}.png", with_gradcam=True, force=True)
        results[i] = res
        ev = res.lesion_evidence
        if res.status != "ok" or not ev.get("available"):
            ims.append(np.zeros((420, 420, 3), np.uint8))
            titles.append(f"true grade {r.true_grade}\n{res.status}")
            continue
        f = ev["features"]
        ims.append(cv2.resize(res.lesion_overlay, (420, 420), interpolation=cv2.INTER_AREA))
        titles.append(f"true grade {r.true_grade} | classifier {res.grade} | rule {ev['rule']['grade']}\nMA {f['MA_count']} HE {f['HE_count']} EX {f['EX_count']} SE {f['SE_count']}")
    P.plot_image_grid(ims, titles, figs / "lesion_examples_aptos.png", ncols=5, figw=2.8,
                      suptitle="Lesion evidence on the first two APTOS test images per grade (domain transfer; no APTOS lesion annotations): red=MA, orange=HE, yellow=EX, cyan=SE; blue ring=disc, green cross=fovea")
    # worked example: first grade-3 test image
    k = next(i for i in picks if te.true_grade[i] == 3)
    res = results[k]
    panels = [cv2.resize(x, (480, 480), interpolation=cv2.INTER_AREA) for x in (res.original if res.original is not None else np.zeros((10, 10, 3), np.uint8),
                                                                                  res.gradcam, res.lesion_overlay) if x is not None]
    P.plot_image_grid(panels, ["original photograph", "Grad-CAM (classifier attention)", "detected lesions + disc/fovea/quadrant axes"][: len(panels)],
                      figs / "example_report.png", ncols=3, figw=3.6, suptitle=f"Worked example: APTOS test image {te.image_id[k]} (true grade {te.true_grade[k]})")
    (ROOT / "results" / "example_report.md").write_text(f"Image: APTOS test image `{te.image_id[k]}` (true grade {te.true_grade[k]}), first grade-3 image in the test split.\n\n" + res.to_markdown(), encoding="utf-8")
    agree = sum(1 for i in picks if results[i].status == "ok" and results[i].lesion_evidence.get("available") and results[i].lesion_evidence["rule"]["grade"] == te.true_grade[i])
    print(f"wrote figures and example report; rule grade equals the true grade for {agree}/{len(picks)} of these example images")


if __name__ == "__main__":
    main()
