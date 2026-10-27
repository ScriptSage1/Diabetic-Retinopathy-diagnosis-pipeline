"""Compare lesion model v2 (IDRiD + e-ophtha, partial labels, acquisition augmentation, large-region filter) with the deployed v1.

    python scripts/compare_lesion_v2.py [--v2-dir DIR] [--dry-run]

DIR defaults to LOCALAPPDATA/drscreen/kaggle_output_lesion2 (output of `kaggle kernels output candyguy/dr-lesion-evidence-v2`).
Decision rule (fixed BEFORE the results were seen) - adopt v2 if BOTH hold:
  (a) mean AUPR over MA/HE/EX/SE on the IDRiD TEST images is not more than 0.03 below v1  (it must remain a competent lesion detector), and
  (b) the calibrated rule engine's QWK on the APTOS VALIDATION rows (thresholds fitted on TRAIN rows only) improves by >= 0.03
      (the lesion evidence agrees better with expert grades; the only indirect measure available for APTOS lesions).
Reports test-split numbers too, but test rows play no part in the decision.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drscreen import metrics as M  # noqa: E402
from drscreen.lesion_eval import fit_gb, gb_probs  # noqa: E402
from drscreen.rules import fit_rule_params, grade_vector  # noqa: E402


def mean_aupr(seg: dict) -> float:
    v = [seg["per_lesion"][k]["aupr"] for k in ("MA", "HE", "EX", "SE") if seg["per_lesion"][k]["aupr"] is not None and not math.isnan(seg["per_lesion"][k]["aupr"])]
    return float(np.mean(v)) if v else float("nan")


def apt_summary(feats: pd.DataFrame, manifest: pd.DataFrame) -> dict:
    f = feats.drop(columns=["split"], errors="ignore").merge(manifest[["image_id", "split"]], on="image_id", how="inner")
    f = f[f.loc_ok == 1]
    tr, va, te = (f[f.split == s] for s in ("train", "val", "test"))
    params = fit_rule_params(tr, tr.grade.to_numpy(int))
    q = lambda d: M.quadratic_weighted_kappa(d.grade.to_numpy(int), grade_vector(d, params))
    gb = fit_gb(tr)
    auc = lambda d: M.binary_metrics(M.referable_labels(d.grade), M.referable_prob(gb_probs(gb, d)), 0.5)["roc_auc"]
    g0 = f[f.grade == 0]
    return {"rule_params": params.to_dict(), "rules_val_qwk": q(va), "rules_test_qwk": q(te), "gb_val_auc": auc(va), "gb_test_auc": auc(te),
            "grade0_share_with_HE_ge_20": float((g0.HE_count >= 20).mean()), "grade0_mean_EX_count": float(g0.EX_count.mean()),
            "share_images_with_large_regions": float(((f[[c for c in f.columns if c.endswith("_large")]].sum(axis=1) > 0).mean())) if any(c.endswith("_large") for c in f.columns) else None}


def main():
    default = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".cache"))) / "drscreen" / "kaggle_output_lesion2"
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-dir", default=str(default))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    v2 = Path(a.v2_dir) / "results"
    manifest = pd.read_csv(ROOT / "artifacts" / "split_manifest.csv")
    m1 = json.loads((ROOT / "results" / "metrics_lesion.json").read_text())
    m2 = json.loads((v2 / "metrics_lesion.json").read_text())
    f1 = pd.read_csv(ROOT / "results" / "lesion_features_aptos.csv")
    f2 = pd.read_csv(v2 / "lesion_features_aptos.csv")
    s1, s2 = m1["lesion_segmentation_idrid_test"], m2["lesion_segmentation_idrid_test"]
    a1, a2 = mean_aupr(s1), mean_aupr(s2)
    p1, p2 = apt_summary(f1, manifest), apt_summary(f2, manifest)
    ok_a = a2 >= a1 - 0.03
    gain = p2["rules_val_qwk"] - p1["rules_val_qwk"]
    ok_b = gain >= 0.03
    out = {"idrid_test_mean_aupr": {"v1": a1, "v2": a2, "per_lesion_v1": {k: s1["per_lesion"][k]["aupr"] for k in ("MA", "HE", "EX", "SE", "OD")}, "per_lesion_v2": {k: s2["per_lesion"][k]["aupr"] for k in ("MA", "HE", "EX", "SE", "OD")}},
           "eophtha_test_v2": m2.get("lesion_segmentation_eophtha_test"), "aptos_v1": p1, "aptos_v2": p2,
           "decision": {"rule": "see scripts/compare_lesion_v2.py (fixed before the results were seen)", "idrid_not_worse_than_0.03": bool(ok_a), "aptos_val_rule_qwk_gain": float(gain),
                        "gain_at_least_0.03": bool(ok_b), "adopt_v2": bool(ok_a and ok_b)},
           "training_v2": m2.get("lesion_training"), "locator_v2": m2.get("locator_final_idrid_test")}
    r3 = lambda v: "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"
    print("IDRiD test AUPR   v1", {k: r3(v) for k, v in out["idrid_test_mean_aupr"]["per_lesion_v1"].items()}, "mean", r3(a1))
    print("IDRiD test AUPR   v2", {k: r3(v) for k, v in out["idrid_test_mean_aupr"]["per_lesion_v2"].items()}, "mean", r3(a2))
    if out["eophtha_test_v2"]:
        print("e-ophtha test (v2):", {k: r3(v["aupr"]) for k, v in out["eophtha_test_v2"]["per_lesion"].items()})
    for tag, p in (("v1", p1), ("v2", p2)):
        print(f"APTOS {tag}: rules val QWK {r3(p['rules_val_qwk'])} test {r3(p['rules_test_qwk'])} | GB val AUC {r3(p['gb_val_auc'])} test {r3(p['gb_test_auc'])} | grade-0 images with >=20 HE: {r3(p['grade0_share_with_HE_ge_20'])} | large-region share {r3(p['share_images_with_large_regions'])}")
    print("decision:", json.dumps(out["decision"]))
    if not a.dry_run:
        mp = ROOT / "results" / "metrics.json"
        met = json.loads(mp.read_text())
        met["lesion_v2_comparison"] = out
        M.save_json(met, mp)


if __name__ == "__main__":
    main()
