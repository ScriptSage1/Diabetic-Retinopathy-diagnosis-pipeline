"""Evaluate the lesion-rule path and its fusion with the classifier from the saved remote outputs (CPU only, seconds).

    python scripts/evaluate_lesion_fusion.py [--results results] [--figures report/figures]

Needs (from the Kaggle runs): results/predictions.csv + results/metrics.json (classifier run), results/lesion_features_aptos.csv,
results/lesion_features_idrid.csv, results/idrid_cnn_probs.csv and results/metrics_lesion.json (lesion run).
Writes the findings into results/metrics.json under 'lesion_evidence' and 'lesion_fusion', and figures into the figures folder.
Also stores the validation-selected rule thresholds in artifacts/efficientnet_b0/calibration.json so the app uses them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drscreen import metrics as M  # noqa: E402
from drscreen import plots as P  # noqa: E402
from drscreen.lesion_eval import evaluate_idrid_external, evaluate_lesion_fusion  # noqa: E402
from drscreen.rules import RuleParams, grade_vector  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--figures", default=str(ROOT / "report" / "figures"))
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    res, figs = Path(a.results), Path(a.figures)
    met = json.loads((res / "metrics.json").read_text())
    thr = met["validation_threshold_selection"]["threshold"]
    preds = pd.read_csv(res / "predictions.csv")
    fa = pd.read_csv(res / "lesion_features_aptos.csv")
    manifest = ROOT / "artifacts" / "split_manifest.csv"
    if fa["split"].isna().all() and manifest.is_file():  # the lesion run does not know the classifier's splits: join them here
        fa = fa.drop(columns=["split"]).merge(pd.read_csv(manifest)[["image_id", "split"]], on="image_id", how="left")
    out = evaluate_lesion_fusion(fa, preds, thr, n_boot=a.n_boot)

    # which rule variant does the app use? chosen on VALIDATION rows by QWK against the APTOS grade
    va = fa[(fa["split"] == "val") & (fa["loc_ok"] == 1)]
    q = {k: M.quadratic_weighted_kappa(va["grade"].to_numpy(int), grade_vector(va, RuleParams(**out["rule_params"][k]))) for k in ("literal", "calibrated")}
    chosen = max(q, key=lambda k: -9 if np.isnan(q[k]) else q[k])
    out["rule_variant_for_app"] = {"chosen": chosen, "validation_qwk": q, "selected_on": "validation"}

    ext = None
    fi, cp = res / "lesion_features_idrid.csv", res / "idrid_cnn_probs.csv"
    if fi.is_file():
        ext = evaluate_idrid_external(pd.read_csv(fi), pd.read_csv(cp) if cp.is_file() else None, fa, thr, out["rule_params"], a.n_boot)

    lm = res / "metrics_lesion.json"
    lesion_run = json.loads(lm.read_text()) if lm.is_file() else {}
    met["lesion_evidence"] = {"segmentation_idrid_test": lesion_run.get("lesion_segmentation_idrid_test"), "locator_idrid_test": lesion_run.get("locator_idrid_test"),
                              "locator_final_idrid_test": lesion_run.get("locator_final_idrid_test"), "training": lesion_run.get("lesion_training"), "idrid": lesion_run.get("idrid"), "thresholds": lesion_run.get("lesion_thresholds_fitted_on_validation"),
                              "environment": lesion_run.get("environment"), "aptos_features": lesion_run.get("aptos_features")}
    out["aptos_locator_fallback_share"] = float(fa["loc_fallback"].mean()) if "loc_fallback" in fa else None
    met["lesion_fusion"] = out
    if ext:
        met["external_idrid"] = ext
    M.save_json(met, res / "metrics.json")

    for cal_path in sorted((ROOT / "artifacts").glob("efficientnet_b0*/calibration.json")):  # standard and robust checkpoints
        cal = json.loads(cal_path.read_text())
        cal["rule_params"] = out["rule_params"][chosen]
        cal_path.write_text(json.dumps(cal, indent=2))

    figs.mkdir(parents=True, exist_ok=True)
    rows = [r for r in out["rows"] if not np.isnan(r["roc_auc"] if r["roc_auc"] is not None else np.nan)]
    nl = chr(10)

    def short(n):  # short two-line labels so the x axis stays readable
        for key, lab in (("classifier alone", f"Classifier{nl}alone"), ("ICDR-literal", f"Lesion rules{nl}(ICDR counts)"), ("data-calibrated", f"Lesion rules{nl}(calibrated)"),
                         ("gradient boosting", f"Lesion-feature{nl}grader"), ("Fusion", f"Fusion{nl}(classifier + grader)")):
            if key in n:
                return lab
        return n

    P.plot_bars([short(r["name"]) for r in rows], [r["roc_auc"] for r in rows], figs / "lesion_fusion_auc.png",
                "Referable-DR ROC-AUC: classifier vs lesion path vs fusion (same test images)", "ROC-AUC", ylim=(0, 1.1))
    for r in out["rows"]:
        print(f"{r['name'][:70]:72s} sens {r['sensitivity']:.3f} spec {r['specificity']:.3f} auc {r['roc_auc'] if r['roc_auc'] is not None else float('nan'):.3f} qwk {r['qwk'] if r['qwk'] is not None else float('nan'):.3f}")
    print("rule variant for app:", chosen, q)
    if ext and "rows" in ext:
        print(f"IDRiD external: n={ext['n_evaluated']}")
        for r in ext["rows"]:
            print(f"  {r['name'][:60]:62s} sens {r['sensitivity']:.3f} spec {r['specificity']:.3f} auc {r['roc_auc']:.3f} qwk {r['qwk']:.3f}")


if __name__ == "__main__":
    main()
