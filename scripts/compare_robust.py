"""Compare the standard classifier with the robustness-augmented one (same split, architecture, schedule; only training augmentation differs).

    python scripts/compare_robust.py [--robust-dir DIR]

Reads the robust run's output (default: LOCALAPPDATA/drscreen/kaggle_output_robust, downloaded with
`kaggle kernels output candyguy/dr-screening-effnet-b0-aptos-robust`) and the standard run in results/. Aborts if the two runs did not use the
identical train/val/test split. Writes results/metrics.json['robustness_comparison'] and report/figures/robustness_comparison.png,
installs the robust checkpoint at artifacts/efficientnet_b0_robust/ (best.pt + calibration.json only).

Decision rule (fixed BEFORE seeing the results): recommend the robust model if
  (a) on the clean test split the paired-bootstrap 95% CI lower bounds are > -0.010 for the ROC-AUC difference and > -0.020 for the QWK difference, and
  (b) the mean ungated balanced accuracy over the degraded conditions (blur, dark, overexposed, low_contrast, jpeg) improves by >= 0.05.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drscreen import metrics as M  # noqa: E402
from drscreen import plots as P  # noqa: E402

DEGRADED = ("blur", "dark", "overexposed", "low_contrast", "jpeg")


def main():
    default_dir = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".cache"))) / "drscreen" / "kaggle_output_robust"
    ap = argparse.ArgumentParser()
    ap.add_argument("--robust-dir", default=str(default_dir))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true", help="print the comparison only; write/copy nothing")
    a = ap.parse_args()
    rd = Path(a.robust_dir)
    mine = json.loads((ROOT / "results" / "metrics.json").read_text())
    rob = json.loads((rd / "results" / "metrics.json").read_text())

    # 1) identical split?
    s1 = pd.read_csv(ROOT / "artifacts" / "split_manifest.csv")[["image_id", "split"]].sort_values("image_id").reset_index(drop=True)
    s2 = pd.read_csv(rd / "artifacts" / "split_manifest.csv")[["image_id", "split"]].sort_values("image_id").reset_index(drop=True)
    if not s1.equals(s2):
        raise SystemExit("ABORT: the two runs used different data splits; the comparison would be invalid.")

    # 2) clean test split, paired on the same images
    p1 = pd.read_csv(ROOT / "results" / "predictions.csv")
    p2 = pd.read_csv(rd / "results" / "predictions.csv")
    pc = [f"p_grade{k}" for k in range(5)]
    t1, t2 = p1[p1.split == "test"].set_index("image_id"), p2[p2.split == "test"].set_index("image_id")
    ids = sorted(set(t1.index) & set(t2.index))
    assert len(ids) == len(t1) == len(t2), "test sets differ"
    y = t1.loc[ids, "true_grade"].to_numpy(int)
    A, B = t1.loc[ids, pc].to_numpy(), t2.loc[ids, pc].to_numpy()
    thr_a = mine["validation_threshold_selection"]["threshold"]
    thr_b = rob["validation_threshold_selection"]["threshold"]
    paired = {m: M.paired_bootstrap_diff(y, B, A, m, a.n_boot, 0, thr_b, thr_a) for m in ("roc_auc", "qwk", "sensitivity", "specificity")}  # robust minus standard
    clean = {"standard": mine["internal_holdout"], "robust": rob["internal_holdout"]}

    # 3) degraded conditions
    qa = {r["condition"]: r for r in mine["quality"]["model_under_degradation"]}
    qb = {r["condition"]: r for r in rob["quality"]["model_under_degradation"]}
    conds = [c for c in qa if c in qb]
    rows = []
    for c in conds:
        ra, rb = qa[c], qb[c]
        rows.append({"condition": c, "n": ra["n"], "standard": {k: ra[k] for k in ("ungated_sensitivity", "ungated_specificity", "ungated_auc", "rejected_share")},
                     "robust": {k: rb[k] for k in ("ungated_sensitivity", "ungated_specificity", "ungated_auc", "rejected_share")}})
    bal = lambda r: np.nanmean([r["ungated_sensitivity"], r["ungated_specificity"]])
    deg = [r for r in rows if r["condition"].split(" ")[0] in DEGRADED]
    gain = float(np.mean([bal(r["robust"]) - bal(r["standard"]) for r in deg])) if deg else float("nan")
    ok_a = paired["roc_auc"]["lo"] > -0.010 and paired["qwk"]["lo"] > -0.020
    ok_b = gain >= 0.05
    decision = {"rule": "see scripts/compare_robust.py docstring (fixed before the results were seen)", "clean_not_meaningfully_worse": bool(ok_a),
                "mean_balanced_accuracy_gain_on_degraded": gain, "gain_at_least_0.05": bool(ok_b), "recommend_robust_model": bool(ok_a and ok_b)}

    out = {"clean_test": {k: {"threshold": v["screening"]["threshold"], **{m: v["screening"][m] for m in ("sensitivity", "specificity", "roc_auc", "pr_auc")}, "qwk": v["grades"]["qwk"],
                              "macro_f1": v["grades"]["macro_f1"], "ece_5grade": v["calibration"]["ece_top_label_5grade"]} for k, v in clean.items()},
           "paired_robust_minus_standard": paired, "degraded": rows, "decision": decision,
           "robust_run": {"epochs_completed": rob["run"]["epochs_completed"], "best_epoch": rob["run"]["best_epoch"], "stop_reason": rob["run"]["stop_reason"],
                          "train_seconds": rob["run"]["train_seconds"]},
           "n_test": len(ids), "same_split_verified": True,
           "caveat": "degradations are synthetic and applied to full-resolution originals; the training augmentations were applied to 384-px images, so transfer is not guaranteed"}
    mine["robustness_comparison"] = out
    if not a.dry_run:
        M.save_json(mine, ROOT / "results" / "metrics.json")

    names = [r["condition"].replace(" (severity 2)", "").replace(" (same subset)", "") for r in rows]
    if not a.dry_run:
        P.plot_robustness(names, [r["standard"] for r in rows], [r["robust"] for r in rows], ROOT / "report" / "figures" / "robustness_comparison.png")
        dst = ROOT / "artifacts" / "efficientnet_b0_robust"
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("best.pt", "calibration.json"):
            shutil.copy(rd / "artifacts" / "efficientnet_b0_robust" / f, dst / f)

    f3 = lambda v: "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"
    print(f"clean test (n={len(ids)}), robust minus standard:")
    for m, v in paired.items():
        print(f"  {m:12s} {v['point_diff']:+.3f}  95% CI [{v['lo']:+.3f}, {v['hi']:+.3f}]")
    print("clean:", {k: {m: round(x, 3) for m, x in v.items() if isinstance(x, float)} for k, v in out["clean_test"].items()})
    print(f"{'condition':26s} standard sens/spec   robust sens/spec")
    for r in rows:
        print(f"{r['condition'][:26]:26s} {f3(r['standard']['ungated_sensitivity'])}/{f3(r['standard']['ungated_specificity'])}        {f3(r['robust']['ungated_sensitivity'])}/{f3(r['robust']['ungated_specificity'])}")
    print("decision:", json.dumps(decision, indent=1))


if __name__ == "__main__":
    main()
