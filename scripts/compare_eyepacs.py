"""Evaluate the EyePACS-augmented classifier against the current default (robustness-augmented EfficientNet-B0) from saved logits (CPU, seconds).

    python scripts/compare_eyepacs.py [--eyepacs-dir DIR] [--infer-dir DIR]

--eyepacs-dir : output of the `dr-eyepacs-experiment` kernel (results/logits_*.npz, artifacts/<run>/best.pt)
--infer-dir   : output of the `dr-classifier-logits` kernel (results/logits_*.npz holding the current robust model's logits on the same images)
Both default to LOCALAPPDATA/drscreen/kaggle_output_{eyepacs,infer}. Every model goes through the SAME code path: temperature + threshold are fitted on the
APTOS validation split only, then applied unchanged to APTOS test, the EyePACS patient-level test sample and the IDRiD grading test (external).

Decision rule (fixed BEFORE any result was seen):
  1. Candidate = whichever of {phase A mixed model, phase B APTOS fine-tuned model} has the higher APTOS-VALIDATION QWK (tie: the fine-tuned one).
  2. Adopt the candidate as the application's default only if BOTH hold
     (a) in-domain non-inferiority on APTOS test: paired-bootstrap 95% CI lower bound of (candidate - robust) > -0.010 for ROC-AUC and > -0.020 for QWK;
     (b) external benefit on IDRiD (n = 100, small): ROC-AUC improves by >= 0.02 AND sensitivity at the validation-selected threshold does not decrease.
  Otherwise the candidate is installed as an optional model only (artifacts/efficientnet_b0_eyepacs/) and the report says it was not adopted.
Writes results/metrics.json['eyepacs_experiment'].
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drscreen import metrics as M  # noqa: E402
from drscreen.fusion import choose_review_margin  # noqa: E402
from drscreen.models import grade_probs  # noqa: E402

BASE = "efficientnet_b0_robust"
CANDS = ("efficientnet_b0_eyepacs_mixed", "efficientnet_b0_eyepacs")
EP_RUN_DIR = "efficientnet_b0_eyepacs"  # installed folder name


def _load(d: Path, name: str):
    f = d / "results" / f"logits_{name}.npz"
    if not f.is_file():
        return None
    z = np.load(f, allow_pickle=False)
    return {k: z[k] for k in z.files}


def main():
    base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".cache"))) / "drscreen"
    ap = argparse.ArgumentParser()
    ap.add_argument("--eyepacs-dir", default=str(base / "kaggle_output_eyepacs"))
    ap.add_argument("--infer-dir", default=str(base / "kaggle_output_infer"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    ed, idr = Path(a.eyepacs_dir), Path(a.infer_dir)

    sets = {}
    for name in ("val", "test", "idrid_test", "eyepacs_test"):
        e = _load(ed, name)
        if e is None:
            continue
        logits = {r: e[r] for r in (*CANDS, BASE) if r in e}
        if name != "eyepacs_test":  # the current model's logits come from the infer kernel (same images, matched by id)
            i = _load(idr, name)
            if i is None or f"{BASE}__plain" not in i:
                raise SystemExit(f"missing {BASE} logits for '{name}' in {idr}; run the `infer` kernel first")
            order = {s: k for k, s in enumerate(i["image_id"])}
            pos = [order[s] for s in e["image_id"]]  # raises KeyError if the image sets differ
            logits[BASE] = i[f"{BASE}__plain"][pos]
            assert (i["grade"][pos] == e["grade"]).all(), "label mismatch between kernels"
        sets[name] = {"grade": e["grade"].astype(int), "logits": logits}
    if "val" not in sets or "test" not in sets:
        raise SystemExit("need at least the val and test logits")

    yv = sets["val"]["grade"]
    fitted = {}
    for r in (BASE, *CANDS):
        T = M.fit_temperature(sets["val"]["logits"][r], yv)
        pv = grade_probs(sets["val"]["logits"][r], T)
        thr = M.select_threshold(M.referable_labels(yv), M.referable_prob(pv), 0.90)["threshold"]
        fitted[r] = {"T": T, "thr": thr, "val_qwk": M.grade_metrics(yv, pv.argmax(1))["qwk"], "val_probs": pv}
    cand = max(CANDS[::-1], key=lambda r: fitted[r]["val_qwk"])  # tie -> fine-tuned (listed first in the reversed order)
    probs = lambda r, s: grade_probs(sets[s]["logits"][r], fitted[r]["T"])

    table = {}
    for r in (BASE, *CANDS):
        table[r] = {"temperature": fitted[r]["T"], "threshold": fitted[r]["thr"], "val_qwk": fitted[r]["val_qwk"]}
        for s in sets:
            if s == "val" or r not in sets[s]["logits"]:
                continue
            ev = M.evaluate_predictions(sets[s]["grade"], probs(r, s), fitted[r]["thr"], f"{s}", "see compare_eyepacs.py", a.n_boot, 0)
            table[r][s] = {"n": int(len(sets[s]["grade"])), **{k: ev["screening"][k] for k in ("sensitivity", "specificity", "roc_auc", "pr_auc")}, "qwk": ev["grades"]["qwk"],
                           "sens_ci": ev["confidence_intervals"].get("sensitivity"), "auc_ci": ev["confidence_intervals"].get("roc_auc")}
    paired = {}
    for s, metrics in (("test", ("roc_auc", "qwk", "sensitivity", "specificity")), ("idrid_test", ("roc_auc", "sensitivity", "specificity"))):
        if s in sets:
            paired[s] = {m: M.paired_bootstrap_diff(sets[s]["grade"], probs(cand, s), probs(BASE, s), m, a.n_boot, 0, fitted[cand]["thr"], fitted[BASE]["thr"]) for m in metrics}
    ok_a = paired["test"]["roc_auc"]["lo"] > -0.010 and paired["test"]["qwk"]["lo"] > -0.020
    has_idrid = "idrid_test" in sets
    ok_b = bool(has_idrid and table[cand]["idrid_test"]["roc_auc"] - table[BASE]["idrid_test"]["roc_auc"] >= 0.02
                and table[cand]["idrid_test"]["sensitivity"] >= table[BASE]["idrid_test"]["sensitivity"])
    decision = {"candidate": cand, "selected_on": "APTOS validation QWK", "non_inferior_on_aptos_test": bool(ok_a), "idrid_benefit": ok_b,
                "adopt_as_default": bool(ok_a and ok_b)}

    out = {"models": table, "paired_candidate_minus_robust": paired, "decision": decision,
           "data": json.loads((ed / "results" / "eyepacs_data_summary.json").read_text()) if (ed / "results" / "eyepacs_data_summary.json").is_file() else None,
           "caveats": ["EyePACS grades are noisier than APTOS/IDRiD (single-reader labels from varied sites).",
                       "The IDRiD check has n = 100: its intervals are wide, so rule (b) is a weak guard rather than proof.",
                       "The robust baseline never saw EyePACS, so its EyePACS-test numbers are an external-domain result for it only."]}
    f3 = lambda v: "  n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"
    print(f"{'model':32s} valQWK | " + " | ".join(f"{s}: sens spec auc qwk" for s in sets if s != "val"))
    for r, t in table.items():
        print(f"{r:32s} {t['val_qwk']:.3f} | " + " | ".join(f"{f3(t[s]['sensitivity'])} {f3(t[s]['specificity'])} {f3(t[s]['roc_auc'])} {f3(t[s]['qwk'])}" if s in t else "n/a" for s in sets if s != "val"))
    for s, p in paired.items():
        print(f"paired {cand} - {BASE} on {s}:", {m: (round(v["point_diff"], 3), round(v["lo"], 3), round(v["hi"], 3)) for m, v in p.items()})
    print("decision:", json.dumps(decision))
    if a.dry_run:
        return
    mp = ROOT / "results" / "metrics.json"
    met = json.loads(mp.read_text())
    met["eyepacs_experiment"] = out
    M.save_json(met, mp)
    dst = ROOT / "artifacts" / EP_RUN_DIR
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(ed / "artifacts" / cand / "best.pt", dst / "best.pt")
    pv = fitted[cand]["val_probs"]
    rob_cal = json.loads((ROOT / "artifacts" / BASE / "calibration.json").read_text())
    cal = {"temperature": fitted[cand]["T"], "threshold": fitted[cand]["thr"], "review_margin": choose_review_margin(yv, M.referable_prob(pv), fitted[cand]["thr"], 0.10),
           "threshold_rule": "highest threshold with validation sensitivity >= 0.90", "fitted_on": "APTOS validation split",
           "quality_thresholds": rob_cal.get("quality_thresholds"), "trained_on": "APTOS 2019 train + EyePACS (patient-level split) [" + cand + "]",
           "adopted_as_default": decision["adopt_as_default"]}
    (dst / "calibration.json").write_text(json.dumps(cal, indent=2))
    print("installed", dst, "adopted_as_default =", decision["adopt_as_default"])


if __name__ == "__main__":
    main()
