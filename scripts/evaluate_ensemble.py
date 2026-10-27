"""Select and evaluate ensemble / test-time-augmentation candidates from the logits written by the Kaggle `infer` kernel (CPU, seconds).

    python scripts/evaluate_ensemble.py [--logit-dir DIR]     (default DIR: LOCALAPPDATA/drscreen/kaggle_output_infer/results)

Writes results/metrics.json['ensemble_selection'] and, if the selected candidate is an ensemble or uses TTA and the decision rule adopts it,
artifacts/ensemble/ensemble.json (members, TTA flag, threshold, review margin) that the application can load. See drscreen/ensemble.py for the protocol.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drscreen import metrics as M  # noqa: E402
from drscreen.ensemble import candidate_probs, evaluate_candidates, load_sets  # noqa: E402
from drscreen.fusion import choose_review_margin  # noqa: E402

ROBUST_RUNS = ["efficientnet_b0_robust", "efficientnet_b3_robust"]
DEFAULT = (("efficientnet_b0_robust",), False)  # the model the application uses today


def main():
    default_dir = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".cache"))) / "drscreen" / "kaggle_output_infer" / "results"
    ap = argparse.ArgumentParser()
    ap.add_argument("--logit-dir", default=str(default_dir))
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    d = Path(a.logit_dir)
    sets = load_sets(d)
    cal = json.loads((d / "model_calibrations.json").read_text())
    runs = [r for r in ROBUST_RUNS if f"{r}__plain" in sets["val"]]
    temps = {r: float(cal[r]["temperature"]) for r in runs}
    res = evaluate_candidates(sets, temps, runs, DEFAULT if DEFAULT[0][0] in runs else ((runs[0],), False), n_boot=a.n_boot)
    f3 = lambda v: "  n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"
    print(f"{'candidate':58s} valScore  testSens testSpec testAUC testQWK | idridSens idridAUC")
    for c in sorted(res["candidates"], key=lambda c: -c["val_score"]):
        name = ("+".join(m.replace("efficientnet_", "").replace("_robust", "") for m in c["members"])) + (" +TTA" if c["tta"] else "")
        print(f"{name:58s} {c['val_score']:.4f}   {f3(c['test_sensitivity'])}   {f3(c['test_specificity'])}   {f3(c['test_auc'])}  {f3(c['test_qwk'])}  | {f3(c.get('idrid_sensitivity'))}   {f3(c.get('idrid_auc'))}")
    print("selected:", res["selected"]["members"], "TTA" if res["selected"]["tta"] else "no TTA")
    print("paired (selected - default, test):", {k: (round(v["point_diff"], 4), round(v["lo"], 4), round(v["hi"], 4)) for k, v in res["paired_selected_minus_default_test"].items()})
    print("decision:", json.dumps(res["decision"]))
    mp = ROOT / "results" / "metrics.json"
    met = json.loads(mp.read_text())
    met["ensemble_selection"] = res
    M.save_json(met, mp)
    if res["decision"]["adopt_selected"]:
        sel = res["selected"]
        pv = candidate_probs(sets["val"], tuple(sel["members"]), sel["tta"], temps)
        margin = choose_review_margin(sets["val"]["grade"], M.referable_prob(pv), sel["threshold"], 0.10)
        dst = ROOT / "artifacts" / "ensemble"
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "ensemble.json").write_text(json.dumps({"members": [f"artifacts/{m}/best.pt" for m in sel["members"]], "tta": sel["tta"], "threshold": sel["threshold"],
                                                       "review_margin": margin, "temperatures": {m: temps[m] for m in sel["members"]},
                                                       "selected_on": "validation mean(QWK, ROC-AUC)"}, indent=2))
        print("wrote artifacts/ensemble/ensemble.json")


if __name__ == "__main__":
    main()
