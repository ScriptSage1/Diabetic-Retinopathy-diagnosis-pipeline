"""Model selection among robustness-augmented classifiers, ensembles and flip test-time augmentation, from saved logits (CPU, seconds).

Protocol (fixed before any result is seen):
  * candidates: every non-empty subset of the robustness-augmented models, each with and without flip TTA (TTA logits = mean of the plain, h-flip and
    v-flip views); an ensemble averages the members' temperature-scaled probabilities (each with its own validation-fitted temperature);
  * selection uses the VALIDATION split only: score = mean(QWK, referable ROC-AUC); the operating threshold is the usual validation rule
    (max specificity subject to sensitivity >= target);
  * the TEST split is used only for (a) reporting every candidate and (b) a paired non-inferiority check of the selected candidate against the
    current default; the IDRiD images are an independent external check and never used for selection.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from . import metrics as M
from .models import grade_probs


def _logits(npz, run: str, tta: bool) -> np.ndarray:
    plain, flip = npz[f"{run}__plain"], npz[f"{run}__flip"]
    return (plain + 2 * flip) / 3.0 if tta else plain


def candidate_probs(npz, runs: tuple[str, ...], tta: bool, temps: dict) -> np.ndarray:
    """Average of the members' calibrated 5-grade probabilities."""
    return np.mean([grade_probs(_logits(npz, r, tta), temps[r]) for r in runs], axis=0)


def evaluate_candidates(sets: dict, temps: dict, runs: list[str], default: tuple[tuple[str, ...], bool], target_sensitivity: float = 0.90, n_boot: int = 1000, seed: int = 0) -> dict:
    """sets: {'val': npz, 'test': npz, 'idrid_test': npz or None}; each npz holds image_id, grade and '<run>__plain'/'<run>__flip' logits."""
    cands = []
    for k in range(1, len(runs) + 1):
        for sub in itertools.combinations(runs, k):
            for tta in (False, True):
                cands.append((sub, tta))
    rows = {}
    yv = sets["val"]["grade"]
    for sub, tta in cands:
        pv = candidate_probs(sets["val"], sub, tta, temps)
        thr = M.select_threshold(M.referable_labels(yv), M.referable_prob(pv), target_sensitivity)["threshold"]
        auc = M.binary_metrics(M.referable_labels(yv), M.referable_prob(pv), thr)["roc_auc"]
        qwk = M.quadratic_weighted_kappa(yv, pv.argmax(1))
        rows[(sub, tta)] = {"val_qwk": qwk, "val_auc": auc, "val_score": float(np.mean([qwk, auc])), "threshold": thr}
    best = max(rows, key=lambda c: rows[c]["val_score"])
    out = {"protocol": "selected on VALIDATION mean(QWK, ROC-AUC); test only reported + non-inferiority vs default; IDRiD independent", "default": {"members": list(default[0]), "tta": default[1]},
           "selected": {"members": list(best[0]), "tta": best[1], **rows[best]}, "candidates": []}
    yt = sets["test"]["grade"]
    for (sub, tta), r in rows.items():
        pt = candidate_probs(sets["test"], sub, tta, temps)
        b = M.binary_metrics(M.referable_labels(yt), M.referable_prob(pt), r["threshold"])
        row = {"members": list(sub), "tta": tta, **r, "test_sensitivity": b["sensitivity"], "test_specificity": b["specificity"], "test_auc": b["roc_auc"], "test_qwk": M.quadratic_weighted_kappa(yt, pt.argmax(1)),
               "test_ece_5grade": M.calibration_metrics(pt, yt)["ece_top_label_5grade"]}
        if sets.get("idrid_test") is not None:
            yi = sets["idrid_test"]["grade"]
            pi = candidate_probs(sets["idrid_test"], sub, tta, temps)
            bi = M.binary_metrics(M.referable_labels(yi), M.referable_prob(pi), r["threshold"])
            row.update({"idrid_sensitivity": bi["sensitivity"], "idrid_specificity": bi["specificity"], "idrid_auc": bi["roc_auc"], "idrid_qwk": M.quadratic_weighted_kappa(yi, pi.argmax(1))})
        out["candidates"].append(row)
    d = (default[0], default[1])
    ps, pd_ = candidate_probs(sets["test"], best[0], best[1], temps), candidate_probs(sets["test"], d[0], d[1], temps)
    paired = {m: M.paired_bootstrap_diff(yt, ps, pd_, m, n_boot, seed, rows[best]["threshold"], rows[d]["threshold"]) for m in ("roc_auc", "qwk")}
    gain = rows[best]["val_score"] - rows[d]["val_score"]
    ok_noninf = paired["roc_auc"]["lo"] > -0.010 and paired["qwk"]["lo"] > -0.020
    out["paired_selected_minus_default_test"] = paired
    out["decision"] = {"validation_score_gain": float(gain), "gain_at_least_0.003": bool(gain >= 0.003), "non_inferior_on_test": bool(ok_noninf),
                       "adopt_selected": bool(best != d and gain >= 0.003 and ok_noninf), "selected_is_default": bool(best == d)}
    return out


def load_sets(logit_dir: str | Path) -> dict:
    d = Path(logit_dir)
    sets = {k: np.load(d / f"logits_{k}.npz", allow_pickle=False) for k in ("val", "test")}
    sets["idrid_test"] = np.load(d / "logits_idrid_test.npz", allow_pickle=False) if (d / "logits_idrid_test.npz").is_file() else None
    return {k: (dict(v) if v is not None else None) for k, v in sets.items()}
