"""Clinically interpretable evaluation: one module used by notebook, app and report.

Positive class for screening = REFERABLE DR (ICDR grade >= 2). A false negative is a referable case the
system missed; a false positive is a non-referable case that was flagged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import average_precision_score, roc_auc_score

GRADE_NAMES = ["0 No DR", "1 Mild NPDR", "2 Moderate NPDR", "3 Severe NPDR", "4 Proliferative DR"]
REFERABLE_MIN_GRADE = 2
NAN = float("nan")

# plain-language definitions used in the metrics table (report + app share these)
METRIC_INFO = {
    "sensitivity": ("Sensitivity (recall)", "Of truly referable cases, the share the system flagged.",
                    "Missed referable cases (false negatives) are the costly error in screening."),
    "specificity": ("Specificity", "Of truly non-referable cases, the share correctly passed as non-referable.",
                    "Low specificity floods reviewers with false alarms."),
    "precision": ("Precision (PPV)", "Of all cases flagged referable, the share that really were.",
                  "Depends on prevalence; tells how trustworthy a flag is."),
    "npv": ("Negative predictive value", "Of cases passed as non-referable, the share that really were.",
            "How reassuring a 'no referral' result is."),
    "roc_auc": ("ROC-AUC", "Probability a random referable case scores higher than a random non-referable case.",
                "Threshold-free separation ability."),
    "pr_auc": ("PR-AUC (average precision)", "Area under the precision-recall curve.",
               "More informative than ROC-AUC when referable cases are the minority."),
    "balanced_accuracy": ("Balanced accuracy", "Mean of sensitivity and specificity.", "Not distorted by class imbalance."),
    "f1": ("F1 score", "Harmonic mean of precision and sensitivity.", "Single number balancing flags and misses."),
    "qwk": ("Quadratic weighted kappa", "Agreement between predicted and true grade beyond chance; "
            "disagreements are penalised by the squared grade distance.",
            "Predicting grade 0 for a grade 4 eye is punished far more than grade 2 vs 3."),
    "macro_f1": ("Macro F1 (5 grades)", "Unweighted mean of per-grade F1.", "Rewards doing well on rare severe grades."),
    "accuracy_5": ("Accuracy (5 grades, secondary)", "Share of exact grade matches.", "Misleading alone under class imbalance."),
    "balanced_accuracy_5": ("Balanced accuracy (5 grades)", "Mean of per-grade recall.", "Each grade counts equally."),
    "ece": ("Expected calibration error", "Average gap between confidence and observed accuracy over equal-width bins.",
            "Shows whether probabilities can be read as frequencies."),
    "brier": ("Brier score", "Mean squared error of the referable probability (lower is better).", "Proper scoring rule: rewards honest probabilities."),
}


def isnan(x) -> bool:
    """NaN-or-None test (values read back from JSON use None for NaN)."""
    return x is None or (isinstance(x, float) and np.isnan(x))


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else NAN


# ----------------------------------------------------------------------------- labels / probs
def referable_labels(grades) -> np.ndarray:
    return (np.asarray(grades) >= REFERABLE_MIN_GRADE).astype(int)


def referable_prob(probs5: np.ndarray) -> np.ndarray:
    return np.asarray(probs5)[:, REFERABLE_MIN_GRADE:].sum(axis=1)


# ----------------------------------------------------------------------------- binary screening metrics
def confusion_counts(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    return {
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
    }


def _auc(y, s) -> float:
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else NAN


def _ap(y, s) -> float:
    return float(average_precision_score(y, s)) if y.sum() > 0 else NAN


def binary_metrics(y_true, score, threshold: float = 0.5) -> dict:
    """Referable-vs-non-referable metrics at a fixed threshold on P(referable). Undefined values are NaN."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    pred = (s >= threshold).astype(int)
    c = confusion_counts(y, pred)
    sens, spec = safe_div(c["tp"], c["tp"] + c["fn"]), safe_div(c["tn"], c["tn"] + c["fp"])
    prec, npv = safe_div(c["tp"], c["tp"] + c["fp"]), safe_div(c["tn"], c["tn"] + c["fn"])
    f1 = safe_div(2 * prec * sens, prec + sens) if not (np.isnan(prec) or np.isnan(sens)) else NAN
    return {
        "threshold": float(threshold), "n": int(len(y)), "n_positive": int(y.sum()), "n_negative": int((1 - y).sum()),
        "prevalence": float(y.mean()) if len(y) else NAN, **c,
        "sensitivity": sens, "specificity": spec, "precision": prec, "npv": npv,
        "balanced_accuracy": float(np.nanmean([sens, spec])) if not (np.isnan(sens) and np.isnan(spec)) else NAN,
        "f1": f1, "roc_auc": _auc(y, s), "pr_auc": _ap(y, s),
    }


# ----------------------------------------------------------------------------- five-grade metrics
def quadratic_weighted_kappa(y_true, y_pred, n_classes: int = 5) -> float:
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    if len(y_true) == 0:
        return NAN
    O = np.bincount(y_true * n_classes + y_pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes).astype(float)
    W = (np.subtract.outer(np.arange(n_classes), np.arange(n_classes)) ** 2) / (n_classes - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    den = (W * E).sum()
    return float(1 - (W * O).sum() / den) if den > 0 else NAN


def grade_metrics(y_true, y_pred, n_classes: int = 5) -> dict:
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    cm = np.zeros((n_classes, n_classes), int)
    for a, b in zip(y_true, y_pred):
        cm[a, b] += 1
    per = {}
    recalls, f1s = [], []
    for k in range(n_classes):
        tp, fp, fn = cm[k, k], cm[:, k].sum() - cm[k, k], cm[k, :].sum() - cm[k, k]
        p, r = safe_div(tp, tp + fp), safe_div(tp, tp + fn)
        f = safe_div(2 * p * r, p + r) if not (np.isnan(p) or np.isnan(r)) else (0.0 if cm[k].sum() else NAN)
        per[str(k)] = {"name": GRADE_NAMES[k], "precision": p, "recall": r, "f1": f, "support": int(cm[k].sum())}
        if cm[k].sum():
            recalls.append(r)
            f1s.append(0.0 if np.isnan(f) else f)
    return {
        "per_grade": per, "confusion_matrix": cm.tolist(),
        "macro_f1": float(np.mean(f1s)) if f1s else NAN,
        "balanced_accuracy_5": float(np.mean(recalls)) if recalls else NAN,
        "accuracy_5": float((y_true == y_pred).mean()) if len(y_true) else NAN,
        "qwk": quadratic_weighted_kappa(y_true, y_pred, n_classes),
    }


# ----------------------------------------------------------------------------- calibration
def expected_calibration_error(confidence, correct, n_bins: int = 15) -> float:
    """ECE with n_bins equal-width bins on [0,1]: sum_b (n_b/N) * |acc_b - conf_b|. Right-closed bins, conf==0 goes to bin 0."""
    conf, corr = np.asarray(confidence, float), np.asarray(correct, float)
    if len(conf) == 0:
        return NAN
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return float(ece)


def reliability_bins(prob_pos, y, n_bins: int = 10) -> dict:
    p, y = np.asarray(prob_pos, float), np.asarray(y, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    out = {"mean_pred": [], "frac_pos": [], "count": []}
    for b in range(n_bins):
        m = idx == b
        out["count"].append(int(m.sum()))
        out["mean_pred"].append(float(p[m].mean()) if m.any() else NAN)
        out["frac_pos"].append(float(y[m].mean()) if m.any() else NAN)
    return out


def brier_binary(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p - y) ** 2)) if len(y) else NAN


def nll(logits: np.ndarray, y: np.ndarray, T: float = 1.0) -> float:
    z = np.asarray(logits, float) / T
    z = z - z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-logp[np.arange(len(y)), np.asarray(y).astype(int)].mean())


def fit_temperature(logits: np.ndarray, y_grade: np.ndarray) -> float:
    """Single scalar T minimising NLL of softmax(logits / T). Fit on VALIDATION data only."""
    res = minimize_scalar(lambda lt: nll(logits, y_grade, float(np.exp(lt))), bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(res.x))


def calibration_metrics(probs5: np.ndarray, y_grade: np.ndarray, n_bins: int = 15) -> dict:
    probs5, y_grade = np.asarray(probs5), np.asarray(y_grade).astype(int)
    pred = probs5.argmax(1)
    pref, yref = referable_prob(probs5), referable_labels(y_grade)
    return {
        "ece_top_label_5grade": expected_calibration_error(probs5.max(1), pred == y_grade, n_bins),
        "ece_referable": expected_calibration_error(np.maximum(pref, 1 - pref), (pref >= 0.5).astype(int) == yref, n_bins),
        "brier_referable": brier_binary(yref, pref),
        "nll_5grade": float(-np.log(np.clip(probs5[np.arange(len(y_grade)), y_grade], 1e-12, 1)).mean()),
        "ece_bins": n_bins, "ece_binning": "equal-width bins on [0,1], weighted by bin size",
    }


# ----------------------------------------------------------------------------- operating threshold (validation only)
def select_threshold(y_true, score, target_sensitivity: float = 0.90) -> dict:
    """Choose the operating threshold on VALIDATION data: the highest threshold whose sensitivity >= target
    (i.e. max specificity subject to the sensitivity target). Falls back to Youden's J if unattainable."""
    y, s = np.asarray(y_true).astype(int), np.asarray(score, float)
    cands = np.unique(np.concatenate([s, [0.0, 1.0]]))
    best = None
    for t in cands:
        m = binary_metrics(y, s, t)
        if not np.isnan(m["sensitivity"]) and m["sensitivity"] >= target_sensitivity:
            if best is None or t > best[0]:
                best = (t, m)
    rule = f"max specificity s.t. validation sensitivity >= {target_sensitivity:.2f}"
    if best is None:
        scores = [(m["sensitivity"] + m["specificity"] - 1, t) for t in cands for m in [binary_metrics(y, s, t)]
                  if not (np.isnan(m["sensitivity"]) or np.isnan(m["specificity"]))]
        t = max(scores)[1] if scores else 0.5
        rule = "Youden's J (target sensitivity unattainable on validation data)"
        best = (t, binary_metrics(y, s, t))
    return {"threshold": float(best[0]), "rule": rule, "validation_sensitivity": best[1]["sensitivity"],
            "validation_specificity": best[1]["specificity"], "fitted_on": "validation"}


# ----------------------------------------------------------------------------- bootstrap CIs
CI_KEYS = ["sensitivity", "specificity", "precision", "npv", "roc_auc", "pr_auc", "balanced_accuracy", "f1",
           "qwk", "macro_f1", "accuracy_5", "balanced_accuracy_5", "brier"]


def _headline(y_grade, probs5, threshold):
    yb, pr = referable_labels(y_grade), referable_prob(probs5)
    m = binary_metrics(yb, pr, threshold)
    g = grade_metrics(y_grade, probs5.argmax(1))
    return {**{k: m[k] for k in ("sensitivity", "specificity", "precision", "npv", "roc_auc", "pr_auc", "balanced_accuracy", "f1")},
            **{k: g[k] for k in ("qwk", "macro_f1", "accuracy_5", "balanced_accuracy_5")}, "brier": brier_binary(yb, pr)}


def bootstrap_ci(y_grade, probs5, threshold: float, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile bootstrap: resample evaluation observations with replacement (same resample for every metric).
    Replicates where a metric is undefined (e.g. no positives drawn) are skipped for that metric; the number of valid
    replicates is reported. Images are treated as independent: no patient IDs exist, so within-patient correlation
    (left/right eyes) is not accounted for, which can make intervals too narrow."""
    y_grade, probs5 = np.asarray(y_grade).astype(int), np.asarray(probs5)
    n = len(y_grade)
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in CI_KEYS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        h = _headline(y_grade[idx], probs5[idx], threshold)
        for k in CI_KEYS:
            if not np.isnan(h[k]):
                draws[k].append(h[k])
    out = {}
    for k, v in draws.items():
        out[k] = ({"lo": float(np.percentile(v, 100 * alpha / 2)), "hi": float(np.percentile(v, 100 * (1 - alpha / 2))), "n_valid": len(v)}
                  if len(v) >= max(20, n_boot // 10) else {"lo": NAN, "hi": NAN, "n_valid": len(v)})
    out["_method"] = {"type": "percentile bootstrap", "n_boot": n_boot, "seed": seed, "alpha": alpha, "n_observations": n,
                      "note": "images treated as independent; no patient grouping available"}
    return out


def paired_bootstrap_diff(y_grade, probs_a, probs_b, metric: str = "roc_auc", n_boot: int = 1000, seed: int = 0, threshold_a: float = 0.5,
                          threshold_b: float = 0.5) -> dict:
    """Paired bootstrap of metric(A) - metric(B) on the SAME evaluation images (resampled together).
    The 95% interval excluding 0 is evidence that the difference is not resampling noise; it does not account for
    selection of A/B on other data or for within-patient correlation."""
    y_grade = np.asarray(y_grade).astype(int)
    pa, pb = np.asarray(probs_a), np.asarray(probs_b)
    rng = np.random.default_rng(seed)
    n = len(y_grade)
    ha, hb = _headline(y_grade, pa, threshold_a), _headline(y_grade, pb, threshold_b)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a, b = _headline(y_grade[idx], pa[idx], threshold_a)[metric], _headline(y_grade[idx], pb[idx], threshold_b)[metric]
        if not (np.isnan(a) or np.isnan(b)):
            diffs.append(a - b)
    ok = len(diffs) >= max(20, n_boot // 10)
    return {"metric": metric, "point_diff": float(ha[metric] - hb[metric]),
            "lo": float(np.percentile(diffs, 2.5)) if ok else NAN, "hi": float(np.percentile(diffs, 97.5)) if ok else NAN,
            "n_valid": len(diffs), "n_boot": n_boot, "seed": seed}


# ----------------------------------------------------------------------------- one-call evaluation
def evaluate_predictions(y_grade, probs5, threshold: float, dataset_name: str, eval_type: str = "internal holdout",
                         n_boot: int = 1000, seed: int = 0, probs5_uncal=None) -> dict:
    """Full report dict (JSON-serialisable). probs5 = the probabilities actually used for decisions."""
    y_grade, probs5 = np.asarray(y_grade).astype(int), np.asarray(probs5)
    yb, pr = referable_labels(y_grade), referable_prob(probs5)
    rep = {
        "dataset": dataset_name, "evaluation_type": eval_type, "n_samples": int(len(y_grade)),
        "positive_class": "referable DR (ICDR grade >= 2)",
        "screening": binary_metrics(yb, pr, threshold),
        "grades": grade_metrics(y_grade, probs5.argmax(1)),
        "calibration": calibration_metrics(probs5, y_grade),
    }
    rep["confidence_intervals"] = bootstrap_ci(y_grade, probs5, threshold, n_boot, seed)
    if probs5_uncal is not None:
        rep["calibration_uncalibrated"] = calibration_metrics(probs5_uncal, y_grade)
    return rep


def metrics_table_rows(rep: dict) -> list[list[str]]:
    """Rows of [Metric, Measured result, 95% CI, What it means, Why it matters] for report and app."""
    s, g, ci, cal = rep["screening"], rep["grades"], rep["confidence_intervals"], rep["calibration"]
    vals = {"sensitivity": s["sensitivity"], "specificity": s["specificity"], "precision": s["precision"], "npv": s["npv"],
            "roc_auc": s["roc_auc"], "pr_auc": s["pr_auc"], "balanced_accuracy": s["balanced_accuracy"], "f1": s["f1"],
            "qwk": g["qwk"], "macro_f1": g["macro_f1"], "balanced_accuracy_5": g["balanced_accuracy_5"], "accuracy_5": g["accuracy_5"],
            "ece": cal["ece_top_label_5grade"], "brier": cal["brier_referable"]}
    rows = []
    for k, v in vals.items():
        name, means, why = METRIC_INFO[k]
        c = ci.get(k)
        cis = f"{c['lo']:.3f} to {c['hi']:.3f}" if c and not isnan(c["lo"]) else "n/a"
        rows.append([name, "n/a" if isnan(v) else f"{v:.3f}", cis, means, why])
    return rows


def save_json(obj, path) -> None:
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        return o

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(clean(obj), indent=2))
