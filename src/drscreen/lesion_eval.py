"""Evaluate the lesion-rule path and its fusion with the classifier, on the SAME held-out APTOS test images (CPU only).

Inputs (written by the remote runs): lesion features for APTOS / IDRiD, the classifier's predictions.csv, and its validation-selected
threshold. Protocol: everything tunable (rule thresholds, gradient-boosting grader, fusion weight, thresholds) is fitted on TRAIN or
VALIDATION rows only; the test split is touched once for reporting.

Caveat reported with the results: the lesion model was trained on IDRiD and is applied to APTOS without lesion ground truth, so
APTOS lesion detections are unvalidated; their usefulness is only measured indirectly, through agreement with the APTOS grade.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as M
from .fusion import cross_check
from .lesion import FEATURE_NAMES
from .rules import RuleParams, apply_rules, fit_rule_params, grade_vector

N_GRADES = 5


def _onehot_grade_probs(grades: np.ndarray) -> np.ndarray:
    p = np.zeros((len(grades), N_GRADES))
    p[np.arange(len(grades)), np.clip(grades, 0, 4)] = 1.0
    return p


def _row(name: str, y: np.ndarray, probs5: np.ndarray, thr: float, extra: dict | None = None) -> dict:
    pr = M.referable_prob(probs5)
    b = M.binary_metrics(M.referable_labels(y), pr, thr)
    return {"name": name, "n": int(len(y)), "sensitivity": b["sensitivity"], "specificity": b["specificity"], "roc_auc": b["roc_auc"],
            "qwk": M.grade_metrics(y, probs5.argmax(1))["qwk"], "tp": b["tp"], "fp": b["fp"], "tn": b["tn"], "fn": b["fn"], **(extra or {})}


def _cols(df: pd.DataFrame) -> list[str]:
    return [c for c in FEATURE_NAMES if c in df.columns]


def fit_gb(train: pd.DataFrame, seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier

    gb = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_depth=4, class_weight="balanced", random_state=seed)
    gb.fit(train[_cols(train)].fillna(0), train["grade"].astype(int))
    gb._cols = _cols(train)  # remember which feature columns were used (older feature files lack the *_large columns)
    return gb


def gb_probs(gb, df: pd.DataFrame) -> np.ndarray:
    out = np.zeros((len(df), N_GRADES))
    out[:, gb.classes_] = gb.predict_proba(df[gb._cols].fillna(0))
    return out


def evaluate_lesion_fusion(feats: pd.DataFrame, preds: pd.DataFrame, cnn_threshold: float, target_sensitivity: float = 0.90, n_boot: int = 1000, seed: int = 0) -> dict:
    """feats: lesion_features_aptos.csv (image_id, split, grade, FEATURE_NAMES..., loc_ok). preds: predictions.csv (val/test rows, calibrated p_grade0-4)."""
    f = feats[(feats["loc_ok"] == 1) & feats["split"].isin(["train", "val", "test"])].copy()
    pc = [f"p_grade{k}" for k in range(5)]
    p = preds[["image_id", "split", *pc]].rename(columns={"split": "split_pred"})
    va, te = f[f["split"] == "val"].merge(p, on="image_id"), f[f["split"] == "test"].merge(p, on="image_id")
    tr = f[f["split"] == "train"]
    if min(len(tr), len(va), len(te)) < 20:
        raise ValueError(f"not enough rows: train {len(tr)}, val {len(va)}, test {len(te)}")
    yv, yt = va["grade"].to_numpy(int), te["grade"].to_numpy(int)
    cv, ct = va[pc].to_numpy(), te[pc].to_numpy()
    out: dict = {"n_train_features": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)), "n_lesion_failed": int((feats["loc_ok"] != 1).sum()),
                 "protocol": "rule thresholds, GB grader on TRAIN rows; fusion weight + thresholds on VALIDATION rows; test touched once"}
    rows = [_row("EfficientNet-B0 classifier alone", yt, ct, cnn_threshold)]
    # (1) rule engine: literal ICDR and data-calibrated thresholds
    literal = RuleParams()
    calib = fit_rule_params(tr, tr["grade"].to_numpy(int))
    out["rule_params"] = {"literal": literal.to_dict(), "calibrated": calib.to_dict()}
    rule_grades = {}
    for key, prm in (("literal", literal), ("calibrated", calib)):
        g = grade_vector(te, prm)
        rule_grades[key] = g
        b = M.binary_metrics(M.referable_labels(yt), (g >= 2).astype(float), 0.5)
        rows.append({"name": f"Lesion rules alone ({prm.name})", "n": int(len(yt)), "sensitivity": b["sensitivity"], "specificity": b["specificity"],
                     "roc_auc": M.binary_metrics(M.referable_labels(yt), g.astype(float), 0.5)["roc_auc"], "qwk": M.quadratic_weighted_kappa(yt, g),
                     "tp": b["tp"], "fp": b["fp"], "tn": b["tn"], "fn": b["fn"]})
        out[f"rule_{key}_confusion_5grade"] = M.grade_metrics(yt, g)["confusion_matrix"]
    # (2) gradient-boosting grader on the same lesion features
    gb = fit_gb(tr, seed)
    gv, gt = gb_probs(gb, va), gb_probs(gb, te)
    thr_gb = M.select_threshold(M.referable_labels(yv), M.referable_prob(gv), target_sensitivity)["threshold"]
    rows.append(_row("Lesion features + gradient boosting", yt, gt, thr_gb))
    # (3) probability fusion: weight on VALIDATION
    best_w, best_auc = 1.0, -1
    for w in np.linspace(0, 1, 11):
        a = M.binary_metrics(M.referable_labels(yv), M.referable_prob(w * cv + (1 - w) * gv), 0.5)["roc_auc"]
        if not np.isnan(a) and a > best_auc + 1e-9:
            best_w, best_auc = float(w), float(a)
    fv, ft = best_w * cv + (1 - best_w) * gv, best_w * ct + (1 - best_w) * gt
    thr_f = M.select_threshold(M.referable_labels(yv), M.referable_prob(fv), target_sensitivity)["threshold"]
    rows.append(_row(f"Fusion: classifier (w={best_w:.1f}) + lesion-feature grader", yt, ft, thr_f))
    out["fusion"] = {"weight_classifier": best_w, "val_auc_at_weight": best_auc, "threshold": thr_f, "note": "w=1.0 means the validation set preferred the classifier alone"}
    # (4) paired bootstrap: fused vs classifier alone (same test images)
    out["paired_fusion_minus_classifier"] = {m: M.paired_bootstrap_diff(yt, ft, ct, m, n_boot, seed, thr_f, cnn_threshold) for m in ("roc_auc", "qwk")}
    out["paired_lesion_gb_minus_classifier"] = {m: M.paired_bootstrap_diff(yt, gt, ct, m, n_boot, seed, thr_gb, cnn_threshold) for m in ("roc_auc", "qwk")}
    # (5) cross-check policies on the classifier: do lesion rules flag cases the classifier misses, and at what cost?
    pref_t = M.referable_prob(ct)
    ybt = M.referable_labels(yt)
    cross = {}
    for key, prm in (("literal", literal), ("calibrated", calib)):
        status, review = [], []
        for i in range(len(te)):
            rr = {"referable": bool(rule_grades[key][i] >= 2)}
            cc = cross_check(float(pref_t[i]), cnn_threshold, rr)
            status.append(cc["status"])
            review.append(cc["review"])
        status, review = np.array(status), np.array(review)
        cnn_pos = pref_t >= cnn_threshold
        escalated = cnn_pos | review  # conservative: human-review cases are counted as referred
        b = M.binary_metrics(ybt, escalated.astype(float), 0.5)
        counts = {s: int((status == s).sum()) for s in ("agree", "lesions_only", "classifier_only")}
        err_agree = float((cnn_pos[status == "agree"] != ybt[status == "agree"]).mean()) if (status == "agree").any() else float("nan")
        err_dis = float((cnn_pos[status != "agree"] != ybt[status != "agree"]).mean()) if (status != "agree").any() else float("nan")
        cross[key] = {"status_counts": counts, "classifier_error_rate_when_agree": err_agree, "classifier_error_rate_when_disagree": err_dis,
                      "classifier_alone": {"sensitivity": M.binary_metrics(ybt, cnn_pos.astype(float), 0.5)["sensitivity"], "specificity": M.binary_metrics(ybt, cnn_pos.astype(float), 0.5)["specificity"],
                                           "fn": int(((~cnn_pos) & (ybt == 1)).sum())},
                      "with_escalation": {"sensitivity": b["sensitivity"], "specificity": b["specificity"], "fn": b["fn"], "n_extra_referrals": int((escalated & ~cnn_pos).sum()),
                                          "referable_cases_rescued": int((escalated & ~cnn_pos & (ybt == 1)).sum())}}
        rows.append({"name": f"Classifier + lesion-evidence escalation ({prm.name})", "n": int(len(yt)), "sensitivity": b["sensitivity"], "specificity": b["specificity"],
                     "roc_auc": float("nan"), "qwk": float("nan"), "tp": b["tp"], "fp": b["fp"], "tn": b["tn"], "fn": b["fn"]})
    out["cross_check"] = cross
    out["rows"] = rows
    # (6) which lesion features separate referable from non-referable on TRAIN (descriptive, not a performance claim)
    out["lesion_count_by_grade_train"] = {str(g): {k: float(tr.loc[tr["grade"] == g, f"{k}_count"].mean()) for k in ("MA", "HE", "EX", "SE")} for g in range(5) if (tr["grade"] == g).any()}
    return out


def evaluate_idrid_external(feats_idrid: pd.DataFrame, cnn_probs: pd.DataFrame | None, feats_aptos: pd.DataFrame, cnn_threshold: float,
                            rule_params: dict, n_boot: int = 1000, seed: int = 0) -> dict:
    """Independent check on IDRiD grading TEST images (excluding near-duplicates of the lesion model's training images).
    Classifier and lesion rules were never fitted on IDRiD grading data (only APTOS train + IDRiD lesion masks)."""
    t = feats_idrid[(feats_idrid["split"] == "test") & (feats_idrid["in_lesion_train_or_val"] == 0) & (feats_idrid["loc_ok"] == 1)].copy()
    out = {"dataset": "IDRiD disease-grading test split", "evaluation_type": "EXTERNAL (never used for training or selection of the classifier or rule thresholds)",
           "n_test_total": int((feats_idrid["split"] == "test").sum()), "n_evaluated": int(len(t)),
           "n_excluded_duplicates_of_lesion_training": int(((feats_idrid["split"] == "test") & (feats_idrid["in_lesion_train_or_val"] == 1)).sum())}
    if len(t) < 20:
        out["error"] = "too few images"
        return out
    y = t["grade"].to_numpy(int)
    yb = M.referable_labels(y)
    out["referable_prevalence"] = float(yb.mean())
    rows = []
    for key in ("literal", "calibrated"):
        g = grade_vector(t, RuleParams(**{k: v for k, v in rule_params[key].items()}))
        b = M.binary_metrics(yb, (g >= 2).astype(float), 0.5)
        rows.append({"name": f"Lesion rules alone ({rule_params[key]['name']})", "n": int(len(y)), "sensitivity": b["sensitivity"], "specificity": b["specificity"],
                     "roc_auc": M.binary_metrics(yb, g.astype(float), 0.5)["roc_auc"], "qwk": M.quadratic_weighted_kappa(y, g)})
    gb = fit_gb(feats_aptos[(feats_aptos["split"] == "train") & (feats_aptos["loc_ok"] == 1)])
    gp = gb_probs(gb, t)
    rows.append({"name": "Lesion features + gradient boosting (APTOS-trained)", "n": int(len(y)), **{k: v for k, v in _row("", y, gp, 0.5).items() if k in ("sensitivity", "specificity", "roc_auc", "qwk")}})
    if cnn_probs is not None:
        cp_ = cnn_probs[cnn_probs["split"] == "test"] if "split" in cnn_probs.columns else cnn_probs  # image names repeat across IDRiD train/test
        c = t[["image_id"]].merge(cp_.drop(columns=["split"], errors="ignore"), on="image_id", how="left")
        cp = c[[f"p_grade{k}" for k in range(5)]].to_numpy()
        ok = ~np.isnan(cp).any(1)
        if ok.sum() >= 20:
            r = _row("EfficientNet-B0 classifier alone", y[ok], cp[ok], cnn_threshold)
            rows.insert(0, r)
            out["classifier_ci"] = M.bootstrap_ci(y[ok], cp[ok], cnn_threshold, n_boot, seed)
            out["classifier_5grade_confusion"] = M.grade_metrics(y[ok], cp[ok].argmax(1))["confusion_matrix"]
    out["rows"] = rows
    if (t["dme"] >= 0).all():
        d = (t["dme"].to_numpy(int) >= 1).astype(int)
        s = t["EX_within_1DD"].to_numpy(float)
        out["dme_proxy"] = {"definition": "hard-exudate regions within 1 disc diameter of the estimated fovea vs IDRiD macular-oedema risk >= 1",
                            "roc_auc": M.binary_metrics(d, s, 0.5)["roc_auc"], "n": int(len(d)), "n_dme_risk": int(d.sum())}
    return out
