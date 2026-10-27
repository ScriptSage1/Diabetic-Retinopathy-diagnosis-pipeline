"""Decision fusion: combine the quality gate and the classifier's calibrated referable probability into one action.

Scope note: this fuses *quality status* and *classifier confidence*. It does NOT fuse lesion evidence because no
lesion detector exists in this project (see rules.lesion_evidence). Whether the fused pipeline is better than the
classifier alone is a measured question answered in the ablation, not an assumption.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .metrics import binary_metrics, referable_labels, safe_div


@dataclass
class Decision:
    action: str   # refer | no_referral | human_review | recapture
    reason: str

    def to_dict(self):
        return asdict(self)


def decide(quality_status: str, p_ref: float | None, threshold: float, review_margin: float = 0.0) -> Decision:
    """quality reject -> recapture; |p - threshold| <= margin -> human review; else refer / no_referral."""
    if quality_status == "reject":
        return Decision("recapture", "image quality insufficient for automated analysis")
    if p_ref is None:
        return Decision("human_review", "no model output available")
    if abs(p_ref - threshold) <= review_margin:
        return Decision("human_review", f"P(referable) {p_ref:.2f} is within {review_margin:.2f} of the operating threshold {threshold:.2f}")
    if p_ref >= threshold:
        return Decision("refer", f"P(referable) {p_ref:.2f} >= threshold {threshold:.2f}")
    return Decision("no_referral", f"P(referable) {p_ref:.2f} < threshold {threshold:.2f}")


def pipeline_metrics(y_grade, p_ref, quality_status, threshold: float, review_margin: float = 0.0) -> dict:
    """End-to-end accounting on a fixed set of images, stating how rejected / reviewed cases are handled.

    analysed_only: sensitivity/specificity on images the pipeline decided automatically (excludes recapture and
                   human_review); coverage says how many images that is.
    conservative:  recapture and human_review cases are counted as 'referred' (the safe default for screening),
                   so every image gets an outcome and nothing is silently dropped.
    """
    y = referable_labels(y_grade)
    p = np.asarray(p_ref, float)
    acts = [decide(q, pi, threshold, review_margin).action for q, pi in zip(quality_status, p)]
    acts = np.array(acts)
    auto = np.isin(acts, ["refer", "no_referral"])
    out = {"n": int(len(y)), "n_recapture": int((acts == "recapture").sum()), "n_human_review": int((acts == "human_review").sum()),
           "n_auto": int(auto.sum()), "coverage_auto": float(auto.mean()) if len(y) else float("nan")}
    out["analysed_only"] = binary_metrics(y[auto], (acts[auto] == "refer").astype(float), 0.5) if auto.any() else None
    pred_cons = np.isin(acts, ["refer", "recapture", "human_review"]).astype(float)
    out["conservative"] = binary_metrics(y, pred_cons, 0.5)
    out["referral_rate_conservative"] = float(pred_cons.mean()) if len(y) else float("nan")
    out["referable_in_recapture"] = int(y[acts == "recapture"].sum())
    out["referable_in_human_review"] = int(y[acts == "human_review"].sum())
    return out


def choose_review_margin(y_grade_val, p_ref_val, threshold: float, max_review_fraction: float = 0.15) -> float:
    """Pick the review margin on VALIDATION data: widest margin such that at most `max_review_fraction` of images fall in it."""
    p = np.asarray(p_ref_val, float)
    d = np.abs(p - threshold)
    return float(np.quantile(d, max_review_fraction)) if len(d) else 0.0


# ----------------------------------------------------------------------------- classifier x lesion-evidence cross-check
def cross_check(p_ref: float, threshold: float, rule: dict | None, review_on_cnn_only: bool = False) -> dict:
    """Compare the classifier's referable decision with the lesion-rule grade. `rule` is RuleResult.to_dict().

    agree                  both say referable, or both say non-referable
    lesions_only           lesion rules say referable, classifier does not  -> human review (safety first: a possible missed case)
    classifier_only        classifier says referable, detected lesions do not meet the referable rule criteria -> flagged; review only if review_on_cnn_only
    Lesion evidence is limited (see rules.py): 'classifier_only' is expected for subtle disease or lesions the model cannot see.
    """
    if rule is None:
        return {"status": "no_lesion_evidence", "review": False, "message": "Lesion evidence was not computed; the decision rests on the classifier alone."}
    cnn_ref, rule_ref = p_ref >= threshold, bool(rule["referable"])
    if cnn_ref == rule_ref:
        return {"status": "agree", "review": False, "message": "The classifier and the lesion rules reach the same referable / non-referable decision."}
    if rule_ref and not cnn_ref:
        return {"status": "lesions_only", "review": True,
                "message": "Lesion rules indicate referable-level disease that the classifier did not flag: recommend human review."}
    return {"status": "classifier_only", "review": bool(review_on_cnn_only),
            "message": "The classifier flags referable DR, but the lesions found by the (limited) detector do not meet the rules' referable criteria: the flag is not corroborated by lesion evidence."}


def decide_with_evidence(quality_status: str, p_ref: float | None, threshold: float, review_margin: float, cross: dict | None) -> Decision:
    d = decide(quality_status, p_ref, threshold, review_margin)
    if cross and cross.get("review") and d.action == "no_referral":
        return Decision("human_review", "classifier below threshold but lesion rules suggest referable disease; " + cross["message"])
    return d
