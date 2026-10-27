"""Reproducible report pipeline:  template.md + results/*.json + figures  ->  Final_Report.md  ->  Final_Report.pdf

Every number in the report enters through a placeholder resolved from a saved result file:
    {{val:internal_holdout.screening.sensitivity|.3f}}      value from results/metrics.json
    {{table:main_metrics}} / {{table:grade_metrics}} / {{table:ablation}} / ...   generated tables
    {{figure:name.png|Caption}}                              figure (only if the PNG exists, otherwise an explicit "not available" note)
Unresolved placeholders are rendered as "N/A (not computed)" and listed in report/missing_values.txt so nothing is invented.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

from . import INTENDED_USE
from .metrics import GRADE_NAMES, isnan, metrics_table_rows

NA = "N/A (not computed)"


# ----------------------------------------------------------------------------- placeholders
def _get(d, path: str):
    cur = d
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def fmt_value(v, fmt: str = ".3f", na: str = NA) -> str:
    """Format a value; `na` is shown for NaN/None (use "n/a" where a metric is undefined by construction, NA where a result was not computed)."""
    if isnan(v):
        return na
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return format(v, fmt)
    return str(v)


def md_table(header: list[str], rows: list[list]) -> str:
    esc = lambda c: str(c).replace("|", "\\|").replace("\n", " ")
    out = ["| " + " | ".join(esc(h) for h in header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _ci(rep: dict, k: str, fmt=".3f") -> str:
    c = rep.get("confidence_intervals", {}).get(k)
    return NA if not c or isnan(c.get("lo")) else f"{c['lo']:{fmt}} to {c['hi']:{fmt}}"


def build_tables(res: dict, sim: dict | None) -> dict[str, str]:
    t: dict[str, str] = {}
    h = res.get("internal_holdout")
    if h:
        t["main_metrics"] = md_table(["Metric", "Measured result", "95% CI", "What it means", "Why it matters"], metrics_table_rows(h))
        s = h["screening"]
        t["counts"] = md_table(
            ["", "Predicted non-referable", "Predicted referable"],
            [["**Actual non-referable**", f"TN = {s['tn']}", f"FP = {s['fp']}  (false alarm)"],
             ["**Actual referable**", f"FN = {s['fn']}  (missed referable case)", f"TP = {s['tp']}"]])
        t["setting"] = md_table(["Quantity", "Value"], [
            ["Evaluation set", f"{h['dataset']} ({h['evaluation_type']})"], ["Images evaluated", s["n"]],
            ["Referable (positive) / non-referable (negative)", f"{s['n_positive']} / {s['n_negative']}"],
            ["Prevalence of referable DR in this set", f"{s['prevalence']:.3f}"], ["Operating threshold on P(referable)", f"{s['threshold']:.3f}"]])
        g = h["grades"]
        t["grade_metrics"] = md_table(["Grade", "Precision", "Recall", "F1", "Support"], [
            [GRADE_NAMES[int(k)], fmt_value(v["precision"]), fmt_value(v["recall"]), fmt_value(v["f1"]), v["support"]] for k, v in g["per_grade"].items()])
        cal, cu = h["calibration"], h.get("calibration_uncalibrated")
        if cu:
            t["calibration"] = md_table(["Quantity", "Before temperature scaling", "After temperature scaling"], [
                ["ECE (5-grade top-label, 15 equal-width bins)", fmt_value(cu["ece_top_label_5grade"]), fmt_value(cal["ece_top_label_5grade"])],
                ["ECE (referable, 15 bins)", fmt_value(cu["ece_referable"]), fmt_value(cal["ece_referable"])],
                ["Brier score (referable probability)", fmt_value(cu["brier_referable"]), fmt_value(cal["brier_referable"])],
                ["NLL (5-grade)", fmt_value(cu["nll_5grade"]), fmt_value(cal["nll_5grade"])]])
    v = res.get("validation_threshold_selection")
    if v:
        t["threshold_selection"] = md_table(["Quantity", "Value"], [["Rule", v["rule"]], ["Chosen threshold", fmt_value(v["threshold"])],
                                            ["Validation sensitivity at threshold", fmt_value(v["validation_sensitivity"])],
                                            ["Validation specificity at threshold", fmt_value(v["validation_specificity"])], ["Fitted on", v["fitted_on"]]])
    if res.get("ablation"):
        t["ablation"] = md_table(
            ["Configuration", "N", "Sensitivity", "Specificity", "ROC-AUC", "QWK (5-grade)", "Auto-decided", "Notes"],
            [[r["name"], r["n"], fmt_value(r.get("sensitivity")), fmt_value(r.get("specificity")), fmt_value(r.get("roc_auc"), na="n/a"),
              fmt_value(r.get("qwk"), na="n/a"), r.get("coverage", "all"), r.get("notes", "")] for r in res["ablation"]])
    rc = res.get("robustness_comparison")
    if rc:
        ct = rc["clean_test"]
        t["robust_clean"] = md_table(["Model (clean test split, own validation-selected threshold)", "Threshold", "Sensitivity", "Specificity", "ROC-AUC", "PR-AUC", "QWK", "Macro-F1", "ECE (5-grade)"],
                                     [[k, fmt_value(v["threshold"]), fmt_value(v["sensitivity"]), fmt_value(v["specificity"]), fmt_value(v["roc_auc"]), fmt_value(v["pr_auc"]),
                                       fmt_value(v["qwk"]), fmt_value(v["macro_f1"]), fmt_value(v["ece_5grade"])] for k, v in ct.items()])
        t["robust_paired"] = md_table(["Robust minus standard (same %d test images)" % rc["n_test"], "Difference", "95% CI", "Interval excludes 0"],
                                      [[m, fmt_value(v["point_diff"], "+.3f"), f"{v['lo']:+.3f} to {v['hi']:+.3f}", "yes" if (v["lo"] > 0 or v["hi"] < 0) else "no"]
                                       for m, v in rc["paired_robust_minus_standard"].items()])
        t["robust_degraded"] = md_table(["Condition (grade-stratified subset, no gate)", "N", "Standard sens.", "Standard spec.", "Standard AUC", "Robust sens.", "Robust spec.", "Robust AUC"],
                                        [[r["condition"], r["n"], fmt_value(r["standard"]["ungated_sensitivity"], ".2f"), fmt_value(r["standard"]["ungated_specificity"], ".2f"),
                                          fmt_value(r["standard"]["ungated_auc"], ".2f"), fmt_value(r["robust"]["ungated_sensitivity"], ".2f"),
                                          fmt_value(r["robust"]["ungated_specificity"], ".2f"), fmt_value(r["robust"]["ungated_auc"], ".2f")] for r in rc["degraded"]])
        d = rc["decision"]
        t["robust_decision"] = md_table(["Pre-set criterion", "Outcome"], [
            ["(a) clean test not meaningfully worse (AUC difference CI lower bound > -0.010 and QWK > -0.020)", "met" if d["clean_not_meaningfully_worse"] else "not met"],
            ["(b) mean balanced-accuracy gain on degraded conditions >= 0.05", f"{d['mean_balanced_accuracy_gain_on_degraded']:+.3f}: " + ("met" if d["gain_at_least_0.05"] else "not met")],
            ["Recommendation of the rule", "adopt the robust model for the demo" if d["recommend_robust_model"] else "keep the standard model"]])
    ts = res.get("test_summary")
    if ts:
        what = {"test_preprocess.py": "cropping, masking, resizing, Ben Graham normalisation, invalid-file handling",
                "test_datasets.py": "label-table discovery (submission files are never ground truth), splits, duplicate groups, no leakage between splits, caching",
                "test_metrics.py": "every metric against scikit-learn / hand calculations on controlled examples (QWK, AUPR ties, ECE, Brier, bootstrap and paired bootstrap, temperature scaling, threshold selection)",
                "test_models_train.py": "model shapes, the training loop, checkpoint save/reload equality, resume, time budget",
                "test_quality_gradcam_fusion.py": "quality gate on synthetic degradations, Grad-CAM and its weight-randomisation check, decision and fusion logic, review margin",
                "test_lesion_rules.py": "lesion networks, disc/fovea location, quadrant features, ICDR rule engine and its fitted thresholds, lesion evaluation",
                "test_ensemble.py": "ensemble and test-time-augmentation candidates, validation-only selection and the adoption rule",
                "test_eyepacs.py": "EyePACS labels, patient-level split, split-archive reader, cross-dataset near-duplicate guard",
                "test_explain_casereport.py": "verdict wording, confidence heuristic, narrative, every HTML fragment of the interface (escaping, viewer, threshold chart), PDF case report, end-to-end application flow with real models",
                "test_app_report_sim.py": "application prediction function, report generation, the workflow simulation, Little's law check",
                "test_kaggle_build.py": "that every Kaggle kernel script builds, embeds the source and has a valid configuration header"}
        t["tests"] = md_table(["Test file", "Tests", "Passed", "What it checks"],
                              [[r["file"], r["tests"], r["passed"], what.get(r["file"], "")] for r in ts["per_file"]]
                              + [["**Total**", "**%d**" % ts["total"], "**%d**" % ts["passed"], "%d failed, %d skipped; %.0f s on %s" % (ts["failed"], ts["skipped"], ts["seconds"], ts["platform"])]])
    es = res.get("ensemble_selection")
    if es:
        nice = lambda m: {"efficientnet_b0_robust": "B0 (robust, 384 px)", "efficientnet_b3_robust": "B3 (robust, 448 px)"}.get(m, m)
        cands = sorted(es["candidates"], key=lambda c: -c["val_score"])
        sel = es["selected"]
        label = lambda c: " + ".join(nice(m) for m in c["members"]) + (" + flip TTA" if c["tta"] else "")
        is_sel = lambda c: c["members"] == sel["members"] and c["tta"] == sel["tta"]
        t["ensemble_candidates"] = md_table(
            ["Candidate (all use APTOS-validation temperature and threshold)", "Validation score", "Test sens.", "Test spec.", "Test AUC", "Test QWK", "IDRiD sens.", "IDRiD AUC"],
            [[("**" + label(c) + " (selected)**") if is_sel(c) else label(c), fmt_value(c["val_score"], ".4f"), fmt_value(c["test_sensitivity"]), fmt_value(c["test_specificity"]),
              fmt_value(c["test_auc"]), fmt_value(c["test_qwk"]), fmt_value(c.get("idrid_sensitivity")), fmt_value(c.get("idrid_auc"))] for c in cands])
        d = es["decision"]
        pd_ = es["paired_selected_minus_default_test"]
        t["ensemble_decision"] = md_table(["Pre-set criterion", "Outcome"], [
            ["Validation score gain over the current default (B0 robust, no TTA) at least 0.003", f"{d['validation_score_gain']:+.4f}: " + ("met" if d["gain_at_least_0.003"] else "not met")],
            ["Not worse on the test split (paired intervals)", "ROC-AUC %+.4f (%+.4f to %+.4f); QWK %+.4f (%+.4f to %+.4f): %s" % (
                pd_["roc_auc"]["point_diff"], pd_["roc_auc"]["lo"], pd_["roc_auc"]["hi"], pd_["qwk"]["point_diff"], pd_["qwk"]["lo"], pd_["qwk"]["hi"],
                "met" if d["non_inferior_on_test"] else "not met")],
            ["Decision", "adopt the selected candidate as the application default" if d["adopt_selected"] else "keep the current default"]])
    ee = res.get("eyepacs_experiment")
    if ee:
        mname = {"efficientnet_b0_robust": "Robust B0 (APTOS only; current model)", "efficientnet_b0_eyepacs_mixed": "Phase A: APTOS + EyePACS",
                 "efficientnet_b0_eyepacs": "Phase B: phase A fine-tuned on APTOS (candidate)"}
        rows = []
        for k, v in ee["models"].items():
            ts, ir, ep = v["test"], v["idrid_test"], v.get("eyepacs_test")
            rows.append([mname.get(k, k), fmt_value(v["val_qwk"]), fmt_value(ts["sensitivity"]), fmt_value(ts["specificity"]), fmt_value(ts["roc_auc"]), fmt_value(ts["qwk"]),
                         fmt_value(ir["sensitivity"]), fmt_value(ir["specificity"]), fmt_value(ir["roc_auc"]), fmt_value(ir["qwk"]),
                         fmt_value(ep["sensitivity"]) if ep else "n/a", fmt_value(ep["roc_auc"]) if ep else "n/a"])
        t["eyepacs_models"] = md_table(["Model (T and threshold fitted on APTOS validation only)", "Val. QWK", "APTOS test sens.", "spec.", "AUC", "QWK", "IDRiD sens.", "spec.", "AUC", "QWK",
                                        "EyePACS test sens.", "AUC"], rows)
        pr = []
        for split, label in (("test", "APTOS test (n=%d)" % ee["models"]["efficientnet_b0_eyepacs"]["test"]["n"]), ("idrid_test", "IDRiD grading test (n=%d)" % ee["models"]["efficientnet_b0_eyepacs"]["idrid_test"]["n"])):
            for m, v in ee["paired_candidate_minus_robust"][split].items():
                pr.append([label, m, fmt_value(v["point_diff"], "+.3f"), f"{v['lo']:+.3f} to {v['hi']:+.3f}", "yes" if (v["lo"] > 0 or v["hi"] < 0) else "no"])
        t["eyepacs_paired"] = md_table(["Evaluation set", "Metric", "Candidate minus robust B0", "95% CI", "Interval excludes 0"], pr)
        dd, dt = ee["decision"], ee["data"]
        t["eyepacs_decision"] = md_table(["Pre-set criterion", "Outcome"], [
            ["Candidate chosen on APTOS validation QWK", dd["candidate"]],
            ["(a) non-inferior on APTOS test: AUC CI lower bound > -0.010 and QWK CI lower bound > -0.020", "met" if dd["non_inferior_on_aptos_test"] else "not met"],
            ["(b) IDRiD: ROC-AUC gain >= 0.02 and sensitivity not lower", "met" if dd["idrid_benefit"] else "not met"],
            ["Decision", "adopt as the application default" if dd["adopt_as_default"] else "not adopted as default; installed as an optional model"]])
        g = lambda d: " / ".join(str(d[str(i)]) for i in range(5))
        t["eyepacs_data"] = md_table(["Quantity", "Value"], [
            ["EyePACS training images used (grades 0 / 1 / 2 / 3 / 4)", g(dt["eyepacs_train_grades"])],
            ["EyePACS training patients", dt["eyepacs_train_patients"]],
            ["EyePACS held-out test sample (grades 0 / 1 / 2 / 3 / 4; natural mix; patients never seen in training)", g(dt["eyepacs_test_grades"])],
            ["EyePACS test patients", dt["eyepacs_test_patients"]],
            ["Near-duplicates (dHash <= 6 bits or identical MD5) of APTOS validation/test or IDRiD test images found among %d EyePACS images" % dt["dedup"]["n_eyepacs_checked"], dt["dedup"]["n_removed_near_duplicates"]],
            ["APTOS train / validation / test images (same split as every other model)", "%d / %d / %d" % (dt["aptos"]["train"]["n"], dt["aptos"]["val"]["n"], dt["aptos"]["test"]["n"])]])
    lv = res.get("lesion_v2_comparison")
    if lv:
        a1, a2 = lv["idrid_test_mean_aupr"]["per_lesion_v1"], lv["idrid_test_mean_aupr"]["per_lesion_v2"]
        nm = {"MA": "Microaneurysms", "HE": "Haemorrhages", "EX": "Hard exudates", "SE": "Soft exudates", "OD": "Optic disc"}
        ev_ = lv["eophtha_test_v2"]["per_lesion"]
        t["lesion_v2_aupr"] = md_table(["IDRiD test structure", "AUPR v1 (deployed)", "AUPR v2", "e-ophtha test AUPR (v2 only)"],
                                       [[nm[k], fmt_value(a1[k]), fmt_value(a2[k]), fmt_value(ev_[k]["aupr"]) if k in ev_ else "n/a"] for k in nm]
                                       + [["**Mean of MA, HE, EX, SE**", "**%s**" % fmt_value(lv["idrid_test_mean_aupr"]["v1"]), "**%s**" % fmt_value(lv["idrid_test_mean_aupr"]["v2"]), ""]])
        v1, v2 = lv["aptos_v1"], lv["aptos_v2"]
        t["lesion_v2_aptos"] = md_table(["On APTOS (lesion detections transferred without APTOS lesion labels)", "v1 (deployed)", "v2"], [
            ["Rule-engine QWK, validation rows (rule counts fitted on training rows)", fmt_value(v1["rules_val_qwk"]), fmt_value(v2["rules_val_qwk"])],
            ["Rule-engine QWK, test rows", fmt_value(v1["rules_test_qwk"]), fmt_value(v2["rules_test_qwk"])],
            ["Gradient-boosting grader on lesion features, ROC-AUC (val / test)", "%s / %s" % (fmt_value(v1["gb_val_auc"]), fmt_value(v1["gb_test_auc"])), "%s / %s" % (fmt_value(v2["gb_val_auc"]), fmt_value(v2["gb_test_auc"]))],
            ["Grade-0 images with >= 20 detected haemorrhage regions (false-positive proxy)", fmt_value(v1["grade0_share_with_HE_ge_20"], ".1%"), fmt_value(v2["grade0_share_with_HE_ge_20"], ".1%")],
            ["Grade-0 images: mean detected hard-exudate regions", fmt_value(v1["grade0_mean_EX_count"], ".1f"), fmt_value(v2["grade0_mean_EX_count"], ".1f")]])
        d = lv["decision"]
        t["lesion_v2_decision"] = md_table(["Pre-set criterion", "Outcome"], [
            ["(a) mean IDRiD-test AUPR not more than 0.03 below v1", f"{lv['idrid_test_mean_aupr']['v2'] - lv['idrid_test_mean_aupr']['v1']:+.3f}: " + ("met" if d["idrid_not_worse_than_0.03"] else "not met")],
            ["(b) rule-engine QWK on APTOS validation improves by >= 0.03", f"{d['aptos_val_rule_qwk_gain']:+.3f}: " + ("met" if d["gain_at_least_0.03"] else "not met")],
            ["Decision", "adopt v2" if d["adopt_v2"] else "keep v1 (v2 not adopted)"]])
    le = res.get("lesion_evidence") or {}
    seg = (le.get("segmentation_idrid_test") or {}).get("per_lesion")
    if seg:
        names = {"MA": "Microaneurysms", "HE": "Haemorrhages", "EX": "Hard exudates", "SE": "Soft exudates", "OD": "Optic disc"}
        t["lesion_seg"] = md_table(["Structure", "AUPR", "Dice (pixel F1)", "IoU", "Precision", "Recall", "Threshold (val.)", "Test images containing it"],
                                   [[names[k], fmt_value(v["aupr"]), fmt_value(v["dice"]), fmt_value(v["iou"]), fmt_value(v["precision"]), fmt_value(v["recall"]),
                                     fmt_value(v["threshold"], ".2f"), v["n_test_images_with_lesion"]] for k, v in seg.items()])
    loc, fin = le.get("locator_idrid_test"), le.get("locator_final_idrid_test")
    if loc:
        cols = [("Regressor alone", loc)] + ([("Deployed method (segmented disc; regressor or prior fovea)", fin)] if fin else [])
        t["locator"] = md_table(["Quantity (IDRiD test, original pixels)"] + [c[0] for c in cols], [
            ["Images"] + [c[1]["n_test"] for c in cols],
            ["Optic-disc centre error, mean / median (px)"] + [f"{c[1]['od_error_px_mean']:.1f} / {c[1]['od_error_px_median']:.1f}" for c in cols],
            ["Fovea centre error, mean / median (px)"] + [f"{c[1]['fovea_error_px_mean']:.1f} / {c[1]['fovea_error_px_median']:.1f}" for c in cols],
            ["Approx. optic-disc diameter (px)"] + [f"{c[1]['disc_diameter_px_approx']:.0f}" for c in cols],
            ["Fovea within one disc diameter of ground truth"] + [f"{c[1]['fovea_within_1DD']:.1%}" for c in cols],
            ["Optic disc within one disc diameter of ground truth"] + [f"{c[1]['od_within_1DD']:.1%}" for c in cols]])
    lf = res.get("lesion_fusion")
    if lf:
        t["lesion_fusion"] = md_table(["Configuration", "N", "Sensitivity", "Specificity", "ROC-AUC", "QWK (5-grade)", "FN (missed)"],
                                      [[r["name"], r["n"], fmt_value(r.get("sensitivity")), fmt_value(r.get("specificity")), fmt_value(r.get("roc_auc"), na="n/a"),
                                        fmt_value(r.get("qwk"), na="n/a"), r.get("fn", "")] for r in lf["rows"]])
        t["rule_params"] = md_table(["Rule variant", "min MA", "min HE", "min EX", "min SE", "HE per quadrant ('4' of 4-2-1)"],
                                    [[v["name"], v["min_ma"], v["min_he"], v["min_ex"], v["min_se"], v["he_per_quadrant"]] for v in lf["rule_params"].values()])
        rows = []
        for k, v in lf["cross_check"].items():
            sc = v["status_counts"]
            rows.append([k, sc["agree"], sc["lesions_only"], sc["classifier_only"], fmt_value(v["classifier_error_rate_when_agree"]),
                         fmt_value(v["classifier_error_rate_when_disagree"]), v["classifier_alone"]["fn"], v["with_escalation"]["fn"], v["with_escalation"]["n_extra_referrals"],
                         v["with_escalation"]["referable_cases_rescued"]])
        t["lesion_cross"] = md_table(["Rule variant", "Agree", "Lesions only", "Classifier only", "Classifier error when agree", "when disagree", "FN classifier alone",
                                      "FN with escalation", "Extra referrals", "Referable cases rescued"], rows)
        pr = []
        for key, label in (("paired_fusion_minus_classifier", "Fusion minus classifier"), ("paired_lesion_gb_minus_classifier", "Lesion-feature grader minus classifier")):
            for m, ml in (("roc_auc", "ROC-AUC"), ("qwk", "QWK")):
                v = lf[key][m]
                pr.append([label, ml, fmt_value(v["point_diff"], "+.3f"), "n/a" if isnan(v["lo"]) else f"{v['lo']:+.3f} to {v['hi']:+.3f}",
                           "yes" if (not isnan(v["lo"])) and (v["lo"] > 0 or v["hi"] < 0) else "no"])
        t["lesion_paired"] = md_table(["Comparison", "Metric", "Difference", "95% CI", "Interval excludes 0"], pr)
    ex = res.get("external_idrid")
    if ex and ex.get("rows"):
        t["idrid_external"] = md_table(["Configuration", "N", "Sensitivity", "Specificity", "ROC-AUC", "QWK (5-grade)"],
                                       [[r["name"], r["n"], fmt_value(r.get("sensitivity")), fmt_value(r.get("specificity")), fmt_value(r.get("roc_auc")), fmt_value(r.get("qwk"))]
                                        for r in ex["rows"]])
        t["idrid_external_info"] = md_table(["Quantity", "Value"], [["Dataset", ex["dataset"]], ["Evaluation type", ex["evaluation_type"]], ["Test images in split", ex["n_test_total"]],
                                            ["Excluded (duplicates of lesion-model training images)", ex["n_excluded_duplicates_of_lesion_training"]], ["Evaluated", ex["n_evaluated"]],
                                            ["Referable prevalence", fmt_value(ex.get("referable_prevalence"))]] +
                                           ([["Macular-involvement proxy ROC-AUC (EX within 1 disc diameter of fovea vs IDRiD oedema risk)", fmt_value(ex["dme_proxy"]["roc_auc"])]] if ex.get("dme_proxy") else []))
    if res.get("paired_comparison_primary_minus_baseline"):
        rows = []
        for name, d in res["paired_comparison_primary_minus_baseline"].items():
            for m, label in (("roc_auc", "ROC-AUC"), ("qwk", "QWK")):
                v = d[m]
                excl = (not isnan(v["lo"])) and (v["lo"] > 0 or v["hi"] < 0)
                rows.append([f"EfficientNet-B0 minus {name}", label, fmt_value(v["point_diff"], "+.3f"),
                             "n/a" if isnan(v["lo"]) else f"{v['lo']:+.3f} to {v['hi']:+.3f}", "yes" if excl else "no"])
        nb = next(iter(res["paired_comparison_primary_minus_baseline"].values()))["roc_auc"]["n_boot"]
        t["paired"] = (f"**Paired bootstrap of metric differences (same test images, {nb} resamples).**\n\n"
                       + md_table(["Comparison", "Metric", "Difference", "95% CI", "Interval excludes 0"], rows))
    q = res.get("quality")
    if q and q.get("degradation"):
        t["quality_degradation"] = md_table(
            ["Degradation", "Severity", "N", "Gate: reject", "Gate: enhance", "Gate: good"],
            [[r["kind"], r["severity"], r["n"], f"{r['reject']:.1%}", f"{r['enhance']:.1%}", f"{r['good']:.1%}"] for r in q["degradation"]])
    if q and q.get("model_under_degradation"):
        t["quality_perf"] = md_table(
            ["Condition (grade-stratified subset)", "N", "Rejected by gate", "Ungated sens.", "Ungated spec.", "Ungated AUC", "Analysed-only sens.", "Analysed-only spec."],
            [[r["condition"], r["n"], f"{r['rejected_share']:.0%}", fmt_value(r["ungated_sensitivity"], ".2f"), fmt_value(r["ungated_specificity"], ".2f"),
              fmt_value(r["ungated_auc"], ".2f"), fmt_value(r["gated_analysed_sensitivity"], ".2f", "n/a (all rejected)"), fmt_value(r["gated_analysed_specificity"], ".2f", "n/a (all rejected)")]
             for r in q["model_under_degradation"]])
    if q and q.get("clean"):
        c = q["clean"]
        t["quality_clean"] = md_table(["Status on unmodified test images", "Count", "Share"], [[k, c[k], f"{c[k] / max(1, c['n']):.1%}"] for k in ("good", "enhance", "reject")])
    if res.get("latency"):
        L = res["latency"]
        t["latency"] = md_table(["Setting", "Value"], [[k, v] for k, v in L.items()])
    if res.get("environment"):
        t["environment"] = md_table(["Item", "Value"], [[k, v] for k, v in res["environment"].items()])
    if res.get("dataset", {}).get("splits"):
        sp = res["dataset"]["splits"]
        t["splits"] = md_table(["Split", "Images", "Gr.0", "Gr.1", "Gr.2", "Gr.3", "Gr.4", "Referable prevalence"],
                               [[k, v["n"], *[v["grade_counts"][str(i)] for i in range(5)], f"{v['referable_prevalence']:.3f}"] for k, v in sp.items()])
    if res.get("baselines"):
        t["baselines"] = md_table(["Model", "Sensitivity", "Specificity", "ROC-AUC", "QWK", "Operating threshold"],
                                  [[k, fmt_value(v.get("sensitivity")), fmt_value(v.get("specificity")), fmt_value(v.get("roc_auc")), fmt_value(v.get("qwk")),
                                    fmt_value(v.get("threshold"))] for k, v in res["baselines"].items()])
    if res.get("external"):
        e = res["external"]
        if "screening" in e:
            t["external"] = md_table(["Quantity", "Value"], [["Dataset", e["dataset"]], ["N", e["n_samples"]], ["Sensitivity", fmt_value(e["screening"]["sensitivity"])],
                                                           ["Specificity", fmt_value(e["screening"]["specificity"])], ["ROC-AUC", fmt_value(e["screening"]["roc_auc"])]])
    if sim and sim.get("sweep"):
        rows = sim["sweep"]["rows"]
        t["sim_sweep"] = md_table(
            ["Cameras", "Reviewers", "Camera wait p95 (min)", "Review wait p95 (min)", "Camera util.", "Reviewer util.", "Completed/day", "Meets targets"],
            [[r["cameras"], r["reviewers"], f"{r['camera_wait_p95_min']:.1f}", f"{r['review_wait_p95_min']:.1f}", f"{r['util_cameras']:.2f}",
              f"{r['util_reviewers']:.2f}", f"{r['completed_per_day']:.0f}", "yes" if r["meets_targets"] else "no"] for r in rows])
        c = sim["sweep"]["base_config"]
        t["sim_assumptions"] = md_table(["Assumption (not observed data)", "Value"], [[k, c[k]] for k in (
            "annual_patients", "working_days", "hours_per_day", "imaging_min", "upload_min", "inference_min", "review_min", "p_quality_reject",
            "prevalence", "sensitivity", "specificity", "p_uncertain")])
        b = sim["baseline_run"]["littles_law"]
        t["sim_littles"] = md_table(["Little's law check (baseline run)", "Value"], [["λ (patients per working minute)", f"{b['lambda_per_min']:.4f}"],
                                    ["W (mean time in system, min)", f"{b['W_min']:.1f}"], ["λ·W", f"{b['lambda_times_W']:.2f}"], ["L observed (sampled mean in system)", f"{b['L_observed']:.2f}"]])
    return t


_PH = re.compile(r"\{\{(val|table|figure|file):([^}|]+)(?:\|([^}]*))?\}\}")


def render_markdown(template: str, res: dict, sim: dict | None, fig_dir: Path, results_dir: Path | None = None) -> tuple[str, list[str]]:
    tables = build_tables(res, sim)
    missing: list[str] = []
    ctx = {**res, "simulation": sim or {}}

    def sub(m):
        kind, key, arg = m.group(1), m.group(2).strip(), m.group(3)
        if kind == "file":  # raw text of a saved results file (e.g. the worked example report)
            f = (results_dir or Path(".")) / key
            if not f.is_file():
                missing.append(f"file:{key}")
                return f"*{NA}: {key} was not generated.*"
            return f.read_text(encoding="utf-8")
        if kind == "val":
            v = _get(ctx, key)
            if isnan(v):
                missing.append(f"val:{key}")
                return NA
            return fmt_value(v, arg or ".3f")
        if kind == "table":
            if key not in tables:
                missing.append(f"table:{key}")
                return f"*{NA}: table '{key}' requires an experiment that has not been run.*"
            return tables[key]
        if not (fig_dir / key).is_file():
            missing.append(f"figure:{key}")
            return f"*Figure not available ({key}): the corresponding experiment was not run or produced no output.*"
        return f"![{(arg or '').strip()}](figures/{key})"

    return _PH.sub(sub, template), missing


# ----------------------------------------------------------------------------- markdown -> PDF
_FONTS_DONE = False


def _register_fonts():
    global _FONTS_DONE
    if _FONTS_DONE:
        return
    d = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for name, f in (("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSans-Oblique.ttf"),
                    ("DV-BI", "DejaVuSans-BoldOblique.ttf"), ("DVM", "DejaVuSansMono.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(d / f)))
    pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")
    _FONTS_DONE = True


def _styles():
    _register_fonts()
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontName="DV", fontSize=9.2, leading=13.2, alignment=TA_JUSTIFY, spaceAfter=5)
    return {
        "body": body,
        "h1": ParagraphStyle("h1", parent=body, fontName="DV-B", fontSize=15, leading=19, spaceBefore=14, spaceAfter=7, textColor=colors.HexColor("#1f3a5f"), alignment=0),
        "h2": ParagraphStyle("h2", parent=body, fontName="DV-B", fontSize=11.5, leading=15, spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#1f3a5f"), alignment=0),
        "h3": ParagraphStyle("h3", parent=body, fontName="DV-B", fontSize=10, leading=13, spaceBefore=6, spaceAfter=3, alignment=0),
        "bullet": ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=4, spaceAfter=2),
        "cell": ParagraphStyle("cell", parent=body, fontSize=7.4, leading=9.4, alignment=0, spaceAfter=0),
        "cellh": ParagraphStyle("cellh", parent=body, fontName="DV-B", fontSize=7.4, leading=9.4, alignment=0, textColor=colors.white, spaceAfter=0),
        "caption": ParagraphStyle("caption", parent=body, fontName="DV-I", fontSize=8, leading=10.5, alignment=TA_CENTER, textColor=colors.HexColor("#444444"), spaceAfter=9),
        "quote": ParagraphStyle("quote", parent=body, leftIndent=12, rightIndent=12, backColor=colors.HexColor("#eef3f8"), borderPadding=6, spaceBefore=4, spaceAfter=8),
        "code": ParagraphStyle("code", fontName="DVM", fontSize=7.5, leading=9.5),
        "title": ParagraphStyle("title", parent=body, fontName="DV-B", fontSize=24, leading=30, alignment=TA_CENTER, textColor=colors.HexColor("#1f3a5f"), spaceAfter=14),
        "subtitle": ParagraphStyle("subtitle", parent=body, fontSize=11.5, leading=16, alignment=TA_CENTER, spaceAfter=6),
    }


def _inline(s: str) -> str:
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"`([^`]+)`", r'<font name="DVM">\1</font>', s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![\w*])\*([^*\s][^*]*)\*(?![\w*])", r"<i>\1</i>", s)
    s = re.sub(r"(?<![\w])_([^_\s][^_]*)_(?![\w])", r"<i>\1</i>", s)
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<link href="\2" color="#1a5fb4">\1</link>', s)
    return s


class _Doc(SimpleDocTemplate):
    def afterFlowable(self, f):
        if isinstance(f, Paragraph) and f.style.name in ("h1", "h2"):
            key = f"h{self.seq.nextf('toc')}"
            self.canv.bookmarkPage(key)
            self.notify("TOCEntry", (0 if f.style.name == "h1" else 1, f.getPlainText(), self.page, key))


def _table(rows: list[list[str]], st, width: float):
    ncol = len(rows[0])
    data = [[Paragraph(_inline(c), st["cellh"] if i == 0 else st["cell"]) for c in r] for i, r in enumerate(rows)]
    lens = [max(len(re.sub(r"\W", "", r[j])) for r in rows) for j in range(ncol)]
    weights = [min(max(l, 15), 40) for l in lens]  # numbers always fit; long text columns shrink instead
    cw = [width * w / sum(weights) for w in weights]
    t = Table(data, colWidths=cw, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5f9")]),
                           ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b8c2cf")), ("LEFTPADDING", (0, 0), (-1, -1), 3),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 3), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return t


def markdown_to_story(md: str, base_dir: Path, st, width: float) -> list:
    story, lines, i = [], md.splitlines(), 0
    para: list[str] = []

    def flush():
        if para:
            story.append(Paragraph(_inline(" ".join(para)), st["body"]))
            para.clear()

    while i < len(lines):
        ln = lines[i].rstrip()
        if ln.startswith("```"):
            flush()
            buf, i = [], i + 1
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(lines[i])
                i += 1
            story += [Preformatted("\n".join(buf), st["code"]), Spacer(1, 6)]
        elif ln.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            flush()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                if not re.match(r"^\|[\s:|-]+\|$", lines[i].strip()):
                    rows.append([c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", lines[i].strip().strip("|"))])
                i += 1
            n = len(rows[0])
            rows = [(r + [""] * n)[:n] for r in rows]
            story += [_table(rows, st, width), Spacer(1, 8)]
            continue
        elif m := re.match(r"^!\[([^\]]*)\]\(([^)]+)\)", ln):
            flush()
            p = base_dir / m.group(2)
            if p.is_file():
                from PIL import Image as PI

                src = PI.open(p)
                w0, h0 = src.size
                w = min(width, width * 0.92)
                h = w * h0 / w0
                if h > 11.5 * cm:
                    w, h = w * 11.5 * cm / h, 11.5 * cm
                # keep the PDF small: downsample large figures (>1500 px wide) and store big photographic ones as JPEG
                import io

                buf = io.BytesIO()
                im = src.convert("RGB")
                if w0 > 1500:
                    im = im.resize((1500, int(h0 * 1500 / w0)), PI.LANCZOS)
                if p.stat().st_size > 400_000:
                    im.save(buf, "JPEG", quality=88, optimize=True)
                else:
                    im.save(buf, "PNG", optimize=True)
                buf.seek(0)
                story.append(KeepTogether([Image(buf, width=w, height=h), Paragraph(_inline(m.group(1)), st["caption"])]))
        elif ln.startswith("### "):
            flush(); story.append(Paragraph(_inline(ln[4:]), st["h3"]))
        elif ln.startswith("## "):
            flush(); story.append(Paragraph(_inline(ln[3:]), st["h2"]))
        elif ln.startswith("# "):
            flush(); story.append(Paragraph(_inline(ln[2:]), st["h1"]))
        elif re.match(r"^\s*([-*]|\d+\.)\s+", ln):
            flush()
            m = re.match(r"^\s*([-*]|\d+\.)\s+(.*)", ln)
            story.append(Paragraph(_inline(m.group(2)), st["bullet"], bulletText="•" if m.group(1) in "-*" else m.group(1)))
        elif ln.startswith(">"):
            flush(); story.append(Paragraph(_inline(ln.lstrip("> ")), st["quote"]))
        elif not ln.strip():
            flush()
        elif ln.strip() == "<!--pagebreak-->":
            flush(); story.append(PageBreak())
        else:
            para.append(ln.strip())
        i += 1
    flush()
    return story


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DV", 7)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(2 * cm, 1.1 * cm, "DR screening research prototype - not a medical diagnosis")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.restoreState()


def markdown_to_pdf(md_path: str | Path, pdf_path: str | Path, meta: dict) -> dict:
    """Render markdown (with images relative to md_path's folder) to a PDF with title page and contents. Returns stats."""
    md_path, pdf_path = Path(md_path), Path(pdf_path)
    st = _styles()
    width = A4[0] - 4 * cm
    doc = _Doc(str(pdf_path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=1.8 * cm,
               title=meta.get("title", "Report"), author=meta.get("author", ""))
    story = [Spacer(1, 4.5 * cm), Paragraph(_inline(meta["title"]), st["title"])]
    for k in ("subtitle", "author", "institution", "date"):
        if meta.get(k):
            story.append(Paragraph(_inline(meta[k]), st["subtitle"]))
    story += [Spacer(1, 1.2 * cm), Paragraph(_inline("**Intended use.** " + INTENDED_USE), st["quote"]), PageBreak()]
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle("t0", fontName="DV-B", fontSize=9, leading=13, leftIndent=0),
                       ParagraphStyle("t1", fontName="DV", fontSize=8.2, leading=11, leftIndent=14)]
    story += [Paragraph("Contents", st["h1"]), toc, PageBreak()]
    story += markdown_to_story(md_path.read_text(encoding="utf-8"), md_path.parent, st, width)
    doc.multiBuild(story, onFirstPage=_footer, onLaterPages=_footer)
    data = pdf_path.read_bytes()
    return {"pages": len(re.findall(rb"/Type\s*/Page[^s]", data)), "images": len(re.findall(rb"/Subtype\s*/Image", data)), "bytes": len(data)}


def build_report(template: str | Path, results_dir: str | Path, out_md: str | Path, out_pdf: str | Path, meta: dict) -> dict:
    results_dir, out_md = Path(results_dir), Path(out_md)
    res = json.loads((results_dir / "metrics.json").read_text()) if (results_dir / "metrics.json").is_file() else {}
    sim = json.loads((results_dir / "simulation.json").read_text()) if (results_dir / "simulation.json").is_file() else None
    if (results_dir / "test_summary.json").is_file():  # written by scripts/test_summary.py: the real outcome of the last pytest run
        res["test_summary"] = json.loads((results_dir / "test_summary.json").read_text())
    md, missing = render_markdown(Path(template).read_text(encoding="utf-8"), res, sim, out_md.parent / "figures", results_dir)
    out_md.write_text(md, encoding="utf-8")
    (out_md.parent / "missing_values.txt").write_text("\n".join(sorted(set(missing))) + ("\n" if missing else ""))
    stats = markdown_to_pdf(out_md, out_pdf, meta)
    stats["missing_placeholders"] = len(set(missing))
    return stats
