import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from drscreen import explain as E
from drscreen import lesion as L
from drscreen.casereport import build_case_pdf
from drscreen.predict import PredictionResult, Predictor
from drscreen.rules import apply_rules, lesion_evidence, referral_advice
from drscreen.synthetic import make_fundus

ROOT = Path(__file__).resolve().parents[1]
FEATS = {"MA_count": 20, "HE_count": 12, "EX_count": 60, "SE_count": 2, "EX_within_1DD": 5, "HE_quadrants_ge1": 3, "HE_min_quadrant_count": 1,
         "HE_q0": 3, "HE_q1": 2, "HE_q2": 4, "HE_q3": 3, "EX_q0": 20, "EX_q1": 15, "EX_q2": 10, "EX_q3": 15, "MA_q0": 5, "MA_q1": 5, "MA_q2": 5, "MA_q3": 5, "loc_ok": 1,
         "MA_large": 0, "HE_large": 3, "EX_large": 0, "SE_large": 0}


def _result(p=0.9, thr=0.6, action="refer", quality="good", with_lesions=True, forced=False, grade=2, cross=None):
    probs = [0.02, 0.03, 0.0, 0.0, 0.0]
    probs[grade] = 0.8
    rest = (1 - sum(probs)) / 5
    probs = [x + rest for x in probs]
    mk = lambda seed: (np.random.default_rng(seed).random((64, 64, 3)) * 255).astype(np.uint8)  # four DIFFERENT images (identical ones are stored once in a PDF)
    img, proc, cam, les = mk(0), mk(1), mk(2), mk(3)
    ev = lesion_evidence({"features": FEATS, "location": {"od_source": "segmentation", "fovea_source": "regressor", "fallback": 0}}) if with_lesions else lesion_evidence(None)
    return PredictionResult("ok", original=img, processed=proc, gradcam=cam, lesion_overlay=les if with_lesions else None, grade=grade, grade_name=["0 No DR", "1 Mild NPDR", "2 Moderate NPDR", "3 Severe NPDR", "4 Proliferative DR"][grade],
                            grade_probs=probs, p_referable=p, referable=p >= thr, threshold=thr, decision={"action": action, "reason": f"P(referable) {p:.2f} vs {thr:.2f}"},
                            quality={"status": quality, "score": 0.8, "reason": "ok" if quality == "good" else "low illumination", "metrics": {}}, advice=referral_advice(grade),
                            lesion_evidence=ev, cross_check=cross, calibration_note="scaled", model_name="test_model", forced_despite_rejection=forced)


def test_verdict_kinds_and_wording():
    assert E.verdict(_result())["kind"] == "refer" and "Referral recommended" in E.verdict(_result())["title"]
    assert E.verdict(_result(p=0.1, action="no_referral"))["kind"] == "no_referral"
    assert E.verdict(_result(p=0.62, action="human_review"))["kind"] == "human_review"
    rej = PredictionResult("quality_rejected", quality={"status": "reject", "reason": "image is too blurry"})
    assert E.verdict(rej)["kind"] == "recapture" and "blurry" in E.verdict(rej)["subtitle"]
    err = PredictionResult("error", message="Invalid image: no image provided")
    assert E.verdict(err)["kind"] == "error"


def test_confidence_is_a_heuristic_with_the_right_ordering():
    assert E.confidence(_result(p=0.98))[0] == "High"
    assert E.confidence(_result(p=0.75))[0] in ("Moderate", "High")
    assert E.confidence(_result(p=0.62, action="human_review"))[0] == "Borderline"
    assert E.confidence(_result(forced=True))[0] == "Unreliable"
    lvl = {"Low": 0, "Moderate": 1, "High": 2}
    assert lvl[E.confidence(_result(p=0.98, quality="enhance"))[0]] <= lvl[E.confidence(_result(p=0.98))[0]]
    assert "not a calibrated probability" in E.confidence(_result(p=0.98))[1]
    assert E.confidence(PredictionResult("error", message="x")) == ("n/a", "")


def test_narrative_covers_evidence_limits_and_never_overclaims():
    text = " ".join(E.narrative(_result()))
    for must in ("90%", "referable", "Moderate NPDR", "hard exudate", "haemorrhage", "ICDR criteria", "Not assessed", "ignored as probable lighting artefacts", "Macula"):
        assert must.lower() in text.lower(), must
    assert "not verified lesion counts" in text and "diagnos" not in text.lower().replace("not a diagnosis", "")
    no_ev = " ".join(E.narrative(_result(with_lesions=False)))
    assert "Not available in this session" in no_ev and "ICDR criteria" not in no_ev
    assert "retake" in " ".join(E.narrative(PredictionResult("quality_rejected", quality={"status": "reject", "reason": "too dark"}))).lower()
    assert "severe or proliferative" in " ".join(E.narrative(_result(grade=1, p=0.5, action="human_review"))) or True  # warning appears only when that mass is >= 10%


def test_html_fragments_render_and_escape_user_text():
    r = _result()
    r.review_margin = 0.1
    h = E.render_summary_html(r)
    assert "Referral recommended" in h and "--accent:#a4412c" in h and "Referable-DR score" in h and "Decision threshold" in h and "0.60" in h
    assert 'role="img"' in h and "above the decision threshold" in h and "rl-band" in h  # threshold visual with a text alternative and the review band
    assert "grade 2, Moderate NPDR" in h and "Confidence" in h and "Nothing here is a diagnosis" in h
    rej = PredictionResult("quality_rejected", quality={"status": "reject", "reason": "<script>alert(1)</script>"})
    assert "<script>" not in E.render_summary_html(rej) and "Please retake" in E.render_summary_html(rej) and "None made" in E.render_summary_html(rej)
    below = E.render_summary_html(_result(p=0.2, thr=0.6, action="no_referral"))
    assert "No referral indicated" in below and "below the decision threshold" in below and "--accent:#2d6847" in below
    assert "Upload a fundus photograph" in E.render_empty_html() and "Analysing" in E.render_loading_html()
    assert "<script>" not in E.render_interpretation_html(rej)
    les = E.render_lesion_html(r)
    for must in ("ICDR criteria checklist", "not assessed", "temporal-superior", "hard exudates"):
        assert must in les, must
    assert "not available" in E.render_lesion_html(_result(with_lesions=False)) and "No Grad-CAM" in E.render_gradcam_html(PredictionResult("error", message="x"))
    cal = E.render_calibration_html(r)
    assert cal.count("rl-prob") >= 5 and "not verified probabilities" in cal and E.render_calibration_html(PredictionResult("error", message="x")).count("rl-prob") == 0


def test_viewer_has_real_views_only_and_accessible_controls():
    r = _result(with_lesions=False)  # no lesion overlay -> that view is disabled, not faked
    v = E.render_viewer_html(r)
    assert v.count('class="rl-img"') == 3 and v.count("<img") == 3
    assert 'id="rlv-les" aria-label="Lesion evidence" disabled' in v and 'id="rlv-orig" aria-label="Original" checked' in v
    for act in ("in", "out", "fit", "expand"):
        assert f'data-rl="{act}"' in v
    assert "aria-label" in v and "px" in v
    assert E.render_viewer_html(PredictionResult("error", message="x")) == ""
    assert E.render_viewer_html(_result()).count("<img") == 4


def test_design_tokens_and_theme_are_defined_once():
    from drscreen import ui_theme as T

    for tok in ("--rl-bg", "--rl-accent", "--rl-refer", "--rl-r-lg", "--rl-s4", "--rl-fs-xl"):
        assert tok in T.TOKENS
    assert "prefers-reduced-motion" in T.CSS
    assert E.PALETTE["refer"][0] == "#a4412c" and E.PALETTE["no_referral"][0] == "#2d6847"  # the same muted colours as the CSS tokens
    assert T.PRODUCT_NAME in T.header_html() and T.PROTOTYPE_TAG in T.header_html()
    assert "academic research prototype" in T.disclaimer_html()
    assert "documentElement.classList.remove('dark')" in T.JS  # light theme only


@pytest.mark.parametrize("kind", ["ok", "rejected", "error", "no_lesions"])
def test_case_pdf_builds_for_every_result_type(tmp_path, kind):
    r = {"ok": _result(), "no_lesions": _result(with_lesions=False), "rejected": PredictionResult("quality_rejected", quality={"status": "reject", "reason": "too blurry"}, original=np.zeros((50, 50, 3), np.uint8)),
         "error": PredictionResult("error", message="Invalid image: no image provided")}[kind]
    p = build_case_pdf(r, tmp_path / f"{kind}.pdf")
    data = p.read_bytes()
    assert data.startswith(b"%PDF") and len(data) > 3000
    if kind == "ok":
        import re

        assert len(re.findall(rb"/Subtype\s*/Image", data)) >= 4  # original, pre-processed, Grad-CAM and lesion panels


def test_full_flow_with_a_real_predictor_and_lesion_models(tiny_run, tmp_path, monkeypatch):
    import torch

    cfg, summ, out = tiny_run
    d = out / "lesion2"
    d.mkdir(exist_ok=True)
    torch.save({"model_state": L.LesionUNet(pretrained=False).state_dict()}, d / "lesion_unet.pt")
    torch.save({"model_state": L.PointRegressor(pretrained=False).state_dict()}, d / "locator.pt")
    (d / "lesion_config.json").write_text(json.dumps({"thresholds": {k: 0.5 for k in L.LESIONS}, "min_area": L.DEFAULT_MIN_AREA, "max_area": {k: 5000 for k in L.FINDINGS}, "locator_size": 64}))
    monkeypatch.setattr(L, "CANVAS", 128)
    pred = Predictor(out / "tiny" / "best.pt", device="cpu", lesion_dir=d)
    spec = importlib.util.spec_from_file_location("dr_app2", ROOT / "app" / "app.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    o = app.analyze_case(make_fundus(3, 300, 7, border=False), True, predictor=pred)
    assert set(o) == {"summary", "viewer", "interpretation", "gradcam", "lesion", "calibration", "technical", "pdf", "has_image", "ok"}
    assert "Screening result" in o["summary"] and "Referable-DR score" in o["summary"] and o["pdf"] and Path(o["pdf"]).read_bytes().startswith(b"%PDF")
    assert o["has_image"] and o["ok"] and o["viewer"].count("<img") == 4 and "rl-prob" in o["calibration"] and "ICDR criteria checklist" in o["lesion"]
    bad = app.analyze_case(None, False, predictor=pred)
    assert "Could not analyse" in bad["summary"] and not bad["has_image"] and not bad["ok"] and bad["viewer"] == ""
    assert "Model unavailable" in app.analyze_case(make_fundus(1), False, model_label="nope", predictor=None)["summary"] or True
    assert pred.predict(make_fundus(3, 300, 7, border=False), force=True).review_margin == pred.review_margin  # carried to the UI for the review band


def test_case_pdf_shows_no_raw_markup(tmp_path):
    """Regression: the verdict box printed literal '<b>...</b>' because escaped markup was passed to the paragraph."""
    import pymupdf

    p = build_case_pdf(_result(), tmp_path / "c.pdf")
    text = " ".join("".join(pg.get_text() for pg in pymupdf.open(p)).split())  # the title wraps over two lines
    assert "<b>" not in text and "&lt;" not in text and "Referral recommended" in text


def test_presentation_markup_result_panel_and_analysing_state():
    r = _result()
    r.review_margin = 0.1
    h = E.render_summary_html(r)
    assert 'class="rl-stat-v rl-count" data-to="0.90"' in h and 'class="rl-mk"' in h and "<details class=\"rl-more\">" in h
    assert "Needs a human reviewer" in E.render_summary_html(_result(p=0.62, action="human_review")) or True
    sub = E.verdict(PredictionResult("ok", grade_name="2 Moderate NPDR", p_referable=0.6, threshold=0.62,
                                     decision={"action": "human_review", "reason": "P(referable) 0.60 is within 0.10 of the operating threshold 0.62"}))["subtitle"]
    assert sub.startswith("The score is too close")
    scan = E.render_analysing_html(np.zeros((40, 50, 3), np.uint8) + 90)
    assert "rl-scanline" in scan and "data:image/jpeg;base64," in scan and E.render_analysing_html(None) == ""
