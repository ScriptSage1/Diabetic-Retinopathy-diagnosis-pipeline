import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from drscreen.predict import Predictor
from drscreen.synthetic import make_fundus

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def predictor(tiny_run):
    cfg, summ, out = tiny_run
    (out / "tiny" / "calibration.json").write_text(json.dumps({"temperature": 1.3, "threshold": 0.4, "review_margin": 0.05}))
    return Predictor(out / "tiny" / "best.pt", device="cpu")


def _load_app():
    spec = importlib.util.spec_from_file_location("dr_app", ROOT / "app" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_predict_valid_image(predictor):
    r = predictor.predict(make_fundus(3, 300, 7, border=False))
    assert r.status in ("ok", "quality_rejected")
    if r.status == "ok":
        assert len(r.grade_probs) == 5 and np.isclose(sum(r.grade_probs), 1.0)
        assert 0 <= r.p_referable <= 1 and r.gradcam.shape == r.processed.shape
        assert r.referable == (r.p_referable >= 0.4)
        assert "not a medical diagnosis" in r.to_markdown()


def test_predict_is_reproducible(predictor):
    img = make_fundus(2, 300, 8, border=False)
    a, b = predictor.predict(img), predictor.predict(img)
    assert a.status == b.status and a.grade_probs == b.grade_probs


def test_predictor_without_lesion_models_says_evidence_not_computed(predictor):
    r = predictor.predict(make_fundus(3, 300, 7, border=False), force=True)
    assert r.status == "ok" and r.lesion_evidence["available"] is False and r.lesion_overlay is None
    assert "Not computed" in r.to_markdown() and r.cross_check["status"] == "no_lesion_evidence"


def test_predictor_with_lesion_models_returns_reasons(tiny_run, tmp_path, monkeypatch):
    import torch

    from drscreen import lesion as L

    cfg, summ, out = tiny_run
    d = out / "lesion"
    d.mkdir(exist_ok=True)
    torch.save({"model_state": L.LesionUNet(pretrained=False).state_dict()}, d / "lesion_unet.pt")
    torch.save({"model_state": L.PointRegressor(pretrained=False).state_dict()}, d / "locator.pt")
    (d / "lesion_config.json").write_text(json.dumps({"thresholds": {k: 0.5 for k in L.LESIONS}, "min_area": L.DEFAULT_MIN_AREA, "locator_size": 64}))
    monkeypatch.setattr(L, "CANVAS", 128)
    p = Predictor(out / "tiny" / "best.pt", device="cpu", lesion_dir=d)
    r = p.predict(make_fundus(3, 300, 7, border=False), force=True)
    assert r.status == "ok" and r.lesion_evidence["available"] and r.lesion_overlay is not None
    md = r.to_markdown()
    assert "ICDR criterion" in md and "not assessed" in md and "Cross-check" in md and "Why: lesion evidence" in md
    assert "Localisation:" in md  # says how disc / fovea positions were obtained
    assert r.cross_check["status"] in ("agree", "lesions_only", "classifier_only")


def test_predict_rejects_bad_inputs_gracefully(predictor):
    for bad in (None, np.zeros((5, 5, 3), np.uint8), np.zeros((300, 300, 3), np.uint8), "not_a_file.png", object()):
        r = predictor.predict(bad)
        assert r.status in ("error", "quality_rejected") and r.grade is None


def test_forced_analysis_of_rejected_image_is_flagged(predictor):
    from drscreen.quality import degrade

    img = degrade(make_fundus(2, 300, 9, border=False), "blur", 3)
    assert predictor.predict(img).status == "quality_rejected"
    r = predictor.predict(img, force=True)
    assert r.status == "ok" and r.forced_despite_rejection


def test_gradio_analyze_function(predictor):
    app = _load_app()
    o, p, g, lo, probs, md = app.analyze(make_fundus(3, 300, 7, border=False), False, predictor=predictor)
    assert isinstance(md, str) and md
    o, p, g, lo, probs, md = app.analyze(None, False, predictor=predictor)
    assert o is None and "Could not analyse" in md
    app._PREDICTOR, app._LOAD_ERROR = None, "no model"
    _, _, _, _, _, md = app.analyze(make_fundus(1), False)
    assert "Model unavailable" in md


def test_gradio_ui_builds():
    pytest.importorskip("gradio")
    assert _load_app().build_ui() is not None


def test_report_generates_pdf_with_figures_and_tables(tmp_path):
    from drscreen import metrics as M
    from drscreen import plots as P
    from drscreen.report import build_report

    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 300)
    p = np.exp(rng.normal(size=(300, 5)) + 2 * np.eye(5)[y])
    p /= p.sum(1, keepdims=True)
    rep = M.evaluate_predictions(y, p, 0.5, "toy set", n_boot=30)
    res = tmp_path / "results"
    res.mkdir()
    M.save_json({"internal_holdout": rep}, res / "metrics.json")
    fig = tmp_path / "report" / "figures"
    P.plot_binary_confusion(rep["screening"], fig / "confusion_referable.png")
    P.plot_roc_pr(y, p, 0.5, fig / "roc_pr.png")
    tpl = tmp_path / "t.md"
    tpl.write_text("# Results\nSensitivity {{val:internal_holdout.screening.sensitivity|.3f}}; missing {{val:nope.x}}.\n\n{{table:main_metrics}}\n\n"
                   "{{figure:confusion_referable.png|Confusion matrix.}}\n\n{{figure:roc_pr.png|ROC and PR.}}\n\n{{figure:absent.png|Absent.}}\n\n> note ≥ ≤ →\n",
                   encoding="utf-8")
    stats = build_report(tpl, res, tmp_path / "report" / "R.md", tmp_path / "report" / "R.pdf", {"title": "Test report", "author": "x"})
    md = (tmp_path / "report" / "R.md").read_text()
    assert f"{rep['screening']['sensitivity']:.3f}" in md and "N/A (not computed)" in md
    assert stats["images"] >= 2 and stats["pages"] >= 3 and (tmp_path / "report" / "R.pdf").stat().st_size > 10_000
    assert "val:nope.x" in (tmp_path / "report" / "missing_values.txt").read_text()
    assert "figure:absent.png" in (tmp_path / "report" / "missing_values.txt").read_text()


def test_simulation_littles_law_and_hand_calculation():
    sys.path.insert(0, str(ROOT / "sim"))
    from screening_simulation import SimConfig, run

    cfg = SimConfig(annual_patients=40_000, n_cameras=4, imaging_min=6.0, n_reviewers=3, p_quality_reject=0.0, sim_days=60, warmup_days=5,
                    sensitivity=0.9, specificity=0.9, p_uncertain=0.0, upload_min=1.0)
    r = run(cfg, service="exponential")
    ll = r["littles_law"]
    assert abs(ll["lambda_times_W"] - ll["L_observed"]) / ll["L_observed"] < 0.08  # L = lambda * W
    # hand calculation: camera utilisation rho = lambda * E[S] / c
    rho = cfg.arrival_rate * cfg.imaging_min / cfg.n_cameras
    assert abs(r["utilisation"]["cameras"] - rho) < 0.06
    assert r["arrived_per_day"] == pytest.approx(cfg.annual_patients / cfg.working_days, rel=0.1)


def test_app_looks_where_the_pipeline_writes_the_checkpoint():
    """Regression: the pipeline saves artifacts/<arch>/best.pt but the app/installer looked in artifacts/effnet_b0/."""
    from drscreen.train import TrainConfig

    app = _load_app()
    default = TrainConfig()
    assert any(Path(c).parent.name == default.run_name == default.arch for c in app.CANDIDATES)  # the standard run's directory is searched
    assert Path(app.CANDIDATES[0]).parent.name == default.arch + "_robust"  # robust model preferred when present
