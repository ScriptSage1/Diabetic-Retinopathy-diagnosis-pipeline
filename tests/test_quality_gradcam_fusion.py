import numpy as np
import pytest
import torch

from drscreen.fusion import decide, pipeline_metrics
from drscreen.gradcam import GradCAM, overlay_heatmap, randomization_sanity_check
from drscreen.models import build_model
from drscreen.preprocess import preprocess_for_model
from drscreen.quality import DEGRADATIONS, QualityThresholds, assess_quality as _assess, degrade, enhance_image
from drscreen.synthetic import make_fundus

# The default thresholds are untuned guesses (the real pipeline fits them from training-set percentiles); the smooth
# synthetic fixtures need explicit, lower sharpness/contrast thresholds to count as "good".
SYN = QualityThresholds(reject_blur=2.0, enhance_contrast_low=12.0, reject_contrast_low=5.0)


def assess_quality(img, thr=SYN):
    return _assess(img, thr)


def test_good_image_passes_and_result_is_structured():
    r = assess_quality(make_fundus(1, 400, 0, border=False))
    assert r.status in ("good", "enhance") and r.reason and 0 <= r.score <= 1 and "blur" in r.metrics
    assert "heuristic" in r.method


def test_invalid_inputs_are_rejected_not_raised():
    for bad in (None, np.zeros((300, 300, 3), np.uint8), np.zeros((20, 20, 3), np.uint8), "x"):
        r = assess_quality(bad)
        assert r.status == "reject" and r.reason


def test_blur_dark_and_crop_are_flagged_with_right_reason():
    img = make_fundus(2, 400, 1)
    assert assess_quality(img).status != "reject"
    assert assess_quality(degrade(img, "blur", 3)).status == "reject"
    assert "blur" in assess_quality(degrade(img, "blur", 3)).reason
    assert assess_quality(degrade(img, "dark", 3)).status == "reject"
    assert assess_quality(degrade(img, "cropped_fov", 3)).status == "reject"


def test_enhancement_changes_dark_image_but_keeps_shape():
    img = degrade(make_fundus(0, 300, 2), "dark", 1)
    e = enhance_image(img)
    assert e.shape == img.shape and e.mean() > img.mean()


def test_all_degradations_valid():
    img = make_fundus(0, 300, 3)
    for k in DEGRADATIONS:
        d = degrade(img, k, 2)
        assert d.shape == img.shape and d.dtype == np.uint8


@pytest.mark.parametrize("arch", ["small_cnn", "efficientnet_b0"])
def test_gradcam_shape_range_and_sanity(arch):
    m = build_model(arch, 5, pretrained=False).eval()
    _, _, x = preprocess_for_model(make_fundus(3, 200, 4), 96, True)
    xt = torch.from_numpy(x)[None]
    gc = GradCAM(m)
    cam, logits = gc(xt, "pred")
    gc.remove()
    assert cam.shape == (96, 96) and 0 <= cam.min() and cam.max() <= 1 + 1e-6 and logits.shape == (5,)
    plain, _, _ = preprocess_for_model(make_fundus(3, 200, 4), 96, True)
    ov = overlay_heatmap(plain, cam)
    assert ov.shape == plain.shape and ov.dtype == np.uint8
    out = randomization_sanity_check(m, xt)
    assert "spearman" in out
    if arch == "efficientnet_b0":
        assert np.isfinite(out["spearman"])  # randomised model must still produce a non-constant map


def test_gradcam_referable_target_runs():
    m = build_model("small_cnn", 5, pretrained=False).eval()
    _, _, x = preprocess_for_model(make_fundus(2, 200, 5), 64, True)
    gc = GradCAM(m)
    cam, _ = gc(torch.from_numpy(x)[None], "referable")
    gc.remove()
    assert np.isfinite(cam).all()


def test_decide_logic():
    assert decide("reject", 0.99, 0.5).action == "recapture"
    assert decide("good", 0.9, 0.5).action == "refer"
    assert decide("good", 0.1, 0.5).action == "no_referral"
    assert decide("enhance", 0.52, 0.5, 0.05).action == "human_review"
    assert decide("good", None, 0.5).action == "human_review"


def test_pipeline_metrics_accounts_for_every_image():
    y = np.array([0, 0, 3, 3, 4, 1, 2, 0])
    p = np.array([0.1, 0.9, 0.9, 0.2, 0.8, 0.3, 0.52, 0.5])
    st = np.array(["good", "good", "good", "reject", "good", "good", "good", "enhance"])
    m = pipeline_metrics(y, p, st, 0.5, 0.05)
    assert m["n"] == 8 and m["n_recapture"] == 1 and m["n_human_review"] == 2 and m["n_auto"] == 5
    assert m["conservative"]["fn"] == 0  # the missed grade-3 case was rejected -> counted as referred
    assert m["referable_in_recapture"] == 1
