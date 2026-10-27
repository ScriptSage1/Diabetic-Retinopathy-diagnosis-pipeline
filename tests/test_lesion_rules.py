import json

import numpy as np
import pandas as pd
import pytest
import torch

from drscreen import lesion as L
from drscreen import rules as R
from drscreen.fusion import cross_check, decide_with_evidence
from drscreen.synthetic import make_fundus


# ------------------------------------------------------------------ geometry
def test_fov_roundtrip_points_and_mask_area():
    img = np.zeros((301, 423, 3), np.uint8)
    img[40:260, 100:330] = 120
    fov = L.make_fov(img, 256)
    assert fov.to_canvas(img).shape == (256, 256, 3)
    pt = np.array([[200.0, 150.0]])
    assert np.allclose(fov.pts_to_orig(fov.pts_to_canvas(pt)), pt, atol=1e-6)
    m = np.zeros((301, 423), np.uint8)
    m[148:153, 198:203] = 1  # 25 px lesion
    assert 5 <= fov.to_canvas(m, mask=True).sum() <= 40  # small lesions survive resizing (area interpolation)
    assert L.Fov.from_array(fov.as_array()) == fov


def test_quadrants_hand_checked():
    od, fovea = (50, 100), (100, 100)  # temporal direction = +x, up = -y
    assert L.quadrant_of((200, 20), fovea, od) == 0   # temporal-superior
    assert L.quadrant_of((200, 200), fovea, od) == 1  # temporal-inferior
    assert L.quadrant_of((10, 10), fovea, od) == 2    # nasal-superior
    assert L.quadrant_of((10, 200), fovea, od) == 3   # nasal-inferior
    assert L.quadrant_of((200, 20), (100, 100), (150, 100)) == 2  # OD on the other side flips temporal/nasal


# ------------------------------------------------------------------ model + loss + batches
def test_unet_shapes_loss_and_batch():
    m = L.LesionUNet(pretrained=False).eval()
    with torch.no_grad():
        out = m(torch.randn(1, 3, 96, 96))
    assert tuple(out.shape) == (1, 5, 96, 96)
    imgs = [np.random.default_rng(0).integers(0, 255, (160, 160, 3), dtype=np.uint8)]
    masks = [np.zeros((5, 160, 160), np.uint8)]
    masks[0][0, 80, 80] = 1
    masks[0][1, 20:40, 20:40] = 1
    x, y = L.sample_batch(imgs, masks, 4, 64, np.random.default_rng(1))
    assert tuple(x.shape) == (4, 3, 64, 64) and tuple(y.shape) == (4, 5, 64, 64)
    logits = torch.randn(4, 5, 64, 64, requires_grad=True)
    loss = L.seg_loss(logits, y)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_pixelstats_aupr_matches_sklearn():
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(3)
    y = (rng.random((2, 5, 90, 90)) < 0.08).astype(np.uint8)
    p = np.clip(0.5 * y + rng.normal(0.25, 0.2, y.shape), 0, 1).astype(np.float32)
    st = L.PixelStats(n_bins=1000)
    for i in range(2):
        st.add(p[i], y[i])
    for k in range(5):
        ref = average_precision_score(y[:, k].ravel(), p[:, k].ravel())
        assert abs(st.aupr(k) - ref) < 0.01, (k, st.aupr(k), ref)


def test_pixelstats_perfect_and_random_predictions():
    rng = np.random.default_rng(0)
    masks = np.zeros((5, 64, 64), np.uint8)
    masks[1, 10:30, 10:30] = 1
    perfect = masks.astype(np.float32) * 0.9 + 0.05
    st = L.PixelStats()
    st.add(perfect, masks)
    assert st.aupr(1) > 0.99
    thr, f1 = st.best_threshold(1)
    assert f1 > 0.99 and st.at_threshold(1, 0.5)["dice"] > 0.99
    st2 = L.PixelStats()
    st2.add(rng.random((5, 64, 64)).astype(np.float32), masks)
    assert st2.aupr(1) < 0.5
    assert np.isnan(st.aupr(0))  # class absent -> undefined, not 0


# ------------------------------------------------------------------ components and features
def test_components_and_features_counts_by_quadrant():
    H = 400
    probs = np.zeros((5, H, H), np.float32)
    od, fovea = np.array([120.0, 200.0]), np.array([220.0, 200.0])

    def blob(k, cx, cy, r=6):
        yy, xx = np.ogrid[:H, :H]
        probs[L.LESIONS.index(k)][(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = 0.9

    blob("HE", 300, 100)   # temporal-superior
    blob("HE", 300, 300)   # temporal-inferior
    blob("HE", 60, 100)    # nasal-superior
    blob("EX", 230, 205, 8)  # right next to the fovea
    blob("MA", 330, 330, 2)
    thr = {k: 0.5 for k in L.LESIONS}
    f, comps = L.extract_features(probs, thr, od, fovea, L.DEFAULT_MIN_AREA, od_mask_area=np.pi * 40**2)
    assert f["HE_count"] == 3 and f["EX_count"] == 1 and f["MA_count"] == 1 and f["SE_count"] == 0
    assert [f[f"HE_q{q}"] for q in range(4)] == [1, 1, 1, 0]
    assert f["HE_quadrants_ge1"] == 3 and f["HE_min_quadrant_count"] == 0
    assert f["EX_within_1DD"] == 1  # DD = 80 px, the exudate is ~11 px from the fovea
    assert set(L.FEATURE_NAMES) <= set(f)


def test_analyzer_end_to_end_with_random_weights(tmp_path, monkeypatch):
    d = tmp_path / "lesion"
    d.mkdir()
    torch.save({"model_state": L.LesionUNet(pretrained=False).state_dict()}, d / "lesion_unet.pt")
    torch.save({"model_state": L.PointRegressor(pretrained=False).state_dict()}, d / "locator.pt")
    (d / "lesion_config.json").write_text(json.dumps({"thresholds": {k: 0.5 for k in L.LESIONS}, "min_area": L.DEFAULT_MIN_AREA, "locator_size": 64}))
    monkeypatch.setattr(L, "CANVAS", 128)
    an = L.LesionAnalyzer(d, "cpu")
    out = an.analyze(make_fundus(3, 200, 1, border=False))
    assert out["overlay"].shape == (128, 128, 3) and set(L.FEATURE_NAMES) <= set(out["features"])
    ev = R.lesion_evidence(out)
    assert ev["available"] and "markdown" in ev and ev["rule"]["grade"] in (0, 1, 2, 3)


# ------------------------------------------------------------------ rule engine (hand-made inputs)
def test_rules_grade_ladder():
    base = {"MA_count": 0, "HE_count": 0, "EX_count": 0, "SE_count": 0}
    assert R.apply_rules(base).grade == 0
    assert R.apply_rules({**base, "MA_count": 6}).grade == 1                      # only microaneurysms -> mild
    assert R.apply_rules({**base, "MA_count": 6, "HE_count": 2}).grade == 2        # haemorrhages -> moderate
    assert R.apply_rules({**base, "EX_count": 3}).grade == 2
    assert R.apply_rules({**base, "SE_count": 1}).grade == 2
    four = {**base, "HE_count": 100, "HE_q0": 25, "HE_q1": 25, "HE_q2": 25, "HE_q3": 25, "HE_min_quadrant_count": 25}
    r = R.apply_rules(four)
    assert r.grade == 3 and r.referable and "4-2-1" in r.reasons[0]
    three_q = {**four, "HE_q3": 0, "HE_min_quadrant_count": 0}
    assert R.apply_rules(three_q).grade == 2  # 3 quadrants is not the '4' criterion


def test_rules_never_claim_what_they_cannot_assess():
    r = R.apply_rules({"MA_count": 50, "HE_count": 90, "EX_count": 40, "SE_count": 9, "HE_min_quadrant_count": 99})
    assert r.grade <= 3  # grade 4 (neovascularisation) can never be rule-derived
    status = {c["name"]: c["status"] for c in r.criteria}
    assert all(v == "not_assessed" for k, v in status.items() if "venous" in k or "IRMA" in k or "Proliferative" in k)
    assert len(r.not_assessed) == 3 and "Proliferative" in " ".join(r.not_assessed)
    assert "| ICDR criterion |" in r.to_markdown()


def test_rules_reasons_and_macular_flag():
    r = R.apply_rules({"MA_count": 3, "HE_count": 4, "EX_count": 5, "SE_count": 0, "HE_quadrants_ge1": 2, "EX_within_1DD": 2})
    assert r.dme_flag and "fovea" in r.dme_detail
    text = " ".join(r.reasons)
    assert "4 haemorrhage" in text and "5 hard-exudate" in text and "moderate" in text
    r0 = R.apply_rules({"MA_count": 0, "HE_count": 0, "EX_count": 0, "SE_count": 0, "EX_within_1DD": 0})
    assert not r0.dme_flag and not r0.referable


def test_vectorised_grades_match_single_rule_and_calibration_improves_fit():
    rng = np.random.default_rng(0)
    rows = [{"MA_count": int(rng.integers(0, 12)), "HE_count": int(rng.integers(0, 12)), "EX_count": int(rng.integers(0, 6)), "SE_count": 0, "HE_min_quadrant_count": int(rng.integers(0, 4))} for _ in range(200)]
    df = pd.DataFrame(rows)
    p = R.RuleParams(min_ma=2, min_he=3, min_ex=2, min_se=1, he_per_quadrant=3)
    assert (R.grade_vector(df, p) == np.array([R.apply_rules(r, p).grade for r in rows])).all()
    # labels generated by a hidden threshold rule; the calibration should recover a rule at least as good as the literal one
    hidden = R.grade_vector(df, R.RuleParams(min_ma=4, min_he=5, min_ex=3, min_se=1, he_per_quadrant=3))
    from drscreen.metrics import quadratic_weighted_kappa as qwk

    fitted = R.fit_rule_params(df, hidden)
    assert qwk(hidden, R.grade_vector(df, fitted)) >= qwk(hidden, R.grade_vector(df, R.RuleParams())) and fitted.name.startswith("data-calibrated")


def test_cross_check_policy():
    assert cross_check(0.8, 0.5, {"referable": True})["status"] == "agree"
    assert cross_check(0.2, 0.5, {"referable": False})["status"] == "agree"
    c = cross_check(0.2, 0.5, {"referable": True})
    assert c["status"] == "lesions_only" and c["review"]
    c2 = cross_check(0.9, 0.5, {"referable": False})
    assert c2["status"] == "classifier_only" and not c2["review"]
    assert cross_check(0.9, 0.5, None)["status"] == "no_lesion_evidence"
    assert decide_with_evidence("good", 0.2, 0.5, 0.0, c).action == "human_review"
    assert decide_with_evidence("good", 0.9, 0.5, 0.0, c2).action == "refer"
    assert decide_with_evidence("reject", 0.2, 0.5, 0.0, c).action == "recapture"


def test_lesion_evidence_without_models_says_not_computed():
    ev = R.lesion_evidence(None)
    assert ev["available"] is False and "NOT computed" in ev["reason"]


# ------------------------------------------------------------------ evaluation on synthetic features
def _synthetic_features(n, rng):
    rows = []
    for i in range(n):
        g = int(rng.choice(5, p=[0.4, 0.1, 0.25, 0.1, 0.15]))
        r = {name: 0 for name in L.FEATURE_NAMES}
        r.update({"image_id": f"i{i}", "grade": g, "loc_ok": 1, "MA_count": int(rng.poisson(1 + 3 * g)), "HE_count": int(rng.poisson(2 * g)),
                  "EX_count": int(rng.poisson(g)), "SE_count": int(rng.poisson(0.3 * g)), "HE_min_quadrant_count": int(rng.poisson(0.5 * g)),
                  "EX_within_1DD": int(rng.poisson(0.3 * g)), "dme": int(g >= 3), "in_lesion_train_or_val": 0})
        rows.append(r)
    return pd.DataFrame(rows)


def test_lesion_fusion_evaluation_protocol_runs_and_is_consistent():
    from drscreen.lesion_eval import evaluate_idrid_external, evaluate_lesion_fusion

    rng = np.random.default_rng(0)
    f = _synthetic_features(900, rng)
    f["split"] = rng.choice(["train", "val", "test"], len(f), p=[0.7, 0.15, 0.15])
    ev = f[f["split"] != "train"]
    logits = rng.normal(size=(len(ev), 5)) + 2.0 * np.eye(5)[ev["grade"].to_numpy()]
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    preds = pd.DataFrame(p, columns=[f"p_grade{k}" for k in range(5)])
    preds.insert(0, "image_id", ev["image_id"].values)
    preds["split"] = ev["split"].values
    out = evaluate_lesion_fusion(f, preds, 0.5, n_boot=50)
    names = [r["name"] for r in out["rows"]]
    assert names[0].startswith("EfficientNet-B0 classifier alone") and any("Fusion" in n for n in names) and any("escalation" in n for n in names)
    cc = out["cross_check"]["literal"]
    assert sum(cc["status_counts"].values()) == out["n_test"]
    assert cc["with_escalation"]["fn"] <= cc["classifier_alone"]["fn"]  # escalation can only add referrals
    assert out["rule_params"]["calibrated"]["name"].startswith("data-calibrated")
    # external IDRiD-style evaluation on the same synthetic features (smoke of the code path)
    idr = f.copy()
    idr["split"] = "test"
    cnn = pd.DataFrame(np.exp(rng.normal(size=(len(idr), 5))), columns=[f"p_grade{k}" for k in range(5)])
    cnn = cnn.div(cnn.sum(axis=1), axis=0)
    cnn.insert(0, "image_id", idr["image_id"].values)
    ext = evaluate_idrid_external(idr, cnn, f, 0.5, out["rule_params"], n_boot=50)
    assert ext["n_evaluated"] == len(idr) and "dme_proxy" in ext and ext["rows"][0]["name"].startswith("EfficientNet")


def test_attach_labels_keeps_duplicate_idrid_names_apart():
    """Regression: IDRiD train and test both contain IDRiD_001; merging on the name alone cross-joined them (722 rows instead of 516)."""
    from drscreen.lesion_pipeline import attach_labels

    df = pd.DataFrame({"image_id": ["IDRiD_001", "IDRiD_002", "IDRiD_001", "IDRiD_002"], "split": ["train", "train", "test", "test"],
                       "grade": [0, 1, 4, 3], "dme": [0, 0, 2, 1], "in_lesion_train_or_val": [0, 1, 0, 0]})
    rows = [{"image_id": i, "MA_count": k} for k, i in enumerate(df["image_id"])]
    out = attach_labels(rows, df)
    assert len(out) == 4 and list(out["split"]) == ["train", "train", "test", "test"]
    assert list(out["MA_count"]) == [0, 1, 2, 3] and list(out["grade"]) == [0, 1, 4, 3]  # train 001 keeps ITS label, test 001 keeps its own
    import pytest

    with pytest.raises(ValueError):
        attach_labels(rows[:3], df)
    with pytest.raises(ValueError):
        attach_labels(rows[::-1], df)


def test_locate_od_fovea_uses_segmented_disc_and_falls_back_to_prior():
    H = 600
    probs = np.zeros((5, H, H), np.float32)
    yy, xx = np.ogrid[:H, :H]
    probs[L.LESIONS.index("OD")][(yy - 200) ** 2 + (xx - 120) ** 2 <= 40**2] = 0.9  # disc at (120, 200), diameter ~80 px
    # regressor agrees with the segmented disc -> its fovea is kept
    od, fo, info = L.locate_od_fovea(probs, np.array([[125.0, 205.0], [330.0, 290.0]]), 0.5)
    assert np.allclose(od, [120, 200], atol=2) and np.allclose(fo, [330, 290]) and info["fovea_source"] == "regressor" and info["fallback"] == 0
    # regressor puts the disc far away (domain shift) -> disc from segmentation, fovea from the anatomical prior toward the retina centre
    od, fo, info = L.locate_od_fovea(probs, np.array([[480.0, 90.0], [500.0, 120.0]]), 0.5)
    assert np.allclose(od, [120, 200], atol=2) and info["fovea_source"] == "anatomical prior" and info["fallback"] == 1
    dd = 2 * np.sqrt(info["od_area"] / np.pi)
    assert abs(np.linalg.norm(fo - od) - 2.5 * dd) < 1e-6 and fo[0] > od[0]  # moves toward the centre (x = 300)
    # no disc segmented -> keep the regressor, flagged as fallback
    od, fo, info = L.locate_od_fovea(np.zeros((5, H, H), np.float32), np.array([[10.0, 20.0], [30.0, 40.0]]), 0.5)
    assert info["od_source"] == "regressor" and info["fallback"] == 1
