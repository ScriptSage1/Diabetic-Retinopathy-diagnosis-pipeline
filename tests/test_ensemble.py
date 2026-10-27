import numpy as np

from drscreen import ensemble as EN


def _fake_sets(seed=0, n=400):
    rng = np.random.default_rng(seed)

    def make(n_):
        y = rng.integers(0, 5, n_)
        d = {"image_id": np.arange(n_).astype(str), "grade": y}
        for name, q in (("efficientnet_b0_robust", 1.8), ("efficientnet_b3_robust", 2.4)):
            base = rng.normal(size=(n_, 5)) + q * np.eye(5)[y]
            d[f"{name}__plain"] = base + rng.normal(scale=0.5, size=base.shape)
            d[f"{name}__flip"] = base + rng.normal(scale=0.5, size=base.shape)
        return d

    return {"val": make(n), "test": make(n), "idrid_test": make(100)}


def test_candidates_enumerated_selected_on_validation_and_decision_is_consistent():
    sets = _fake_sets()
    runs = ["efficientnet_b0_robust", "efficientnet_b3_robust"]
    temps = {r: 1.5 for r in runs}
    res = EN.evaluate_candidates(sets, temps, runs, (("efficientnet_b0_robust",), False), n_boot=60)
    assert len(res["candidates"]) == 6  # 3 subsets x {plain, TTA}
    best_val = max(c["val_score"] for c in res["candidates"])
    assert abs(res["selected"]["val_score"] - best_val) < 1e-12  # chosen purely by validation score
    d = res["decision"]
    assert d["adopt_selected"] == (not d["selected_is_default"] and d["gain_at_least_0.003"] and d["non_inferior_on_test"])
    assert all("idrid_auc" in c for c in res["candidates"]) and "paired_selected_minus_default_test" in res


def test_better_member_wins_and_ensemble_probabilities_are_valid():
    sets = _fake_sets(1)
    temps = {"efficientnet_b0_robust": 1.0, "efficientnet_b3_robust": 1.0}
    p = EN.candidate_probs(sets["val"], ("efficientnet_b0_robust", "efficientnet_b3_robust"), True, temps)
    assert p.shape == (400, 5) and np.allclose(p.sum(1), 1.0)
    res = EN.evaluate_candidates(sets, temps, list(temps), (("efficientnet_b0_robust",), False), n_boot=40)
    sel = res["selected"]
    assert "efficientnet_b3_robust" in sel["members"]  # the stronger synthetic member is part of the best candidate


def test_tta_logits_are_the_mean_of_three_views():
    s = {"m__plain": np.array([[3.0, 0, 0, 0, 0]]), "m__flip": np.array([[0.0, 3.0, 0, 0, 0]])}
    assert np.allclose(EN._logits(s, "m", True), [[1.0, 2.0, 0, 0, 0]])  # (plain + 2*flip) / 3
    assert np.allclose(EN._logits(s, "m", False), s["m__plain"])


def test_predictor_ensemble_and_tta_run_and_average_member_probabilities(tiny_run, tmp_path):
    import json

    from drscreen.models import grade_probs
    from drscreen.predict import Predictor
    from drscreen.synthetic import make_fundus

    cfg, summ, out = tiny_run
    ck = out / "tiny" / "best.pt"
    ens = tmp_path / "artifacts" / "ensemble" / "ensemble.json"
    ens.parent.mkdir(parents=True)
    ens.write_text(json.dumps({"members": [str(ck), str(ck)], "tta": True, "threshold": 0.4, "review_margin": 0.05, "temperatures": {"tiny": 1.7}}))
    p = Predictor(ck, device="cpu", ensemble=ens)
    assert len(p.members) == 2 and p.tta and p.threshold == 0.4 and "ensemble" in p.model_name and "TTA" in p.model_name
    img = make_fundus(2, 300, 5, border=False)
    r = p.predict(img, with_gradcam=False, with_lesions=False, force=True)
    assert r.status == "ok" and abs(sum(r.grade_probs) - 1) < 1e-6 and r.threshold == 0.4 and "ensemble" in r.model_name
    # identical members -> the ensemble equals one member with the same temperature and the same TTA views
    single = Predictor(ck, device="cpu")
    single.temperature, single.tta = 1.7, False
    import torch

    from drscreen.preprocess import preprocess_for_model

    _, _, x = preprocess_for_model(img, single.img_size, single.use_bg)
    xt = torch.from_numpy(x)[None]
    with torch.no_grad():
        lg = np.mean([single.model(v).numpy()[0] for v in (xt, torch.flip(xt, [3]), torch.flip(xt, [2]))], axis=0)
    assert np.allclose(r.grade_probs, grade_probs(lg[None], 1.7)[0], atol=1e-5)
