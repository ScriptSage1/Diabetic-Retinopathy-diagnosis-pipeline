import numpy as np
import pytest
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss, cohen_kappa_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, precision_score, recall_score, roc_auc_score)

from drscreen import metrics as M


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 400)
    logits = rng.normal(size=(400, 5)) + 2.5 * np.eye(5)[y]
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    return y, p


def test_binary_metrics_match_sklearn(toy):
    y5, p5 = toy
    yb, s = M.referable_labels(y5), M.referable_prob(p5)
    t = 0.4
    m = M.binary_metrics(yb, s, t)
    pred = (s >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(yb, pred).ravel()
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (tn, fp, fn, tp)
    assert np.isclose(m["sensitivity"], recall_score(yb, pred))
    assert np.isclose(m["specificity"], recall_score(yb, pred, pos_label=0))
    assert np.isclose(m["precision"], precision_score(yb, pred))
    assert np.isclose(m["npv"], precision_score(yb, pred, pos_label=0))
    assert np.isclose(m["f1"], f1_score(yb, pred))
    assert np.isclose(m["balanced_accuracy"], balanced_accuracy_score(yb, pred))
    assert np.isclose(m["roc_auc"], roc_auc_score(yb, s))
    assert np.isclose(m["pr_auc"], average_precision_score(yb, s))
    assert m["n_positive"] + m["n_negative"] == m["n"] and np.isclose(m["prevalence"], yb.mean())


def test_grade_metrics_match_sklearn(toy):
    y5, p5 = toy
    pred = p5.argmax(1)
    g = M.grade_metrics(y5, pred)
    assert np.isclose(g["qwk"], cohen_kappa_score(y5, pred, weights="quadratic"))
    assert np.isclose(g["accuracy_5"], (y5 == pred).mean())
    assert np.isclose(g["balanced_accuracy_5"], balanced_accuracy_score(y5, pred))
    pr, rc, f1, sup = precision_recall_fscore_support(y5, pred, labels=range(5), zero_division=0)
    for k in range(5):
        assert np.isclose(g["per_grade"][str(k)]["precision"], pr[k]) and np.isclose(g["per_grade"][str(k)]["recall"], rc[k])
        assert np.isclose(g["per_grade"][str(k)]["f1"], f1[k]) and g["per_grade"][str(k)]["support"] == sup[k]
    assert np.isclose(g["macro_f1"], f1.mean())
    assert (np.array(g["confusion_matrix"]) == confusion_matrix(y5, pred, labels=range(5))).all()


def test_qwk_penalises_far_errors_more():
    y = np.array([0, 1, 2, 3, 4] * 20)
    near = np.clip(y + 1, 0, 4)
    far = (y + 3) % 5
    assert M.quadratic_weighted_kappa(y, y) == 1.0
    assert M.quadratic_weighted_kappa(y, near) > M.quadratic_weighted_kappa(y, far)


def test_ece_hand_example():
    conf = np.array([0.9, 0.9, 0.9, 0.9, 0.6, 0.6])
    correct = np.array([1, 1, 1, 0, 1, 0])  # bin(0.9): acc .75 gap .15 weight 4/6; bin(0.6): acc .5 gap .1 weight 2/6
    assert np.isclose(M.expected_calibration_error(conf, correct, 10), 4 / 6 * 0.15 + 2 / 6 * 0.10)
    assert M.expected_calibration_error(np.ones(5), np.ones(5)) == 0.0


def test_brier_matches_sklearn(toy):
    y5, p5 = toy
    yb, s = M.referable_labels(y5), M.referable_prob(p5)
    assert np.isclose(M.brier_binary(yb, s), brier_score_loss(yb, s))


def test_temperature_scaling_reduces_overconfident_nll():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 5, 1500)
    logits = (rng.normal(size=(1500, 5)) + 1.2 * np.eye(5)[y]) * 4.0  # deliberately over-confident
    T = M.fit_temperature(logits, y)
    assert T > 1.5 and M.nll(logits, y, T) < M.nll(logits, y, 1.0)


def test_threshold_selected_on_validation_meets_target():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 500)
    s = np.clip(0.5 * y + rng.normal(0.3, 0.2, 500), 0, 1)
    sel = M.select_threshold(y, s, 0.90)
    m = M.binary_metrics(y, s, sel["threshold"])
    assert m["sensitivity"] >= 0.90 and sel["fitted_on"] == "validation"
    # a higher threshold must violate the target (we picked the highest feasible one)
    assert M.binary_metrics(y, s, min(1.0, sel["threshold"] + 0.02))["sensitivity"] < 0.90 or sel["threshold"] >= 0.98


def test_undefined_metrics_are_nan_not_crash():
    m = M.binary_metrics(np.zeros(10, int), np.linspace(0, 1, 10), 0.5)  # no positives
    assert np.isnan(m["sensitivity"]) and np.isnan(m["roc_auc"]) and np.isnan(m["pr_auc"])
    assert np.isnan(M.quadratic_weighted_kappa([], []))


def test_bootstrap_ci_brackets_point_estimate_and_is_reproducible(toy):
    y5, p5 = toy
    thr = 0.5
    ci1 = M.bootstrap_ci(y5, p5, thr, n_boot=200, seed=3)
    ci2 = M.bootstrap_ci(y5, p5, thr, n_boot=200, seed=3)
    assert ci1["sensitivity"] == ci2["sensitivity"]
    pt = M.binary_metrics(M.referable_labels(y5), M.referable_prob(p5), thr)
    for k in ("sensitivity", "specificity", "roc_auc"):
        assert ci1[k]["lo"] <= pt[k] <= ci1[k]["hi"]
    assert ci1["_method"]["n_boot"] == 200 and ci1["_method"]["seed"] == 3


def test_paired_bootstrap_detects_clear_difference(toy):
    y5, p5 = toy
    noise = np.random.default_rng(5).dirichlet(np.ones(5), len(y5))
    d = M.paired_bootstrap_diff(y5, p5, noise, "roc_auc", n_boot=200)
    assert d["point_diff"] > 0.2 and d["lo"] > 0


def test_evaluate_predictions_and_table_are_json_safe(toy, tmp_path):
    import json

    y5, p5 = toy
    rep = M.evaluate_predictions(y5, p5, 0.5, "toy", n_boot=50)
    M.save_json(rep, tmp_path / "m.json")
    back = json.loads((tmp_path / "m.json").read_text())
    rows = M.metrics_table_rows(back)
    assert len(rows) >= 12 and all(len(r) == 5 for r in rows)
