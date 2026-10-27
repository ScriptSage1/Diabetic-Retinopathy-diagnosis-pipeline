import numpy as np
import pandas as pd
import pytest
import torch

from drscreen.models import build_model, grade_probs, load_checkpoint, referable_prob
from drscreen.train import TrainConfig, train, verify_checkpoint_roundtrip


def test_model_output_shapes():
    for arch in ("small_cnn", "efficientnet_b0"):
        m = build_model(arch, 5, pretrained=False).eval()
        with torch.no_grad():
            o = m(torch.randn(2, 3, 96, 96))
        assert tuple(o.shape) == (2, 5) and torch.isfinite(o).all()


def test_probability_helpers():
    p = grade_probs(np.zeros((1, 5)))
    assert np.allclose(p, 0.2) and np.isclose(referable_prob(p)[0], 0.6)
    z = np.array([[2.0, 0, 0, 0, 0]])
    assert grade_probs(z, temperature=10)[0, 0] < grade_probs(z)[0, 0]


def test_training_loop_runs_and_is_finite(tiny_run):
    cfg, summ, out = tiny_run
    assert summ["epochs_completed"] == 2
    h = pd.read_csv(out / "tiny" / "history.csv")
    assert np.isfinite(h[["train_loss", "val_loss"]].to_numpy()).all()
    for f in ("best.pt", "last.pt", "history.csv", "config.json"):
        assert (out / "tiny" / f).exists()


def test_checkpoint_reload_and_reproducible_inference(tiny_run, prepared):
    cfg, summ, out = tiny_run
    df, cache = prepared
    labels = np.zeros(len(cache), int)
    labels[df.cache_idx.values] = df.grade.values
    idx = df[df.split == "test"].cache_idx.values
    r = verify_checkpoint_roundtrip(out / "tiny" / "best.pt", cache, labels, idx, cfg)
    assert r["identical_within_tol"] and r["finite"]
    model, ck = load_checkpoint(out / "tiny" / "best.pt")
    assert ck["cfg"]["arch"] == "small_cnn" and "history" in ck


def test_resume_continues_from_last_epoch(prepared, tmp_path):
    df, cache = prepared
    cfg = TrainConfig(arch="small_cnn", img_size=64, epochs=1, batch_size=16, num_workers=0, pretrained=False, out_dir=str(tmp_path), run_name="r", amp=False)
    train(cfg, cache, df, "cpu", log=lambda *a: None)
    cfg2 = TrainConfig(**{**cfg.to_dict(), "epochs": 2})
    s = train(cfg2, cache, df, "cpu", log=lambda *a: None)
    assert s["epochs_completed"] == 2  # epoch 1 restored from last.pt, epoch 2 trained


def test_corrupt_or_missing_checkpoint_gives_clear_error(tmp_path):
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a checkpoint")
    with pytest.raises(RuntimeError, match="failed to load checkpoint"):
        load_checkpoint(bad)
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt")
