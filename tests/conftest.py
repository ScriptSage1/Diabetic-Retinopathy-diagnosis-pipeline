import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def synth_root(tmp_path_factory):
    from drscreen.synthetic import make_synthetic_dataset

    root = tmp_path_factory.mktemp("synth")
    make_synthetic_dataset(root, n_per_grade=(30, 12, 24, 12, 12), size=160)
    return root


@pytest.fixture(scope="session")
def prepared(synth_root, tmp_path_factory):
    """Synthetic dataset -> labelled table -> cache -> splits (64px, fast)."""
    from drscreen import datasets as D

    df = D.load_labelled_dataset(synth_root, "synthetic")
    cache_dir = tmp_path_factory.mktemp("cache")
    df_ok, path = D.build_cache(df, 64, cache_dir, progress=False)
    df_ok = D.make_splits(df_ok, seed=0)
    return df_ok, np.load(path, mmap_mode="r")


@pytest.fixture(scope="session")
def tiny_run(prepared, tmp_path_factory):
    """Train a small CNN for 2 epochs on synthetic data and return (cfg, summary, out_dir)."""
    from drscreen.train import TrainConfig, train

    df, cache = prepared
    out = tmp_path_factory.mktemp("art")
    cfg = TrainConfig(arch="small_cnn", img_size=64, epochs=2, batch_size=16, num_workers=0, pretrained=False,
                      out_dir=str(out), run_name="tiny", amp=False)
    summ = train(cfg, cache, df, "cpu", log=lambda *a: None)
    return cfg, summ, out
