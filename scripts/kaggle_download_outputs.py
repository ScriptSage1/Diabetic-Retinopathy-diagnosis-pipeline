"""Download the finished kernel's outputs and install them into the repo, then VERIFY they load.

    python scripts/kaggle_download_outputs.py [--ref user/slug] [--from-dir already_downloaded_dir]

Downloads to a cache dir outside OneDrive, then copies (without last.pt):  results/*  -> results/      figures/* -> report/figures/      artifacts/efficientnet_b0/* -> artifacts/efficientnet_b0/
Verification: the checkpoint is loaded, a forward pass on a synthetic image must be finite, and calibration.json is parsed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _kaggle_common import CACHE_DIR, ROOT, kernel_ref, redact, run_kaggle, save_state  # noqa: E402


def install(src: Path) -> list[str]:
    done = []
    for sub, dst in (("results", ROOT / "results"), ("figures", ROOT / "report" / "figures"), ("artifacts", ROOT / "artifacts")):
        s = src / sub
        if s.is_dir():
            # last.pt (optimizer state, ~3x larger) stays in the cache dir; only what the app/report need enters the synced project
            shutil.copytree(s, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("last.pt", "*.tmp"))
            done.append(f"{sub} -> {dst.relative_to(ROOT)}")
    return done


def verify(ckpt: Path) -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np

    from drscreen.predict import Predictor
    from drscreen.synthetic import make_fundus

    p = Predictor(ckpt)
    r = p.predict(make_fundus(2, 400, 1, border=False), with_gradcam=True, force=True)
    return {"checkpoint": str(ckpt.relative_to(ROOT)), "loads": True, "forward_ok": r.status == "ok" and bool(np.isfinite(r.grade_probs).all()),
            "calibrated": p.calibrated, "temperature": p.temperature, "threshold": p.threshold, "arch": p.cfg["arch"], "img_size": p.img_size}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=None)
    ap.add_argument("--from-dir", default=None)
    a = ap.parse_args()
    if a.from_dir:
        src = Path(a.from_dir)
    else:
        ref = a.ref or kernel_ref()
        if not ref:
            print("No kernel reference known.")
            sys.exit(3)
        src = CACHE_DIR
        src.mkdir(parents=True, exist_ok=True)
        r = run_kaggle(["kernels", "output", ref, "-p", str(src)], timeout=1800)
        print(redact((r.stdout + r.stderr).strip())[-800:])
        if r.returncode != 0:
            sys.exit(r.returncode)
    done = install(src)
    print("installed:", done or "NOTHING (expected results/, figures/, artifacts/ in the output)")
    ck = ROOT / "artifacts" / "efficientnet_b0" / "best.pt"
    if not ck.is_file():
        print("no best.pt found in the output: training did not produce a checkpoint")
        sys.exit(5)
    info = verify(ck)
    print("verification:", json.dumps(info))
    save_state(outputs_downloaded=True, verified=info)
    sys.exit(0 if info["forward_ok"] else 6)


if __name__ == "__main__":
    main()
