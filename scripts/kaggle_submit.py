"""Build and submit the Kaggle kernels.

    python scripts/kaggle_submit.py check                     # CLI present? authenticated? GPU quota (no secrets printed)
    python scripts/kaggle_submit.py build --mode full         # writes notebooks/kaggle/build/<mode>/{script,kernel-metadata.json}
    python scripts/kaggle_submit.py push  --mode smoke        # build + `kaggle kernels push` (runs remotely; you can switch your laptop off)

Modes:
  smoke         tiny classifier run (2 epochs, 240 images): validates data path / weights / outputs        (~6 min)
  full          the real classifier run: EfficientNet-B0, 384 px, up to 30 epochs
  robust        the full classifier run again with blur/JPEG/illumination training augmentation (compared with `full`)
  b3            EfficientNet-B3 at 448 px with robustness augmentation (compared with `robust` and `full`)
  infer         logits of every classifier checkpoint (+flip TTA) on val/test/IDRiD, for ensemble selection on CPU
  lesion-smoke  tiny lesion-evidence run (IDRiD U-Net + locator + feature extraction on a few images)
  lesion        the real lesion-evidence run (independent of the classifier run; APTOS splits are joined locally afterwards)
  features      re-extract ALL lesion features (APTOS + IDRiD, improved disc/fovea localisation) with the trained lesion models
  idrid-features re-extract IDRiD lesion features (keyed by split) with the trained lesion models + classifier probabilities
  lesion2       lesion model v2: IDRiD + e-ophtha, partial labels, acquisition augmentation, large-region filter
  idrid-cnn     tiny follow-up: classifier probabilities on IDRiD grading test images (needs the finished `full` kernel attached)
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _kaggle_common import (BUILD_DIR, COMPETITION, KERNEL_SLUG, ROOT, credentials_present, get_username, load_state, redact, run_kaggle,  # noqa: E402
                            save_state)

IDRID_DATASET = "aaryapatel98/indian-diabetic-retinopathy-image-dataset"
EYEPACS_COMPETITION = "diabetic-retinopathy-detection"  # official 2015 EyePACS data; needs the competition rules accepted by the account holder
EOPHTHA_DATASET = "samriddhibagchi/e-ophtha-diabetic-retinopathy-datasets-ex-ma"  # MIT-declared copy of e-ophtha (EX + MA)

CONFIGS = {
    "smoke": {"smoke": True, "epochs": 2, "img_size": 224, "batch_size": 16, "baseline_epochs": 1, "baseline_img_size": 128, "n_boot": 100,
              "quality_eval_n": 30, "max_minutes": 25},
    # max_minutes keeps the whole run safely inside Kaggle's 12 h session limit; training stops cleanly between epochs
    "full": {"smoke": False, "epochs": 30, "img_size": 384, "batch_size": 24, "lr": 3e-4, "baseline_epochs": 12, "baseline_img_size": 224, "n_boot": 1000,
             "quality_eval_n": 150, "max_minutes": 300, "expected_n_labelled": 3662},
    # same data split, architecture and schedule as `full`; only the training augmentation differs (blur / JPEG / illumination)
    "robust": {"smoke": False, "epochs": 30, "img_size": 384, "batch_size": 24, "lr": 3e-4, "n_boot": 1000, "quality_eval_n": 150, "max_minutes": 300,
               "expected_n_labelled": 3662, "robust_aug": True, "run_name": "efficientnet_b0_robust", "run_baselines": False, "run_gradcam": False},
    # larger backbone / resolution, same split, same robustness augmentation; compared on VALIDATION first (see scripts/compare_models.py)
    "b3": {"smoke": False, "arch": "efficientnet_b3", "epochs": 30, "img_size": 448, "batch_size": 16, "lr": 3e-4, "n_boot": 1000, "quality_eval_n": 150,
           "max_minutes": 300, "expected_n_labelled": 3662, "robust_aug": True, "run_name": "efficientnet_b3_robust", "run_baselines": False, "run_gradcam": False},
    "probe-eyepacs": {"smoke": False},
    "eyepacs-smoke": {"smoke": True, "img_size": 224, "batch_size": 8, "max_minutes": 25},
    # APTOS split identical to the other classifiers; phase A = APTOS + EyePACS, phase B = APTOS-only fine-tune (see notebooks/kaggle/train_eyepacs.py)
    "eyepacs": {"smoke": False, "img_size": 384, "batch_size": 24, "epochs_a": 8, "epochs_b": 5, "n_grade0": 5000, "n_test": 3000, "max_minutes": 540},
    "infer": {"smoke": False},
    "infer-smoke": {"smoke": True},
    "lesion-smoke": {"smoke": True, "iters": 20, "loc_epochs": 1, "max_minutes": 25},
    "lesion": {"smoke": False, "iters": 4000, "batch": 8, "patch": 512, "loc_epochs": 40, "max_minutes": 300},
    "lesion2": {"smoke": False, "v2": True, "iters": 6000, "batch": 8, "patch": 512, "loc_epochs": 40, "max_minutes": 330},
    "lesion2-smoke": {"smoke": True, "v2": True, "iters": 20, "loc_epochs": 1, "max_minutes": 25},
    "idrid-cnn": {"only_cnn": True, "smoke": False},
    "idrid-features": {"only_idrid_features": True, "smoke": False, "num_workers": 4},
    "features": {"only_features": True, "smoke": False, "num_workers": 4},
}

# kernel definitions: script, slug, title (slug must equal Kaggle's slug of the title), inputs, timeout seconds
KERNELS = {
    "smoke": dict(script="train_and_evaluate.py", slug=KERNEL_SLUG + "-smoke", title="DR Screening EffNet B0 APTOS Smoke", timeout=2400, kernel_sources=[], datasets=[]),
    "full": dict(script="train_and_evaluate.py", slug=KERNEL_SLUG, title="DR Screening EffNet B0 APTOS", timeout=39000, kernel_sources=[], datasets=[]),
    "robust": dict(script="train_and_evaluate.py", slug=KERNEL_SLUG + "-robust", title="DR Screening EffNet B0 APTOS Robust", timeout=39000, kernel_sources=[], datasets=[]),
    "b3": dict(script="train_and_evaluate.py", slug="dr-screening-effnet-b3-aptos", title="DR Screening EffNet B3 APTOS", timeout=39000, kernel_sources=[], datasets=[]),
    "probe-eyepacs": dict(script="probe_eyepacs.py", slug="dr-eyepacs-probe", title="DR EyePACS Probe", timeout=3000, kernel_sources=[], datasets=[], extra_competitions=[EYEPACS_COMPETITION]),
    "eyepacs-smoke": dict(script="train_eyepacs.py", slug="dr-eyepacs-experiment-smoke", title="DR EyePACS Experiment Smoke", timeout=3000, kernel_sources=["{user}/" + KERNEL_SLUG + "-robust"],
                          datasets=[IDRID_DATASET], extra_competitions=[EYEPACS_COMPETITION]),
    "eyepacs": dict(script="train_eyepacs.py", slug="dr-eyepacs-experiment", title="DR EyePACS Experiment", timeout=39000, kernel_sources=["{user}/" + KERNEL_SLUG + "-robust"],
                    datasets=[IDRID_DATASET], extra_competitions=[EYEPACS_COMPETITION]),
    "infer": dict(script="infer_models.py", slug="dr-classifier-logits", title="DR Classifier Logits", timeout=7200,
                  kernel_sources=["{user}/" + KERNEL_SLUG, "{user}/" + KERNEL_SLUG + "-robust", "{user}/dr-screening-effnet-b3-aptos"], datasets=[IDRID_DATASET]),
    "infer-smoke": dict(script="infer_models.py", slug="dr-classifier-logits-smoke", title="DR Classifier Logits Smoke", timeout=3000,
                        kernel_sources=["{user}/" + KERNEL_SLUG, "{user}/" + KERNEL_SLUG + "-robust"], datasets=[IDRID_DATASET]),
    "lesion-smoke": dict(script="train_lesions.py", slug="dr-lesion-evidence-smoke", title="DR Lesion Evidence Smoke", timeout=2400,
                         kernel_sources=[], datasets=[IDRID_DATASET]),
    "lesion": dict(script="train_lesions.py", slug="dr-lesion-evidence", title="DR Lesion Evidence", timeout=39000,
                   kernel_sources=[], datasets=[IDRID_DATASET]),  # independent of the classifier run; splits are joined locally
    "idrid-features": dict(script="train_lesions.py", slug="dr-idrid-lesion-features", title="DR IDRiD Lesion Features", timeout=7200,
                           kernel_sources=["{user}/dr-lesion-evidence", "{user}/" + KERNEL_SLUG], datasets=[IDRID_DATASET]),  # needs the finished lesion + classifier kernels
    "features": dict(script="train_lesions.py", slug="dr-lesion-features-v2", title="DR Lesion Features V2", timeout=14400,
                     kernel_sources=["{user}/dr-lesion-evidence", "{user}/" + KERNEL_SLUG], datasets=[IDRID_DATASET]),  # APTOS + IDRiD features with the improved disc/fovea localisation
    "lesion2": dict(script="train_lesions.py", slug="dr-lesion-evidence-v2", title="DR Lesion Evidence V2", timeout=39000, kernel_sources=[], datasets=[IDRID_DATASET, EOPHTHA_DATASET]),
    "lesion2-smoke": dict(script="train_lesions.py", slug="dr-lesion-evidence-v2-smoke", title="DR Lesion Evidence V2 Smoke", timeout=3000, kernel_sources=[], datasets=[IDRID_DATASET, EOPHTHA_DATASET]),
    "idrid-cnn": dict(script="train_lesions.py", slug="dr-idrid-classifier-probs", title="DR IDRiD Classifier Probs", timeout=3600,
                      kernel_sources=["{user}/" + KERNEL_SLUG], datasets=[IDRID_DATASET]),  # needs the finished classifier kernel
}


def embed_source() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((ROOT / "src" / "drscreen").rglob("*.py")):
            z.write(p, p.relative_to(ROOT / "src").as_posix())
    return base64.b64encode(buf.getvalue()).decode()


def build(mode: str, username: str | None, extra: dict | None = None, build_root: Path | None = None, gpu: bool = True) -> Path:
    k = KERNELS[mode]
    out = (build_root or BUILD_DIR) / mode
    out.mkdir(parents=True, exist_ok=True)
    cfg = {**CONFIGS[mode], **(extra or {})}
    body = (ROOT / "notebooks" / "kaggle" / k["script"]).read_text(encoding="utf-8")
    header = f"_EMBEDDED_SRC_B64 = {embed_source()!r}\n_RUN_CONFIG = {cfg!r}\n"
    (out / k["script"]).write_text(header + body, encoding="utf-8")
    user = username or "YOUR_KAGGLE_USERNAME"
    meta = {
        "id": f"{user}/{k['slug']}", "title": k["title"], "code_file": k["script"], "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
        "enable_internet": True,  # needed to download ImageNet weights; requires a phone-verified account
        "dataset_sources": k["datasets"], "competition_sources": [COMPETITION, *k.get("extra_competitions", [])],
        "kernel_sources": [s.format(user=user) for s in k["kernel_sources"]], "model_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    compile((out / k["script"]).read_text(encoding="utf-8"), k["script"], "exec")  # the generated script must at least parse
    return out


def is_authenticated() -> tuple[bool, str]:
    """Authenticated iff a harmless API call (`kaggle quota`) succeeds. Output is redacted."""
    r = run_kaggle(["quota"], timeout=120)
    text = redact((r.stdout + r.stderr).strip())
    if r.returncode != 0 or "authentication required" in text.lower():
        return False, text
    return True, text


def cmd_check(_):
    print("Kaggle CLI:", end=" ")
    try:
        r = run_kaggle(["--version"], timeout=60)
        print((r.stdout or r.stderr).strip())
    except Exception as e:
        print(f"NOT AVAILABLE ({e})")
        return 2
    print("Legacy credential sources found:", json.dumps(credentials_present()["sources"]))
    ok, text = is_authenticated()
    if not ok:
        print("NOT AUTHENTICATED. Run once in a terminal:  kaggle auth login   (opens a browser; approve the sign-in. No token needs to be pasted anywhere.)")
        return 3
    print("API call: OK", text[:600], sep="\n")
    print("Username:", get_username() or "unknown (pass your username via KAGGLE_USERNAME if needed)")
    return 0


def cmd_build(a):
    print(f"built {a.mode} kernel in {build(a.mode, get_username())}")
    return 0


def cmd_push(a):
    user = get_username()
    if not user:
        print("Cannot determine the Kaggle username (set KAGGLE_USERNAME=<your username>, or authenticate; see `check`).")
        return 3
    ok, _ = is_authenticated()
    if not ok:
        print("Not authenticated: run `kaggle auth login` first.")
        return 3
    d = build(a.mode, user, gpu=not a.cpu)
    acc = [] if a.cpu else ["--accelerator", a.accelerator]
    r = run_kaggle(["kernels", "push", "-p", str(d), *acc, "-t", str(KERNELS[a.mode]["timeout"])], timeout=600)
    text = redact((r.stdout + "\n" + r.stderr).strip())
    print(text)
    # the CLI can exit 0 while printing e.g. "Kernel push error: Maximum batch GPU session count of 2 reached."
    if r.returncode != 0 or "push error" in text.lower() or "successfully pushed" not in text.lower():
        print("PUSH FAILED. Common causes: competition rules not accepted, unverified phone (needed for GPU/internet), unavailable dataset/kernel source, or a slug that does not match the title.")
        return r.returncode or 1
    ref = json.loads((d / "kernel-metadata.json").read_text())["id"]
    kernels = {**load_state().get("kernels", {}), a.mode: ref}
    save_state(kernels=kernels, kernel_ref=ref, mode=a.mode, pushed_at=time.strftime("%Y-%m-%d %H:%M:%S"), last_status="submitted")
    print(f"Submitted {ref}. Track with: python scripts/kaggle_status.py --mode {a.mode} --watch")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    for name, fn in (("build", cmd_build), ("push", cmd_push)):
        p = sub.add_parser(name)
        p.add_argument("--mode", choices=list(KERNELS), default="full")
        p.add_argument("--cpu", action="store_true", help="CPU-only session (does not use a GPU slot; for smoke tests of code/data paths)")
        p.add_argument("--accelerator", default="NvidiaTeslaT4", help="e.g. NvidiaTeslaT4 (T4 x2); see `kaggle kernels push --help`")
        p.set_defaults(fn=fn)
    a = ap.parse_args()
    sys.exit(a.fn(a))


if __name__ == "__main__":
    main()
