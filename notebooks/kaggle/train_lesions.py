"""Kaggle entry point for the lesion-evidence run (IDRiD lesion U-Net + OD/fovea locator + lesion features for APTOS/IDRiD).

Inputs it expects under /kaggle/input (attached via kernel-metadata.json):
  * the APTOS 2019 competition data,
  * the IDRiD dataset (folders 'A. Segmentation', 'B. Disease Grading', 'C. Localization'),
  * (full run) the output of the classifier kernel: artifacts/efficientnet_b0/best.pt and artifacts/split_manifest.csv.
`scripts/kaggle_submit.py build --mode lesion` prepends the embedded source and run configuration.
"""
import argparse
import base64
import io
import json
import os
import sys
import zipfile
from pathlib import Path

try:
    _EMBEDDED_SRC_B64
except NameError:
    _EMBEDDED_SRC_B64 = None
try:
    _RUN_CONFIG
except NameError:
    _RUN_CONFIG = {}


def bootstrap_source():
    if _EMBEDDED_SRC_B64:
        dest = Path("/tmp/drsrc")
        dest.mkdir(parents=True, exist_ok=True)
        zipfile.ZipFile(io.BytesIO(base64.b64decode(_EMBEDDED_SRC_B64))).extractall(dest)
        sys.path.insert(0, str(dest))
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def find_inputs(root="/kaggle/input"):
    """Locate inputs by content, not by mount path (Kaggle mounts competitions/datasets/kernel outputs under nested folders)."""
    r = {"aptos": None, "idrid": None, "cnn": None, "manifest": None, "lesion_models": None, "eophtha": None}
    base = Path(root)
    if not base.exists():
        return r
    for p in base.rglob("train.csv"):
        if "aptos" in str(p).lower():
            r["aptos"] = str(p.parent)
            break
    for p in base.rglob("*egmentation*"):  # folders may be literally named 'A.%20Segmentation' on the mount
        if p.is_dir():
            anc = p
            for cand in [p, *p.parents]:  # climb to the ancestor that also contains the localization / grading parts
                if cand == base.parent:
                    break
                try:
                    names = " ".join(x.name.lower() for x in cand.iterdir())
                except OSError:
                    continue
                if "localization" in names and "grading" in names:
                    anc = cand
                    break
            r["idrid"] = str(anc)
            break
    for p in base.rglob("best.pt"):
        if p.parent.name == "efficientnet_b0":
            r["cnn"] = str(p)
            break
    for p in base.rglob("e_ophtha_EX"):
        r["eophtha"] = str(p.parent)
        break
    for p in base.rglob("lesion_config.json"):
        r["lesion_models"] = str(p.parent)
        break
    for p in base.rglob("split_manifest.csv"):
        r["manifest"] = str(p)
        break
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idrid-root", default=None)
    ap.add_argument("--aptos-root", default=None)
    ap.add_argument("--cnn-checkpoint", default=None)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--smoke", action="store_true", default=bool(_RUN_CONFIG.get("smoke", False)))
    args, _ = ap.parse_known_args()
    bootstrap_source()
    from drscreen.lesion_pipeline import LesionRun, LesionRunConfig, log

    on_kaggle = Path("/kaggle/working").exists()
    found = find_inputs() if on_kaggle else {}
    log("inputs found:", json.dumps(found))
    cfg = dict(_RUN_CONFIG)
    cfg["work_dir"] = args.work_dir or ("/kaggle/working" if on_kaggle else "outputs_lesion")
    cfg["smoke"] = args.smoke
    cfg["idrid_root"] = args.idrid_root or found.get("idrid")
    cfg["aptos_root"] = args.aptos_root or found.get("aptos")
    cfg["cnn_checkpoint"] = args.cnn_checkpoint or found.get("cnn")
    cfg["eophtha_root"] = found.get("eophtha") if cfg.get("v2") else None
    cfg["split_manifest"] = found.get("manifest")
    cfg["num_workers"] = min(int(cfg.get("num_workers", 4)), os.cpu_count() or 2)
    if cfg.get("only_features"):
        cfg["lesion_model_dir"] = found.get("lesion_models")
        if not cfg["idrid_root"] or not cfg["aptos_root"] or not cfg["lesion_model_dir"]:
            raise SystemExit(f"only_features mode needs IDRiD + APTOS + the lesion kernel output: {found}")
        from drscreen.lesion_pipeline import LesionRun, LesionRunConfig

        rc = LesionRunConfig(**{k: v for k, v in cfg.items() if k in LesionRunConfig.__dataclass_fields__})
        LesionRun(rc).run_only_features()
        Path(rc.work_dir, "results", "DONE_features.json").write_text(json.dumps({"finished": True}))
        return
    if cfg.get("only_idrid_features"):
        cfg["lesion_model_dir"] = found.get("lesion_models")
        if not cfg["idrid_root"] or not cfg["lesion_model_dir"]:
            raise SystemExit(f"only_idrid_features mode needs IDRiD + the lesion kernel output: {found}")
        from drscreen.lesion_pipeline import LesionRun, LesionRunConfig

        rc = LesionRunConfig(**{k: v for k, v in cfg.items() if k in LesionRunConfig.__dataclass_fields__})
        LesionRun(rc).run_only_idrid_features()
        Path(rc.work_dir, "results", "DONE_idrid_features.json").write_text(json.dumps({"finished": True}))
        return
    if cfg.get("only_cnn"):
        if not cfg["idrid_root"] or not cfg["cnn_checkpoint"]:
            raise SystemExit(f"only_cnn mode needs IDRiD + the classifier kernel output (best.pt): {found}")
        from drscreen.lesion_pipeline import LesionRun, LesionRunConfig, log

        rc = LesionRunConfig(**{k: v for k, v in cfg.items() if k in LesionRunConfig.__dataclass_fields__})
        LesionRun(rc).run_only_cnn()
        Path(rc.work_dir, "results", "DONE_idrid_cnn.json").write_text(json.dumps({"finished": True}))
        return
    if not cfg["idrid_root"] or not cfg["aptos_root"]:
        raise SystemExit(f"Missing inputs (need IDRiD + APTOS): {found}. Attach the IDRiD dataset and the APTOS competition (rules accepted).")
    if cfg.get("pretrained", True):
        import urllib.request

        try:
            urllib.request.urlopen("https://huggingface.co", timeout=8)
        except Exception as e:
            raise SystemExit(f"Cannot download ImageNet weights (no internet: {e}). Enable Internet in the kernel settings.")
    rc = LesionRunConfig(**{k: v for k, v in cfg.items() if k in LesionRunConfig.__dataclass_fields__})
    log("LesionRunConfig:", json.dumps(rc.__dict__, default=str))
    LesionRun(rc).run_all()
    Path(rc.work_dir, "results", "DONE_lesion.json").write_text(json.dumps({"finished": True}))


if __name__ == "__main__":
    main()
