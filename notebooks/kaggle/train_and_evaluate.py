"""Kaggle entry point: trains and evaluates the DR screening model remotely (Save & Run All / `kaggle kernels push`).

Runs on Kaggle's servers: it needs neither your laptop nor an SSH session. `scripts/kaggle_submit.py build` prepends two
generated lines to this file (an embedded copy of src/drscreen and the run configuration) so the kernel is one
self-contained file with no extra Kaggle Dataset to upload.

Can also run locally:   python notebooks/kaggle/train_and_evaluate.py --data-root DIR --work-dir OUT --smoke
"""
import argparse
import base64
import io
import json
import os
import sys
import zipfile
from pathlib import Path

try:  # injected by the build step
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


def discover_inputs(root="/kaggle/input"):
    """Return (aptos_root, external_root, listing). APTOS = the input holding an id_code/diagnosis table with >1000 rows;
    external = another input holding a labelled table with a DR-grade column (e.g. IDRiD)."""
    import pandas as pd
    from drscreen.datasets import find_labelled_tables

    listing, aptos, external = {}, None, None
    base = Path(root)
    if not base.exists():
        return None, None, listing
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        n = 0
        for p, idc, lc in find_labelled_tables(d):
            try:
                n += len(pd.read_csv(p, usecols=[idc]))
            except Exception:
                pass
        listing[d.name] = {"labelled_rows": n}
        if n >= 1000 and aptos is None:
            aptos = str(d)
        elif n >= 20 and external is None and "aptos" not in d.name.lower():
            external = str(d)
    return aptos, external, listing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=_RUN_CONFIG.get("data_root"))
    ap.add_argument("--external-root", default=_RUN_CONFIG.get("external_root"))
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--smoke", action="store_true", default=bool(_RUN_CONFIG.get("smoke", False)))
    ap.add_argument("--config-json", default=None, help="JSON string overriding RunConfig fields")
    args, _ = ap.parse_known_args()  # Kaggle injects extra argv; ignore unknown
    bootstrap_source()

    from drscreen.pipeline import Run, RunConfig, log

    on_kaggle = Path("/kaggle/working").exists()
    cfg = dict(_RUN_CONFIG)
    if args.config_json:
        cfg.update(json.loads(args.config_json))
    cfg["work_dir"] = args.work_dir or ("/kaggle/working" if on_kaggle else "outputs")
    cfg["smoke"] = args.smoke
    cfg["num_workers"] = min(int(cfg.get("num_workers", 4)), os.cpu_count() or 2)
    cfg["cache_dir"] = cfg.get("cache_dir") or ("/tmp/drcache" if on_kaggle else str(Path(cfg["work_dir"]) / "_cache"))
    aptos, ext, listing = discover_inputs() if on_kaggle else (None, None, {})
    cfg["data_root"] = args.data_root or aptos
    cfg["external_root"] = args.external_root or ext or cfg.get("external_root")
    log("inputs discovered:", json.dumps(listing))
    if not cfg["data_root"]:
        raise SystemExit("No APTOS data found. Attach the competition 'aptos2019-blindness-detection' (accept its rules on kaggle.com first) "
                         "or a dataset containing train.csv with id_code/diagnosis columns, or pass --data-root.")
    if on_kaggle and cfg.get("resume_search_root") is None:
        cfg["resume_search_root"] = "/kaggle/input"
    if cfg.get("pretrained", True) and not cfg.get("pretrained_file"):
        import urllib.request

        try:
            urllib.request.urlopen("https://huggingface.co", timeout=8)
        except Exception as e:
            local = [p for p in Path("/kaggle/input").rglob("*") if on_kaggle and p.is_file() and "efficientnet_b0" in p.name.lower()
                     and p.suffix in (".pth", ".safetensors", ".bin")]
            if local:
                cfg["pretrained_file"] = str(local[0])
                log("no internet; using local pretrained weights:", cfg["pretrained_file"])
            else:
                raise SystemExit(f"Cannot download ImageNet weights (no internet: {e}) and none attached. Enable 'Internet' in the notebook settings "
                                 "(requires a phone-verified Kaggle account). Refusing to train from random weights while claiming transfer learning.")
    rc = RunConfig(**{k: v for k, v in cfg.items() if k in RunConfig.__dataclass_fields__})
    log("RunConfig:", json.dumps(rc.__dict__, default=str))
    Run(rc).run_all()
    Path(rc.work_dir, "results", "DONE.json").write_text(json.dumps({"finished": True}))


if __name__ == "__main__":
    main()
