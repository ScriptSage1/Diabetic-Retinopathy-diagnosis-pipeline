"""Kaggle probe (CPU): how is the EyePACS competition data laid out in a notebook, and how fast can images be read?
Writes results/probe.json. Embedded source + config are prepended by scripts/kaggle_submit.py."""
import base64
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

try:
    _EMBEDDED_SRC_B64
except NameError:
    _EMBEDDED_SRC_B64 = None

if _EMBEDDED_SRC_B64:
    dest = Path("/tmp/drsrc")
    dest.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(base64.b64decode(_EMBEDDED_SRC_B64))).extractall(dest)
    sys.path.insert(0, str(dest))
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from drscreen import eyepacs as E  # noqa: E402

out = Path("/kaggle/working/results" if Path("/kaggle/working").exists() else "outputs_probe")
out.mkdir(parents=True, exist_ok=True)
root = Path("/kaggle/input")
info = {"top": sorted(p.name for p in root.iterdir())}
tree = []
for dp, dn, files in os.walk(root):
    depth = Path(dp).relative_to(root).parts
    if len(depth) <= 3 and "diabetic" in "/".join(depth).lower():
        tree.append({"dir": "/".join(depth), "n_files": len(files), "sample_files": sorted(files)[:8], "subdirs": sorted(dn)[:8]})
info["tree"] = tree
ep_root = next((p for p in root.rglob("*") if p.is_dir() and p.name == "diabetic-retinopathy-detection"), None)
info["eyepacs_root"] = str(ep_root)
found = E.find_eyepacs(ep_root or root)
info["found"] = {k: ([str(x) for x in v] if isinstance(v, list) else (str(v) if v else None)) for k, v in found.items()}
if found["labels"]:
    df = E.load_labels(found["labels"])
    info["labels"] = {"n": len(df), "patients": int(df.patient.nunique()), "grades": df.grade.value_counts().sort_index().to_dict()}
    rd = E.EyePacsReader(found)
    t = time.time()
    shapes, md5s = [], []
    for name in df.image.sample(20, random_state=0):
        img, m = rd.read(name)
        shapes.append(list(img.shape[:2]))
        md5s.append(m)
    info["read_20_images_seconds"] = round(time.time() - t, 2)
    info["sample_shapes"] = shapes[:8]
    info["n_zip_members"] = len(rd.members) if rd.zf is not None else None
    info["member_name_examples"] = list(rd.members.values())[:3] if rd.zf is not None else None
    info["train_images_in_labels_found_in_archive"] = int(df.image.isin(set(rd.members)).sum()) if rd.zf is not None else None
(out / "probe.json").write_text(json.dumps(info, indent=2))
print(json.dumps(info, indent=2))
