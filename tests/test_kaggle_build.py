import importlib.util
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_submit():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("kaggle_submit", ROOT / "scripts" / "kaggle_submit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_built_kernel_script_header_executes_and_embeds_source(tmp_path, monkeypatch):
    ks = _load_submit()
    for mode in ("smoke", "full", "robust", "b3", "probe-eyepacs", "eyepacs", "eyepacs-smoke", "infer", "infer-smoke", "lesion2", "lesion2-smoke", "lesion-smoke", "lesion", "idrid-cnn", "idrid-features", "features"):
        out = ks.build(mode, "someuser", build_root=tmp_path)
        script = ks.KERNELS[mode]["script"]
        code = (out / script).read_text(encoding="utf-8")
        ns = {}
        header = "\n".join(code.splitlines()[:2])
        exec(header, ns)  # the injected lines must be valid Python (regression: JSON 'true' broke the first Kaggle run)
        assert isinstance(ns["_RUN_CONFIG"], dict) and ns["_RUN_CONFIG"]["smoke"] is mode.endswith("smoke")
        import base64, io
        names = zipfile.ZipFile(io.BytesIO(base64.b64decode(ns["_EMBEDDED_SRC_B64"]))).namelist()
        assert "drscreen/pipeline.py" in names and "drscreen/lesion_pipeline.py" in names and "drscreen/__init__.py" in names
        import json
        meta = json.loads((out / "kernel-metadata.json").read_text())
        assert meta["competition_sources"][0] == "aptos2019-blindness-detection" and (len(meta["competition_sources"]) == 1 or mode.startswith(("probe-eyepacs", "eyepacs"))) and meta["enable_gpu"] is True
        if mode == "idrid-cnn":
            assert meta["kernel_sources"] == ["someuser/dr-screening-effnet-b0-aptos"]  # classifier output (checkpoint)
        if mode == "lesion":
            assert meta["kernel_sources"] == []  # independent of the classifier run
        slug = "".join(c if c.isalnum() else "-" for c in meta["title"].lower()).strip("-")
        assert meta["id"].split("/")[1] == slug, (meta["id"], slug)  # Kaggle derives the slug from the title
