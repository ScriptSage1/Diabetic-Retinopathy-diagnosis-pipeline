"""Run the pytest suite and save a machine-readable summary for the report (results/test_summary.json).

    python scripts/test_summary.py

The report's testing section is generated from this file, so its counts are the real outcome of the last run, never typed in by hand.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    xml = Path(tempfile.mkdtemp()) / "junit.xml"
    t0 = time.time()
    r = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={xml}", str(ROOT / "tests")], cwd=ROOT,
                       env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"})
    seconds = time.time() - t0
    root = ET.parse(xml).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    per_file: "OrderedDict[str, dict]" = OrderedDict()
    for tc in suite.iter("testcase"):
        f = next((part for part in tc.get("classname", "").split(".") if part.startswith("test_")), tc.get("classname", ""))
        d = per_file.setdefault(f, {"file": f + ".py", "tests": 0, "passed": 0, "failed": 0, "skipped": 0})
        d["tests"] += 1
        if tc.find("failure") is not None or tc.find("error") is not None:
            d["failed"] += 1
        elif tc.find("skipped") is not None:
            d["skipped"] += 1
        else:
            d["passed"] += 1
    rows = sorted(per_file.values(), key=lambda d: d["file"])
    out = {"total": sum(d["tests"] for d in rows), "passed": sum(d["passed"] for d in rows), "failed": sum(d["failed"] for d in rows),
           "skipped": sum(d["skipped"] for d in rows), "seconds": round(seconds, 1), "python": platform.python_version(), "platform": platform.platform(),
           "per_file": rows, "run_date": time.strftime("%Y-%m-%d %H:%M")}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "test_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "per_file"}, indent=1))
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
