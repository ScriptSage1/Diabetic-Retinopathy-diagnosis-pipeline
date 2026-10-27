"""Regenerate report/Final_Report.md and report/Final_Report.pdf from the template + results/*.json + report/figures.

    python scripts/build_report.py

Edit report/meta.json to set author/institution (created with placeholders on first run).
Unresolved placeholders (experiments that did not run) are listed in report/missing_values.txt.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drscreen.report import build_report  # noqa: E402

DEFAULT_META = {
    "title": "Diabetic Retinopathy Screening from Fundus Photographs: A Transfer-Learning Research Prototype",
    "subtitle": "Referable-DR screening and five-grade ICDR classification with transparent clinical evaluation",
    "author": "[Author name: edit report/meta.json]",
    "institution": "[Institution / course: edit report/meta.json]",
    "date": "",
}


def main():
    meta_path = ROOT / "report" / "meta.json"
    if not meta_path.exists():
        meta_path.write_text(json.dumps(DEFAULT_META, indent=2))
    meta = {**DEFAULT_META, **json.loads(meta_path.read_text())}
    if not meta.get("date"):
        import time

        meta["date"] = time.strftime("%d %B %Y")
    stats = build_report(ROOT / "report" / "Final_Report.template.md", ROOT / "results", ROOT / "report" / "Final_Report.md",
                         ROOT / "report" / "Final_Report.pdf", meta)
    print(json.dumps(stats, indent=2))
    if stats["missing_placeholders"]:
        print("NOTE: some values/figures were unavailable; see report/missing_values.txt")


if __name__ == "__main__":
    main()
