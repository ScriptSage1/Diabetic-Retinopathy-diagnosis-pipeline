"""One-to-two page PDF case report for a single analysed image (verdict, reasoning, images, ICDR checklist, limitations)."""
from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from . import INTENDED_USE
from .explain import PALETTE, confidence, narrative, verdict
from .metrics import GRADE_NAMES
from .report import _inline, _styles, _table


def _img(arr: np.ndarray, width: float, max_px: int = 900):
    im = PILImage.fromarray(np.asarray(arr)).convert("RGB")
    if im.width > max_px:
        im = im.resize((max_px, int(im.height * max_px / im.width)), PILImage.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    buf.seek(0)
    return Image(buf, width=width, height=width * im.height / im.width)


def build_case_pdf(r, path: str | Path, title: str = "Diabetic retinopathy screening: case report") -> Path:
    """Write the case report for PredictionResult `r` to `path` and return the path. Works for ok / quality_rejected / error results."""
    path = Path(path)
    st = _styles()
    width = A4[0] - 4 * cm
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.6 * cm, title=title)
    v = verdict(r)
    accent, soft = PALETTE[v["kind"]]
    story = [Paragraph(_inline(f"**{title}**"), st["h2"]),
             Paragraph(_inline(f"Research prototype · generated {time.strftime('%Y-%m-%d %H:%M')} · model: {r.model_name or 'n/a'}"), st["caption"])]
    box = Table([[Paragraph(_inline(v["title"]), st["h1"]), Paragraph(_inline(v["subtitle"]), st["body"])]], colWidths=[width * 0.42, width * 0.58])
    box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(soft)), ("LINEBEFORE", (0, 0), (0, -1), 5, colors.HexColor(accent)),
                             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 12), ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story += [box, Spacer(1, 8)]
    if r.status == "ok":
        conf, conf_note = confidence(r)
        facts = [["Item", "Value"],
                 ["P(referable DR)", f"{r.p_referable:.1%} (operating threshold {r.threshold:.1%})"],
                 ["Decision", (r.decision or {}).get("action", "").replace("_", " ")],
                 ["Most likely grade", f"{r.grade_name} ({r.grade_probs[r.grade]:.1%})"],
                 ["Grade probabilities", " · ".join(f"{GRADE_NAMES[i].split(' ', 1)[0]}: {p:.0%}" for i, p in enumerate(r.grade_probs))],
                 ["Confidence (heuristic)", f"{conf}. {conf_note}"],
                 ["Image quality", f"{(r.quality or {}).get('status', '?')}: {(r.quality or {}).get('reason', '')}"]]
        story += [_table(facts, st, width), Spacer(1, 8)]
    panels = [(r.original, "Original"), (r.processed, "Pre-processed"), (r.gradcam, "Grad-CAM"), (r.lesion_overlay, "Detected lesions")]
    panels = [(a, t) for a, t in panels if a is not None]
    if panels:
        w = width / len(panels) - 4
        cells = [[_img(a, w) for a, _ in panels], [Paragraph(_inline(t), st["caption"]) for _, t in panels]]
        tb = Table(cells, colWidths=[width / len(panels)] * len(panels))
        tb.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
        story += [tb, Spacer(1, 4)]
    story.append(Paragraph("Why this result", st["h2"]))
    for line in narrative(r):
        story.append(Paragraph(_inline(line), st["bullet"], bulletText="•"))
    ev = r.lesion_evidence or {}
    if r.status == "ok" and ev.get("available"):
        rows = [["ICDR criterion", "Status", "Detail"]] + [[c["name"], c["status"].replace("_", " "), c["detail"]] for c in ev["rule"]["criteria"]]
        story += [Spacer(1, 6), KeepTogether([Paragraph(f"ICDR criteria checklist (rule-derived grade {ev['rule']['grade']})", st["h3"]), _table(rows, st, width)])]
        if ev.get("location_note"):
            story.append(Paragraph(_inline("_Localisation: " + ev["location_note"] + "_"), st["caption"]))
        story.append(Paragraph(_inline("_" + ev["caveat"] + "_"), st["caption"]))
    story += [Spacer(1, 8), Paragraph(_inline("**Intended use.** " + INTENDED_USE), st["quote"])]
    doc.build(story)
    return path
