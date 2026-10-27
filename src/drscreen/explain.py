"""Turn a PredictionResult into a clear, honest explanation: verdict, confidence level, plain-language reasoning and HTML cards.

Everything shown is computed from the result object; nothing is invented. Wording is deliberately cautious: lesion detections are evidence for a
clinician to check, the confidence level is a heuristic (distance from the decision threshold), and nothing here is a medical diagnosis.
"""
from __future__ import annotations

import base64
import html
import io

import numpy as np
from PIL import Image

from . import INTENDED_USE
from .metrics import GRADE_NAMES

PALETTE = {  # (accent, soft background): muted clinical palette, see drscreen.ui_theme tokens
    "refer": ("#a4412c", "#f5e6e1"),
    "no_referral": ("#2d6847", "#e2eee6"),
    "human_review": ("#80590f", "#f4ead3"),
    "recapture": ("#4e5963", "#e9ecee"),
    "error": ("#4e5963", "#e9ecee"),
}
QUADRANT_LABELS = ["temporal-superior", "temporal-inferior", "nasal-superior", "nasal-inferior"]
FINDING_NAMES = {"MA": "microaneurysm", "HE": "haemorrhage", "EX": "hard exudate", "SE": "soft exudate (cotton-wool spot)"}


def esc(x) -> str:
    return html.escape(str(x), quote=True)


# --------------------------------------------------------------------------------------------- verdict / confidence
def _grade_phrase(name: str) -> str:
    """'2 Moderate NPDR' -> 'grade 2, Moderate NPDR'."""
    head, _, rest = (name or "?").partition(" ")
    return f"grade {head}, {rest}" if head.isdigit() and rest else name


def verdict(r) -> dict:
    """kind, title, subtitle for the headline card."""
    if r.status == "error":
        return {"kind": "error", "title": "Could not analyse this file", "subtitle": r.message}
    if r.status == "quality_rejected":
        return {"kind": "recapture", "title": "Please retake the photograph", "subtitle": (r.quality or {}).get("reason", "The image quality is too low for analysis.")}
    action = (r.decision or {}).get("action", "human_review")
    name = r.grade_name or "?"
    if action == "refer":
        return {"kind": "refer", "title": "Referral recommended", "subtitle": f"Referable diabetic retinopathy suspected · most likely {_grade_phrase(name)}"}
    if action == "no_referral":
        return {"kind": "no_referral", "title": "No referral indicated", "subtitle": f"Non-referable by this system · most likely {_grade_phrase(name)}"}
    reason = (r.decision or {}).get("reason", "The result is uncertain.")
    if reason.startswith("P(referable)"):
        reason = "The score is too close to the decision threshold to decide automatically."
    return {"kind": "human_review", "title": "Needs a human reviewer", "subtitle": reason}


def confidence(r) -> tuple[str, str]:
    """Heuristic confidence label from how far P(referable) is from the operating threshold (NOT a calibrated probability of being right)."""
    if r.status != "ok" or r.p_referable is None:
        return "n/a", ""
    if r.forced_despite_rejection:
        return "Unreliable", "the analysis was forced on an image that failed the quality gate"
    p, thr = float(r.p_referable), float(r.threshold)
    room = thr if p < thr else 1.0 - thr
    margin = abs(p - thr) / max(room, 1e-6)  # 0 at the threshold, 1 at p = 0 or 1
    order = ["Low", "Moderate", "High"]
    level = 2 if margin >= 0.6 else 1 if margin >= 0.3 else 0
    notes = [f"P(referable) is {abs(p - thr):.0%} away from the decision threshold"]
    if (r.decision or {}).get("action") == "human_review":
        return "Borderline", notes[0] + " (inside the review band)"
    if (r.quality or {}).get("status") == "enhance":
        level = max(0, level - 1)
        notes.append("image quality is below ideal")
    cc = (r.cross_check or {}).get("status")
    if cc in ("lesions_only", "classifier_only"):
        level = min(level, 1)
        notes.append("the lesion evidence does not fully agree with the classifier")
    return order[level], "; ".join(notes) + ". A heuristic, not a calibrated probability of being right"


# --------------------------------------------------------------------------------------------- narrative
def _quadrant_counts(features: dict, key: str) -> list[int]:
    return [int(features.get(f"{key}_q{q}", 0)) for q in range(4)]


def narrative(r) -> list[str]:
    """Plain-language reasoning, one short paragraph per bullet."""
    if r.status == "error":
        return [r.message]
    q = r.quality or {}
    if r.status == "quality_rejected":
        return [f"The image was not analysed: {q.get('reason', 'quality too low')}.",
                "Retake the photograph with the retina in sharp focus, even illumination and the whole field of view in frame, then upload it again."]
    out: list[str] = []
    p, thr = float(r.p_referable), float(r.threshold)
    action = (r.decision or {}).get("action", "")
    word = {"refer": "so the result is a recommendation to refer", "no_referral": "so no referral is indicated",
            "human_review": "which is too close to the threshold to call automatically, so a human reviewer should decide"}.get(action, "")
    out.append(f"**Screening result.** The model's referable-DR score is {p:.0%} (summed probability of moderate non-proliferative DR or worse, temperature-scaled on a "
               f"validation split; not a verified probability for this camera or population). The decision threshold is {thr:.0%}, {word}.")
    probs = list(r.grade_probs or [])
    if probs:
        order = sorted(range(5), key=lambda i: -probs[i])
        top, second = order[0], order[1]
        severe = probs[3] + probs[4]
        txt = f"**Severity.** The most likely grade is {GRADE_NAMES[top]} ({probs[top]:.0%}); next is {GRADE_NAMES[second]} ({probs[second]:.0%})."
        if top < 3 and severe >= 0.10:
            txt += f" There is a {severe:.0%} combined chance of severe or proliferative disease, so the true severity may be higher than the top grade suggests."
        out.append(txt)
    if q.get("status") == "enhance":
        out.append(f"**Image quality.** {q.get('reason', '')} Contrast enhancement was applied before analysis, so treat the result with a little more caution.")
    if r.forced_despite_rejection:
        out.append("**Warning.** This image failed the quality gate and was analysed only because you asked for it. The result is unreliable.")
    ev = r.lesion_evidence or {}
    if ev.get("available"):
        f, rule = ev["features"], ev["rule"]
        found = []
        for k in ("EX", "HE", "MA", "SE"):
            n = int(f.get(f"{k}_count", 0))
            if n:
                found.append(f"{n} {FINDING_NAMES[k]} region{'s' if n != 1 else ''}")
        if found:
            out.append("**Lesion evidence.** A lesion detector (trained on expert-annotated images) found " + ", ".join(found) + ". "
                       "These are detected regions, not verified lesion counts: fragments and false detections happen, particularly with uneven lighting.")
        else:
            out.append("**Lesion evidence.** The lesion detector found no microaneurysms, haemorrhages or exudates. It can miss subtle lesions.")
        large = sum(int(f.get(f"{k}_large", 0)) for k in ("MA", "HE", "EX", "SE"))
        if large:
            out.append(f"{large} very large detected region{'s were' if large != 1 else ' was'} ignored as probable lighting artefacts.")
        if rule.get("dme_flag"):
            n = int(f.get("EX_within_1DD", 0))
            out.append(f"**Macula.** {n} hard-exudate region{'s lie' if n != 1 else ' lies'} within one disc diameter of the estimated fovea, a possible sign of macular involvement (a risk indicator, not a diagnosis).")
        agree = (rule["referable"] == bool(r.referable))
        out.append(f"**ICDR criteria.** Applying the international severity criteria to those findings gives grade {rule['grade']} ({GRADE_NAMES[rule['grade']]}). "
                   + ("That points the same way as the classifier's referral decision." if agree else
                      ("That says referable disease where the classifier does not: a reviewer should look at this image." if rule["referable"] else
                       "That does not corroborate the classifier's referral: the flag rests on the classifier alone.")))
        na = rule.get("not_assessed") or []
        if na:
            out.append("**Not assessed.** Venous beading, intraretinal microvascular abnormalities and new vessel growth cannot be assessed by this system, so signs of proliferative disease may be missed.")
        if ev.get("location_note") and "approximate" in ev["location_note"]:
            out.append("The position of the macula was estimated from an anatomical rule for this image, so statements about quadrants and the macula are approximate.")
    else:
        out.append("**Lesion evidence.** Not available in this session, so the result rests on the classifier alone, which does not explain itself.")
    adv = (r.advice or {}).get("advice")
    if adv:
        out.append(f"**Suggested next step (illustrative).** {adv} A qualified clinician must make any real decision.")
    return out


def _md_bold(s: str) -> str:
    out, parts = [], esc(s).split("**")
    for i, part in enumerate(parts):
        out.append(f"<b>{part}</b>" if i % 2 else part)
    return "".join(out)


# --------------------------------------------------------------------------------------------- HTML fragments (styled by drscreen.ui_theme.CSS)
def _uri(arr, max_side: int = 1600, quality: int = 90) -> str:
    """RGB array -> JPEG data URI (down-scaled so the page stays light)."""
    im = Image.fromarray(np.asarray(arr).astype(np.uint8)).convert("RGB")
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _row(label: str, value: str, small: str = "", tip: str = "") -> str:
    t = f' title="{esc(tip)}"' if tip else ""
    return f'<div><dt{t}>{esc(label)}</dt><dd>{esc(value)}{f"<small>{esc(small)}</small>" if small else ""}</dd></div>'


QUALITY_WORDS = {"good": "Good", "enhance": "Reduced (contrast enhancement applied)", "reject": "Insufficient"}


def _score_svg(p: float, thr: float, margin: float) -> str:
    """Decision-threshold visual: a 0-1 axis, the operating threshold (dashed line), the optional human-review band (hatched) and the score (triangle).
    Position, shape and text carry the meaning, not colour alone."""
    W, x0, x1 = 480, 16, 464
    X = lambda v: x0 + max(0.0, min(1.0, v)) * (x1 - x0)
    xt, xp = X(thr), X(p)
    clamp = lambda x, lo, hi: max(lo, min(hi, x))
    side = "above" if p >= thr else "below"
    label = (f"Referable-DR score {p:.2f} is {side} the decision threshold {thr:.2f}"
             + (f"; it lies inside the human-review band of plus or minus {margin:.2f}" if margin and abs(p - thr) <= margin else ""))
    band, band_legend = "", ""
    if margin and margin > 0:
        band = f'<rect class="rl-band" x="{X(thr - margin):.1f}" y="38" width="{X(thr + margin) - X(thr - margin):.1f}" height="16"/>'
        band_legend = f'<text x="{x0}" y="116" text-anchor="start">hatched: human-review band (± {margin:.2f})</text>'
    ticks = "".join(f'<line class="rl-tick" x1="{X(v):.1f}" y1="42" x2="{X(v):.1f}" y2="50"/>' for v in (0, .25, .5, .75, 1))
    return (f'<div class="rl-score"><svg viewBox="0 0 {W} 122" role="img" aria-label="{esc(label)}"><title>{esc(label)}</title>'
            '<defs><pattern id="rl-hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="5" stroke="#80590f" stroke-width="1.3" opacity=".6"/></pattern></defs>'
            f'<rect class="rl-zone-ref" x="{xt:.1f}" y="38" width="{x1 - xt:.1f}" height="16"/>{band}'
            f'<line class="rl-axis" x1="{x0}" y1="46" x2="{x1}" y2="46"/>{ticks}'
            f'<line class="rl-thr" x1="{xt:.1f}" y1="30" x2="{xt:.1f}" y2="62"/>'
            f'<g class="rl-mk" style="--dx:{x0 - xp:.1f}px"><polygon class="rl-marker" points="{xp - 7:.1f},20 {xp + 7:.1f},20 {xp:.1f},34"/>'
            f'<text class="rl-lab-strong" x="{clamp(xp, 46, W - 46):.1f}" y="13" text-anchor="middle">score {p:.2f}</text></g>'
            f'<text x="{x0}" y="74" text-anchor="start">0</text><text x="{x1}" y="74" text-anchor="end">1</text>'
            f'<text x="{clamp(xt, 52, W - 52):.1f}" y="74" text-anchor="middle">threshold {thr:.2f}</text>'
            f'<text x="{x0}" y="98" text-anchor="start" class="rl-lab-strong">◂ No referral</text><text x="{x1}" y="98" text-anchor="end" class="rl-lab-strong">Referral ▸</text>'
            f'{band_legend}</svg></div>')


def render_empty_html() -> str:
    return ('<section class="rl-panel rl-empty rl-enter"><div class="rl-eyebrow">Screening result</div>'
            '<h2 class="rl-reco">Upload a fundus photograph to begin</h2>'
            '<p>You will get a screening recommendation, the evidence behind it, and the retinal images that support it.</p>'
            '<ol><li>Add a colour fundus photograph of one eye (JPEG or PNG).</li><li>Choose a model and select <b>Analyse image</b>.</li>'
            '<li>Review the recommendation, then the evidence and calibration below.</li></ol></section>')


def render_loading_html() -> str:
    return ('<section class="rl-panel rl-loading rl-enter" aria-busy="true" role="status"><div class="rl-eyebrow">Screening result</div>'
            '<h2 class="rl-reco" style="color:var(--rl-text)">Analysing…</h2>'
            '<p class="rl-reco-sub">Running the image-quality check, the classifier and the lesion detector on this photograph.</p><div class="rl-bar"></div></section>')


def render_summary_html(r) -> str:
    """The primary result panel: recommendation, three headline numbers, the threshold chart, collapsible details and one line on limits."""
    v = verdict(r)
    accent, _soft = PALETTE[v["kind"]]
    q = r.quality or {}
    stats, visual, rows = "", "", []
    if r.status == "ok":
        conf, _conf_note = confidence(r)
        p, thr = float(r.p_referable), float(r.threshold)
        margin = float(getattr(r, "review_margin", 0.0) or 0.0)
        num, _, gname = (r.grade_name or "? n/a").partition(" ")
        stats = ('<div class="rl-stats">'
                 f'<div title="Summed probability of grades 2–4, temperature-scaled"><span class="rl-stat-v rl-count" data-to="{p:.2f}">{p:.2f}</span><span class="rl-stat-l">referable score</span></div>'
                 f'<div title="Most probable of the five ICDR grades"><span class="rl-stat-v">{esc(num)}</span><span class="rl-stat-l">{esc(gname)}</span></div>'
                 f'<div title="Heuristic: distance of the score from the threshold"><span class="rl-stat-v rl-stat-word">{esc(conf)}</span><span class="rl-stat-l">confidence</span></div>'
                 '</div>')
        visual = _score_svg(p, thr, margin)
        rows = [_row("Predicted grade", r.grade_name or "n/a", tip="Most probable of the five ICDR grades"),
                _row("Referable-DR score", f"{p:.3f}", tip="Summed probability of grades 2–4, temperature-scaled on validation data"),
                _row("Decision threshold", f"{thr:.3f}", tip="Chosen on validation data for at least 90% sensitivity"),
                _row("Human-review band", f"± {margin:.3f}" if margin else "not used"),
                _row("Image quality", QUALITY_WORDS.get(q.get("status", ""), "n/a"), q.get("reason", "")),
                _row("Confidence", conf, tip="Heuristic, from the distance to the threshold; not a probability of being right"),
                _row("Model", r.model_name or "n/a")]
    elif r.status == "quality_rejected":
        rows = [_row("Image quality", QUALITY_WORDS["reject"], (q.get("reason", "") + " (heuristic check)").strip()), _row("Prediction", "None made")]
    details = ""
    if rows:
        kv = f'<dl class="rl-kv">{"".join(rows)}</dl>'
        details = f'<details class="rl-more"><summary>Details</summary>{kv}</details>' if r.status == "ok" else kv
    note = "Research model output, not validated for your camera or patient population. Nothing here is a diagnosis." if r.status == "ok" else ""
    note_html = f'<p class="rl-note">{esc(note)}</p>' if note else ""
    return (f'<section class="rl-panel rl-enter rl-result" style="--accent:{accent}" aria-labelledby="rl-res-h"><div class="rl-eyebrow">Screening result</div>'
            f'<h2 id="rl-res-h" class="rl-reco">{esc(v["title"])}</h2><p class="rl-reco-sub">{esc(v["subtitle"])}</p>{stats}{visual}'
            f'{details}{note_html}</section>')


def render_analysing_html(image) -> str:
    """Shown on the image side while the models run: the uploaded photograph with a slow scanning line (no fake progress figures)."""
    try:
        src = _uri(image, 1400)
    except Exception:
        return ""
    return ('<section class="rl-viewer rl-scanning" aria-label="Analysing the photograph" aria-busy="true">'
            '<div class="rl-viewer-bar"><span class="rl-scan-label"><span class="rl-dot"></span>Analysing: quality check, classifier and lesion detector</span></div>'
            f'<div class="rl-stage-wrap"><div class="rl-stage rl-stage-scan"><img class="rl-img rl-img-scan" src="{src}" alt="Photograph being analysed"><div class="rl-scanline"></div></div></div>'
            '</section>')

_ICONS = {"in": '<path d="M11 11m-6 0a6 6 0 1 0 12 0a6 6 0 1 0-12 0M20 20l-4.2-4.2M11 8v6M8 11h6"/>', "out": '<path d="M11 11m-6 0a6 6 0 1 0 12 0a6 6 0 1 0-12 0M20 20l-4.2-4.2M8 11h6"/>',
          "fit": '<path d="M4 12a8 8 0 1 0 2.4-5.7M4 4v4.5h4.5"/>', "expand": '<path d="M9 4H4v5M15 4h5v5M9 20H4v-5M15 20h5v-5"/>'}
_TOOLS = (("in", "Zoom in"), ("out", "Zoom out"), ("fit", "Reset view (fit to window)"), ("expand", "Expand or close the focused inspection view"))
_VIEWS = (("orig", "Original", "Original photograph, as uploaded."),
          ("proc", "Preprocessed", "Cropped to the retina, masked and resized to the model's input size."),
          ("cam", "Grad-CAM", "Grad-CAM: where the classifier's decision was most influenced (warmer = more). A visualisation of model attention, not lesion detection."),
          ("les", "Lesion evidence", "Regions flagged by the lesion detector, which was trained on expert-annotated images. Detections need clinician verification; false detections occur."))


def render_viewer_html(r) -> str:
    """Retinal image workstation shown after analysis: segmented view switcher, zoom/fit/expand tools, one large image at a time. Views that do not
    exist for this analysis are disabled rather than hidden."""
    if r.original is None:
        return ""
    imgs = {"orig": r.original, "proc": r.processed, "cam": r.gradcam, "les": r.lesion_overlay}
    inputs, labels, views = [], [], []
    for k, name, _cap in _VIEWS:
        have = imgs[k] is not None
        inputs.append(f'<input class="rl-sr" type="radio" name="rlv" id="rlv-{k}" aria-label="{esc(name)}"{" checked" if k == "orig" else ""}{"" if have else " disabled"}>')
        labels.append(f'<label for="rlv-{k}"' + ('' if have else ' class="rl-off" title="Not available for this analysis"') + f'>{esc(name)}</label>')
        if have:
            views.append(f'<div class="rl-view" data-view="{k}"><div class="rl-stage"><img class="rl-img" src="{_uri(imgs[k])}" alt="{esc(name)} retinal image" draggable="false"></div></div>')
    tools = "".join(f'<button type="button" data-rl="{a}" aria-label="{esc(t)}" title="{esc(t)}"><svg viewBox="0 0 24 24" aria-hidden="true">{_ICONS[a]}</svg></button>' for a, t in _TOOLS)
    caps = "".join(f'<span class="rl-cap" data-cap="{k}">{esc(cap)}</span>' for k, _n, cap in _VIEWS)
    h, w = np.asarray(r.original).shape[:2]
    return (f'<section class="rl-viewer rl-enter" aria-label="Retinal image viewer">{"".join(inputs)}<div class="rl-viewer-bar"><div class="rl-seg" role="group" aria-label="Image view">{"".join(labels)}</div>'
            f'<div class="rl-tools">{tools}</div></div><div class="rl-stage-wrap">{"".join(views)}</div>'
            f'<div class="rl-viewer-foot"><span>{caps}</span><span>{w} × {h} px · Ctrl+scroll to zoom, drag to pan, double-click to zoom</span></div></section>')


def render_interpretation_html(r) -> str:
    items = "".join(f"<li>{_md_bold(t)}</li>" for t in narrative(r))
    return f'<div class="rl-prose"><ul>{items}</ul></div>'


def _fig(arr, caption: str) -> str:
    return f'<figure class="rl-fig"><img src="{_uri(arr, 1100)}" alt="{esc(caption)}"><figcaption>{esc(caption)}</figcaption></figure>'


def render_gradcam_html(r) -> str:
    if r.status != "ok" or r.gradcam is None:
        return '<div class="rl-prose"><p>No Grad-CAM map was produced for this analysis.</p></div>'
    figs = f'<div class="rl-figs">{_fig(r.processed if r.processed is not None else r.original, "Pre-processed image")}{_fig(r.gradcam, "Grad-CAM overlay")}</div>'
    return (figs + '<p class="rl-small">Grad-CAM highlights the image regions that most increased the classifier\'s score for its predicted grade. It shows model attention, not lesion detection, '
            'and a map can look plausible even when the reasoning is wrong. The weight-randomisation sanity check run on the validation set is reported in the technical report.</p>')


def render_lesion_html(r) -> str:
    ev = r.lesion_evidence or {}
    if r.status != "ok" or not ev.get("available"):
        return ('<div class="rl-prose"><p>Lesion evidence is not available for this analysis, so the result rests on the classifier alone, which does not explain itself. '
                'The lesion detector may be missing or may not have produced output for this image.</p></div>')
    f, rule = ev["features"], ev["rule"]
    counts = "".join(f'<div><b>{int(f.get(f"{k}_count", 0))}</b><span>{esc(FINDING_NAMES[k].split(" (")[0])}s</span></div>' for k in ("MA", "HE", "EX", "SE"))
    counts += f'<div><b>{int(f.get("EX_within_1DD", 0))}</b><span>exudates near macula</span></div>'
    he, ex, ma = _quadrant_counts(f, "HE"), _quadrant_counts(f, "EX"), _quadrant_counts(f, "MA")
    cell = lambda q: f"<td><b>{QUADRANT_LABELS[q]}</b><br>haemorrhage {he[q]} · exudate {ex[q]} · MA {ma[q]}</td>"
    quad = (f'<table class="rl-quad"><tr>{cell(2)}{cell(0)}</tr><tr>{cell(3)}{cell(1)}</tr></table>'
            '<p class="rl-small">Nasal side on the left, temporal on the right; upper row superior. Detected regions per quadrant, relative to the estimated optic disc and fovea.</p>')
    rows = "".join(f'<tr><td>{esc(c["name"])}</td><td><span class="rl-status {esc(c["status"])}">{esc(c["status"].replace("_", " "))}</span></td><td>{esc(c["detail"])}</td></tr>' for c in rule["criteria"])
    thr_note = ""
    if str(rule["params"].get("name", "")).startswith("data-calibrated"):
        thr_note = '<p class="rl-small">The count thresholds are calibrated to this detector\'s output on APTOS images. Detected-region counts are not clinical lesion counts; do not read them as clinical cut-offs.</p>'
    fig = f'<div class="rl-figs">{_fig(r.lesion_overlay, "Detected lesion regions")}</div>' if r.lesion_overlay is not None else ""
    loc = f'<p class="rl-small">Localisation: {esc(ev["location_note"])}</p>' if ev.get("location_note") else ""
    return (f'{fig}<div class="rl-counts">{counts}</div>{quad}<div class="rl-sub">ICDR criteria checklist · rule-derived grade {rule["grade"]} ({esc(GRADE_NAMES[rule["grade"]])})</div>'
            f'<table class="rl-table"><tr><th>Criterion</th><th>Status</th><th>Detail</th></tr>{rows}</table>{thr_note}{loc}')


def render_calibration_html(r) -> str:
    if r.status != "ok" or not r.grade_probs:
        return '<div class="rl-prose"><p>No model output is available for this analysis.</p></div>'
    accent = PALETTE[verdict(r)["kind"]][0]
    bars = "".join(f'<div class="rl-prob{" top" if i == r.grade else ""}" style="--accent:{accent}"><span>{esc(GRADE_NAMES[i])}</span>'
                   f'<span class="track"><i style="--w:{p * 100:.1f}%"></i></span><span class="v">{p:.1%}</span></div>' for i, p in enumerate(r.grade_probs))
    conf, conf_note = confidence(r)
    margin = float(getattr(r, "review_margin", 0.0) or 0.0)
    cc = r.cross_check or {}
    rows = [_row("Referable-DR score", f"{float(r.p_referable):.3f}"), _row("Decision threshold", f"{float(r.threshold):.3f}"),
            _row("Human-review band", f"± {margin:.3f}" if margin else "not used"), _row("Confidence level (heuristic)", conf)]
    if cc.get("message"):
        rows.append(_row("Lesion rules vs classifier", cc["message"]))
    return (f'<div class="rl-eyebrow">Grade probabilities</div>{bars}'
            f'<p class="rl-small">{esc(r.calibration_note)}</p><dl class="rl-kv" style="max-width:40rem">{"".join(rows)}</dl>'
            '<p class="rl-small"><b>Reading these numbers.</b> The grade probabilities and the referable-DR score are temperature-scaled outputs of the network, fitted on a validation split. '
            f'They are not verified probabilities for new cameras or populations. The confidence level is a heuristic. {esc(conf_note)}.</p>')
