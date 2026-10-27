"""Diabetic retinopathy screening: research-prototype web interface.

Upload a colour fundus photograph and get: a screening recommendation, how confident the system is, the reasons behind it (lesion evidence, ICDR
criteria checklist, plain-language explanation), the supporting images (pre-processed, Grad-CAM, detected lesions) and a downloadable PDF case report.

Run:   python app/app.py                      (prefers artifacts/efficientnet_b0_robust, then artifacts/efficientnet_b0; $DR_CHECKPOINT overrides)
       python app/app.py --port 7860 --share

It loads REAL trained checkpoints; if none is available it says so. It never fabricates predictions. It is not a hosted service: it runs where you start it.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drscreen import INTENDED_USE  # noqa: E402
from drscreen.casereport import build_case_pdf  # noqa: E402
from drscreen.explain import (esc, render_calibration_html, render_empty_html, render_gradcam_html, render_interpretation_html, render_lesion_html,  # noqa: E402
                               render_loading_html, render_summary_html, render_viewer_html)
from drscreen.ui_theme import CSS, HEAD, PRODUCT_NAME, build_theme, disclaimer_html, header_html  # noqa: E402
from drscreen.metrics import GRADE_NAMES  # noqa: E402
from drscreen.predict import Predictor  # noqa: E402

# label shown in the model menu -> checkpoint path (relative to the project root). The robustness-augmented model is listed first (recommended by the
# pre-set rule in scripts/compare_robust.py).
MODEL_FILES = {
    "Robust EfficientNet-B0 (recommended)": "artifacts/efficientnet_b0_robust/best.pt",
    "Standard EfficientNet-B0": "artifacts/efficientnet_b0/best.pt",
    "EyePACS-augmented EfficientNet-B0 (experimental)": "artifacts/efficientnet_b0_eyepacs/best.pt",
}
CANDIDATES = [p for p in MODEL_FILES.values()] + ["artifacts/best.pt"]
_PREDICTORS: dict[str, Predictor] = {}
_PREDICTOR: Predictor | None = None  # most recently loaded (kept for scripts that import this module)
_LOAD_ERROR: str | None = None


def available_models(explicit: str | None = None) -> dict[str, Path]:
    """Models that exist on disk, in menu order. $DR_CHECKPOINT / --checkpoint is added first as 'Custom checkpoint'; a validated ensemble
    (artifacts/ensemble/ensemble.json, written by scripts/evaluate_ensemble.py) is offered when it exists."""
    out: dict[str, Path] = {}
    for c in (explicit, os.environ.get("DR_CHECKPOINT")):
        if c and Path(c).is_file():
            out["Custom checkpoint"] = Path(c)
            break
    if (ROOT / "artifacts/ensemble/ensemble.json").is_file():
        out["Ensemble (selected on validation)"] = ROOT / "artifacts/ensemble/ensemble.json"
    for label, rel in MODEL_FILES.items():
        if (ROOT / rel).is_file():
            out[label] = ROOT / rel
    if not out and (ROOT / "artifacts/best.pt").is_file():
        out["Checkpoint"] = ROOT / "artifacts/best.pt"
    return out


def find_checkpoint(explicit: str | None = None) -> Path | None:
    m = available_models(explicit)
    return next(iter(m.values()), None)


def get_predictor(label: str | None = None, explicit: str | None = None) -> Predictor | None:
    """Load (and cache) the predictor for a menu label; the first available model if label is None/unknown."""
    global _PREDICTOR, _LOAD_ERROR
    models = available_models(explicit)
    if not models:
        _LOAD_ERROR = ("No trained checkpoint found. Download the Kaggle run output (`python scripts/kaggle_download_outputs.py`) "
                       "or set DR_CHECKPOINT=path/to/best.pt.")
        return None
    label = label if label in models else next(iter(models))
    if label not in _PREDICTORS:
        try:
            _PREDICTORS[label] = Predictor(models[label], ensemble=models[label]) if models[label].suffix == ".json" else Predictor(models[label])
            _LOAD_ERROR = None
        except Exception as e:
            _LOAD_ERROR = f"Checkpoint {models[label]} could not be loaded: {e}"
            return None
    _PREDICTOR = _PREDICTORS[label]
    return _PREDICTOR


def load_predictor(explicit: str | None = None) -> Predictor | None:
    return get_predictor(None, explicit)


def analyze(image, force: bool = False, predictor: Predictor | None = None):
    """Simple 6-tuple API (original, processed, gradcam, lesion_overlay, grade_probabilities_dict, markdown). Safe for any input."""
    p = predictor or _PREDICTOR
    if p is None:
        return None, None, None, None, {}, f"### Model unavailable\n{_LOAD_ERROR or 'Model not loaded.'}\n\n_{INTENDED_USE}_"
    r = p.predict(image, with_gradcam=True, force=bool(force))
    probs = {GRADE_NAMES[i]: float(v) for i, v in enumerate(r.grade_probs)} if r.grade_probs else {}
    return r.original, r.processed, r.gradcam, r.lesion_overlay, probs, r.to_markdown()


def _message_panel(title: str, text: str) -> str:
    return (f'<section class="rl-panel rl-enter"><div class="rl-eyebrow">Screening result</div><h2 class="rl-reco" style="color:var(--rl-text)">{esc(title)}</h2>'
            f'<p class="rl-reco-sub">{esc(text)}</p></section>')


def _empty_case(summary: str, technical: str) -> dict:
    return {"summary": summary, "viewer": "", "interpretation": "", "gradcam": "", "lesion": "", "calibration": "", "technical": technical, "pdf": None,
            "has_image": False, "ok": False}


def analyze_case(image, force: bool = False, model_label: str | None = None, predictor: Predictor | None = None) -> dict:
    """Everything the interface shows for one image (HTML fragments, the case-report PDF path, flags). Never raises."""
    t0 = time.time()
    try:
        p = predictor or get_predictor(model_label)
        if p is None:
            return _empty_case(_message_panel("Model unavailable", _LOAD_ERROR or "Model not loaded."), f"### Model unavailable\n{_LOAD_ERROR}")
        r = p.predict(image, with_gradcam=True, force=bool(force))
        pdf = None
        try:
            pdf = str(build_case_pdf(r, Path(tempfile.mkdtemp(prefix="drcase_")) / f"dr_case_{time.strftime('%Y%m%d_%H%M%S')}.pdf"))
        except Exception as e:  # the PDF is a convenience; never let it break the analysis
            print(f"[app] case PDF failed: {e}", flush=True)
        return {"summary": render_summary_html(r), "viewer": render_viewer_html(r), "interpretation": render_interpretation_html(r), "gradcam": render_gradcam_html(r),
                "lesion": render_lesion_html(r), "calibration": render_calibration_html(r), "pdf": pdf, "has_image": r.original is not None, "ok": r.status == "ok",
                "technical": r.to_markdown() + f"\n\n_Model: {r.model_name} · analysis time {time.time() - t0:.1f} s_"}
    except Exception as e:
        return _empty_case(_message_panel("Something went wrong", f"{type(e).__name__}: {e}"), f"### Error\n{type(e).__name__}: {e}")


HOWTO = """**For a good result.** Use a colour photograph of one eye, in sharp focus and evenly lit, with the whole circular field of view in the frame (optic disc and macula visible).
JPEG or PNG. A photograph that is blurry, very dark or cut off is returned with a request to retake it instead of a guess.

**Reading the answer.** The result panel gives the recommendation and where the model's score sits against the decision threshold. The tabs below explain it: the reasoning in words,
Grad-CAM (model attention), detected lesions with the ICDR criteria checklist, and the calibration of the numbers. Detected lesions are evidence for a clinician to check, not verified findings."""


def build_ui(default_model: str | None = None, demo_image: str | None = None):
    import gradio as gr

    models = list(available_models().keys()) or ["(no model found)"]
    with gr.Blocks(title=f"{PRODUCT_NAME} · diabetic retinopathy screening research") as demo:
        gr.HTML(header_html(), elem_classes="rl-plain")
        last = gr.State(None)  # the analysed photograph, so a re-analysis with another model works while the uploader is hidden
        with gr.Row(elem_id="rl-workspace", equal_height=False):
            with gr.Column(scale=7, min_width=340):
                inp = gr.Image(label="Fundus photograph", type="numpy", image_mode="RGB", height="62vh", sources=["upload", "clipboard"], value=demo_image,
                               elem_id="rl-input", show_label=False)
                viewer = gr.HTML(visible=False, elem_classes="rl-plain")
                with gr.Row(elem_classes="rl-controls"):
                    model = gr.Dropdown(choices=models, value=default_model or models[0], label="Model", scale=4, interactive=len(models) > 1)
                    btn = gr.Button("Analyse image", variant="primary", scale=2)
                    clear = gr.Button("New image", scale=1)
                force = gr.Checkbox(label="Analyse even if the quality check fails (the result will be unreliable)", value=False, elem_classes="rl-small-check")
                with gr.Accordion("How to use", open=False, elem_classes="rl-accordion"):
                    gr.Markdown(HOWTO)
            with gr.Column(scale=5, min_width=320):
                summary = gr.HTML(render_empty_html(), elem_classes="rl-plain")
                pdf = gr.DownloadButton("Download case report (PDF)", visible=False, variant="secondary", size="sm", elem_classes="rl-download")
        with gr.Column(visible=False, elem_id="rl-details") as details:
            with gr.Tabs():
                with gr.Tab("Interpretation"):
                    t_interp = gr.HTML(elem_classes="rl-plain")
                with gr.Tab("Grad-CAM"):
                    t_cam = gr.HTML(elem_classes="rl-plain")
                with gr.Tab("Lesion evidence"):
                    t_les = gr.HTML(elem_classes="rl-plain")
                with gr.Tab("Calibration"):
                    t_cal = gr.HTML(elem_classes="rl-plain")
                with gr.Tab("Technical details"):
                    t_tech = gr.Markdown()
        status = ("Models available: " + ", ".join(available_models().keys())) if available_models() else ("No model found. " + (_LOAD_ERROR or "Run the Kaggle download script."))
        lesion_state = "available" if (ROOT / "artifacts/lesion/lesion_unet.pt").is_file() else "not available (artifacts/lesion missing)"
        gr.HTML(f'<p class="rl-small" style="margin-top:24px">{esc(status)}. Lesion evidence: {lesion_state}.</p>' + disclaimer_html(), elem_classes="rl-plain")

        outs = [inp, viewer, summary, pdf, details, t_interp, t_cam, t_les, t_cal, t_tech, last]
        keep = gr.update()

        def run(image, force_, model_label, prev):
            img = image if image is not None else prev
            if img is not None:
                yield [keep, keep, render_loading_html(), gr.update(visible=False), keep, keep, keep, keep, keep, keep, keep]  # honest busy state, no fake progress
            o = analyze_case(img, force_, model_label)
            shown = o["has_image"]
            yield [gr.update(visible=not shown), gr.update(value=o["viewer"], visible=shown), o["summary"], gr.update(value=o["pdf"], visible=bool(o["pdf"])),
                   gr.update(visible=shown), o["interpretation"], o["gradcam"], o["lesion"], o["calibration"], o["technical"], img]

        def reset():
            return [gr.update(value=None, visible=True), gr.update(value="", visible=False), render_empty_html(), gr.update(value=None, visible=False),
                    gr.update(visible=False), "", "", "", "", "", None]

        btn.click(run, [inp, force, model, last], outs, show_progress="hidden")
        clear.click(reset, None, outs)
        if demo_image:  # presentation mode: analyse the pre-loaded photograph as soon as the page opens
            demo.load(run, [inp, force, model, last], outs)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--demo-image", default=None, help="pre-load this photograph and analyse it on page load (for presentations)")
    a = ap.parse_args()
    if a.checkpoint:
        os.environ["DR_CHECKPOINT"] = a.checkpoint
    get_predictor()  # warm up the default model so the first request is fast
    build_ui(demo_image=a.demo_image).launch(server_port=a.port, share=a.share, css=CSS, theme=build_theme(), head=HEAD)
