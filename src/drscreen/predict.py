"""Single reusable inference entry point for scripts, notebooks and the Gradio app.

    p = Predictor("artifacts/efficientnet_b0/best.pt")          # loads a REAL trained checkpoint or raises
    r = p.predict(image_or_path)                           # never raises on bad input; see r.status

If a lesion model directory exists (artifacts/lesion/: lesion_unet.pt, locator.pt, lesion_config.json, written by the remote
lesion run) the result also carries rule-derived lesion evidence with reasons; otherwise it says plainly that none was computed.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import INTENDED_USE
from .fusion import cross_check, decide, decide_with_evidence
from .gradcam import GradCAM, overlay_heatmap
from .metrics import GRADE_NAMES
from .models import grade_probs, load_checkpoint, referable_prob
from .preprocess import ImageError, preprocess_for_model, read_image, to_rgb_uint8
from .quality import QualityThresholds, assess_quality, enhance_image
from .rules import RuleParams, grade_name, lesion_evidence, referral_advice


@dataclass
class PredictionResult:
    status: str                      # ok | quality_rejected | error
    message: str = ""
    original: np.ndarray | None = None
    processed: np.ndarray | None = None
    gradcam: np.ndarray | None = None
    lesion_overlay: np.ndarray | None = None
    grade: int | None = None
    grade_name: str | None = None
    grade_probs: list | None = None
    p_referable: float | None = None
    referable: bool | None = None
    threshold: float | None = None
    decision: dict | None = None
    quality: dict | None = None
    advice: dict | None = None
    lesion_evidence: dict = field(default_factory=lesion_evidence)
    cross_check: dict | None = None
    calibration_note: str = ""
    model_name: str = ""
    latency_ms: float | None = None
    forced_despite_rejection: bool = False
    review_margin: float = 0.0       # half-width of the human-review band around the threshold (0 = no band)

    def to_dict(self, include_images: bool = False) -> dict:
        d = asdict(self)
        if not include_images:
            for k in ("original", "processed", "gradcam", "lesion_overlay"):
                d[k] = None
        return d

    def to_markdown(self) -> str:
        if self.status == "error":
            return f"### Could not analyse this file\n{self.message}\n\n_{INTENDED_USE}_"
        q = self.quality or {}
        lines = [f"### Image quality: **{q.get('status', '?').upper()}** (heuristic score {q.get('score', 0):.2f})", q.get("reason", ""), ""]
        if self.status == "quality_rejected":
            lines += ["**No prediction was made** because the image failed the quality gate. Recapture the image and try again.", ""]
        else:
            if self.forced_despite_rejection:
                lines += ["> ⚠️ Analysis was forced despite a quality rejection; treat the output as unreliable.", ""]
            d = self.decision or {}
            lines += [
                f"### Screening result: **{'REFERABLE DR suspected' if self.referable else 'Non-referable'}**",
                f"- Decision: **{d.get('action', '?').replace('_', ' ')}**. {d.get('reason', '')}",
                f"- Classifier's predicted grade: **{self.grade_name}** (argmax of a CNN; it does not by itself explain why)",
                f"- P(referable) = **{self.p_referable:.1%}** (sum of P(grade 2-4); operating threshold {self.threshold:.2f})",
                "- Grade probabilities: " + ", ".join(f"{GRADE_NAMES[i]}: {p:.1%}" for i, p in enumerate(self.grade_probs)),
                f"- {self.calibration_note}",
                f"- Referral policy (illustrative): {self.advice['advice']}",
                "",
                "### Why: lesion evidence",
            ]
            ev = self.lesion_evidence
            if ev.get("available"):
                lines += [ev["markdown"], ""]
                if self.cross_check:
                    lines += [f"**Cross-check with the classifier:** {self.cross_check['message']}", ""]
                if ev.get("location_note"):
                    lines += [f"_Localisation: {ev['location_note']}_", ""]
                lines += [f"_{ev['caveat']}_", "",
                          "Overlay colours: red = microaneurysm, orange = haemorrhage, yellow = hard exudate, cyan = soft exudate; "
                          "blue ring = optic disc, green cross = estimated fovea, white lines = quadrant axes."]
            else:
                lines += [f"**Not computed.** {ev['reason']}"]
            lines += ["", "**Grad-CAM** highlights image regions that influenced the classifier's score. It does not prove a lesion is present "
                          "or that the model reasons like a clinician."]
        lines += ["", f"_{INTENDED_USE}_"]
        return "\n".join(lines)


class Predictor:
    def __init__(self, checkpoint: str | Path, calibration: str | Path | None = None, device: str | None = None, lesion_dir: str | Path | None = None,
                 ensemble: str | Path | None = None):
        """checkpoint: best.pt of a single model. ensemble: optional ensemble.json (members, flip-TTA flag, threshold, review margin, temperatures) written by
        scripts/evaluate_ensemble.py; the first member is then used for Grad-CAM, the quality thresholds and the display image."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.members, self.tta, ens = [], False, None
        if ensemble is not None:
            ep = Path(ensemble)
            ens = json.loads(ep.read_text())
            root = ep.parents[2]
            paths = [Path(m) if Path(m).is_absolute() else root / m for m in ens["members"]]
            checkpoint = paths[0]
            self.tta = bool(ens.get("tta"))
        self.checkpoint = Path(checkpoint)
        self.model_name = self.checkpoint.parent.name
        self.model, ck = load_checkpoint(self.checkpoint, self.device)
        self.cfg = ck["cfg"]
        self.img_size = int(self.cfg["img_size"])
        self.use_bg = bool(self.cfg.get("use_ben_graham", True))
        cal_path = Path(calibration) if calibration else self.checkpoint.parent / "calibration.json"
        self.cal = json.loads(cal_path.read_text()) if cal_path.is_file() else {}
        self.temperature = float(self.cal.get("temperature", 1.0))
        self.threshold = float(self.cal.get("threshold", 0.5))
        self.review_margin = float(self.cal.get("review_margin", 0.0))
        if ens is not None:  # ensemble: its own operating point; every member keeps its own validation-fitted temperature
            self.threshold, self.review_margin = float(ens["threshold"]), float(ens.get("review_margin", 0.0))
            for pth in paths:
                m, ck_ = (self.model, ck) if pth == self.checkpoint else load_checkpoint(pth, self.device)
                self.members.append({"model": m, "size": int(ck_["cfg"]["img_size"]), "bg": bool(ck_["cfg"].get("use_ben_graham", True)),
                                     "T": float(ens["temperatures"].get(pth.parent.name, 1.0))})
            self.model_name = "ensemble: " + " + ".join(p.parent.name.replace("efficientnet_", "") for p in paths) + (" (+flip TTA)" if self.tta else "")
        qt = self.cal.get("quality_thresholds")
        self.qthr = QualityThresholds(**qt) if qt else QualityThresholds()
        self.calibrated = bool(self.cal)
        self.calibration_note = (
            f"Probabilities are temperature-scaled (T = {self.temperature:.2f}, fitted on validation data); they are model scores, "
            "not clinically validated probabilities." if self.calibrated else
            "No calibration file found: raw softmax probabilities and a default 0.5 threshold are used (uncalibrated).")
        self.rule_params = RuleParams(**self.cal["rule_params"]) if self.cal.get("rule_params") else RuleParams()
        self.lesions = None
        self.lesion_error = None
        ld = Path(lesion_dir) if lesion_dir else self.checkpoint.parent.parent / "lesion"
        if (ld / "lesion_unet.pt").is_file() and (ld / "locator.pt").is_file():
            try:
                from .lesion import LesionAnalyzer

                self.lesions = LesionAnalyzer(ld, self.device)
            except Exception as e:  # keep the classifier usable; say why lesion evidence is missing
                self.lesion_error = f"lesion models in {ld} could not be loaded: {e}"

    def _ensemble_probs(self, work: np.ndarray) -> np.ndarray:
        """Mean of the members' temperature-scaled grade probabilities; with TTA each member averages the plain, h-flipped and v-flipped views' logits."""
        ps = []
        for m in self.members:
            _, _, x = preprocess_for_model(work, m["size"], m["bg"])
            xt = torch.from_numpy(x)[None].to(self.device)
            views = [xt, torch.flip(xt, [3]), torch.flip(xt, [2])] if self.tta else [xt]
            with torch.no_grad():
                lg = np.mean([m["model"](v).float().cpu().numpy()[0] for v in views], axis=0)
            ps.append(grade_probs(lg[None], m["T"])[0])
        return np.mean(ps, axis=0)

    def _evidence(self, work: np.ndarray):
        if self.lesions is None:
            ev = lesion_evidence(None)
            if self.lesion_error:
                ev["reason"] = self.lesion_error + " " + ev["reason"]
            return ev, None
        try:
            an = self.lesions.analyze(work)
            return lesion_evidence(an, self.rule_params), an["overlay"]
        except Exception as e:
            ev = lesion_evidence(None)
            ev["reason"] = f"Lesion analysis failed ({type(e).__name__}: {e}). " + ev["reason"]
            return ev, None

    def predict(self, image, with_gradcam: bool = True, force: bool = False, with_lesions: bool = True) -> PredictionResult:
        t0 = time.time()
        try:
            img = read_image(image) if isinstance(image, (str, Path)) else to_rgb_uint8(image)
        except ImageError as e:
            return PredictionResult("error", f"Invalid image: {e}")
        except Exception as e:  # defensive: arbitrary objects from the UI
            return PredictionResult("error", f"Unsupported input ({type(e).__name__}: {e})")
        q = assess_quality(img, self.qthr)
        base = dict(original=img, quality=q.to_dict(), threshold=self.threshold, calibration_note=self.calibration_note, model_name=self.model_name)
        if q.status == "reject" and not force:
            return PredictionResult("quality_rejected", q.reason, decision=decide("reject", None, self.threshold).to_dict(),
                                    latency_ms=(time.time() - t0) * 1000, **base)
        try:
            work = enhance_image(img) if q.status == "enhance" else img
            plain, _, x = preprocess_for_model(work, self.img_size, self.use_bg)
            xt = torch.from_numpy(x)[None].to(self.device)
            cam = None
            if with_gradcam:
                gc = GradCAM(self.model)
                try:
                    cam, logits = gc(xt, "pred")
                finally:
                    gc.remove()
            else:
                with torch.no_grad():
                    logits = self.model(xt)[0].float().cpu().numpy()
            probs = self._ensemble_probs(work) if self.members else grade_probs(logits[None], self.temperature)[0]
            p_ref = float(referable_prob(probs[None])[0])
            grade = int(probs.argmax())
            qs = q.status if q.status != "reject" else "good"
            ev, lesion_overlay, cross = lesion_evidence(None), None, None
            if with_lesions:
                ev, lesion_overlay = self._evidence(work)
                cross = cross_check(p_ref, self.threshold, ev["rule"] if ev.get("available") else None)
            dec = decide_with_evidence(qs, p_ref, self.threshold, self.review_margin, cross)
            return PredictionResult(
                "ok", processed=plain, gradcam=overlay_heatmap(plain, cam) if cam is not None else None, lesion_overlay=lesion_overlay,
                grade=grade, grade_name=grade_name(grade), grade_probs=[float(v) for v in probs], p_referable=p_ref,
                referable=p_ref >= self.threshold, decision=dec.to_dict(), advice=referral_advice(grade), lesion_evidence=ev, cross_check=cross,
                latency_ms=(time.time() - t0) * 1000, forced_despite_rejection=(q.status == "reject"), review_margin=float(self.review_margin or 0.0), **base)
        except Exception as e:
            return PredictionResult("error", f"Inference failed ({type(e).__name__}: {e})", original=img)
