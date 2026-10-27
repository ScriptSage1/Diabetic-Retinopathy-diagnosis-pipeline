# The system as built

From theory to the codebase: the architecture and data flow, what lives where in the repository, how to run everything from scratch, and the engineering decisions behind it.

## Architecture and data flow

The project has two halves that never run at the same time ({{ref:arch}}).

- **Offline, in the cloud.** Training and evaluation run as Kaggle notebooks on a free NVIDIA Tesla T4 GPU. Each notebook is a Python script built by `scripts/kaggle_submit.py`, which embeds the project's own `drscreen` package into it, so the notebook needs nothing but the attached datasets. It writes checkpoints (`best.pt`, `last.pt`), a calibration file (temperature, threshold, review margin, quality thresholds), `metrics.json`, `predictions.csv` and figures, which are downloaded with the Kaggle command-line tool.
- **Locally, on an ordinary computer.** The downloaded checkpoints live in `artifacts/`, the results in `results/`. The report builders turn the results into PDFs, and the RetinaLab application runs the models on new photographs. A CPU is sufficient: one analysis takes a few seconds.

{{fig:architecture.png|The project's real architecture (original diagram). Only components and connections that exist in the code are shown.|100|arch}}

### What happens when a photograph is analysed

One call, `Predictor.predict(image)` in `drscreen/predict.py`, performs the whole runtime pipeline. The interface, the PDF case report and the tests all use this single function, so there is exactly one implementation of the logic.

1. **Read and validate** the input (file path, array or image object), rejecting invalid inputs with a message.
2. **Quality gate** (`assess_quality`): *reject* returns "please retake the photograph" with no prediction; *enhance* applies CLAHE first.
3. **Preprocess** to the model's input (crop, pad, resize, mask, Ben Graham, ImageNet scaling).
4. **Classify**: the network (or the ensemble, with flips) produces logits; temperature scaling turns them into grade probabilities; the referable score is $p_2 + p_3 + p_4$.
5. **Explain**: Grad-CAM heat map from the first model.
6. **Lesion evidence**: segmentation, disc and fovea location, quadrant counts, the rule engine, and the cross-check with the classifier.
7. **Decide** (`fusion.decide_with_evidence`): retake, refer, no referral, or human review when the score lies within the review band around the threshold or the lesion rules suggest a case the classifier missed.
8. **Return** a `PredictionResult` holding every number and image above, plus a plain-language explanation built only from those values (`drscreen/explain.py`).

A short excerpt shows how directly the decision rule is written (`drscreen/fusion.py`):

```python
def decide(quality_status, p_ref, threshold, review_margin=0.0):
    if quality_status == "reject":
        return Decision("recapture", "image quality insufficient for automated analysis")
    if abs(p_ref - threshold) <= review_margin:            # too close to call
        return Decision("human_review", ...)
    if p_ref >= threshold:
        return Decision("refer", ...)
    return Decision("no_referral", ...)
```

## The repository

{{tab:repo|The repository, annotated. Only files that exist were listed; weights and data are excluded from version control.}}

| Path | Responsibility |
|---|---|
| `src/drscreen/preprocess.py` | reading, validating, cropping, masking, resizing, Ben Graham, tensor scaling |
| `src/drscreen/datasets.py` | finding label tables, caching images, duplicate groups, splits, augmentation, the PyTorch dataset |
| `src/drscreen/models.py` | building EfficientNet / the small baseline CNN, saving and loading checkpoints, softmax helpers |
| `src/drscreen/train.py` | the training loop: class weights, AdamW, warm-up + cosine schedule, early stopping, resume |
| `src/drscreen/metrics.py` | every metric, temperature fitting, threshold selection, bootstrap and paired-bootstrap intervals |
| `src/drscreen/pipeline.py` | the end-to-end classifier experiment run on Kaggle (data, train, evaluate, baselines, quality, Grad-CAM, latency) |
| `src/drscreen/quality.py` | the heuristic quality gate and the synthetic degradations used to test it |
| `src/drscreen/gradcam.py` | Grad-CAM and its weight-randomisation check |
| `src/drscreen/lesion.py`, `lesion_pipeline.py`, `lesion_eval.py` | lesion U-Net, locator, features, training and evaluation on IDRiD, e-ophtha and APTOS |
| `src/drscreen/rules.py`, `fusion.py` | ICDR rule engine, decision logic, cross-check, fusion |
| `src/drscreen/ensemble.py`, `eyepacs.py` | ensemble/TTA candidate selection; EyePACS reading and patient-level split |
| `src/drscreen/predict.py` | `Predictor`: the single runtime entry point |
| `src/drscreen/explain.py`, `casereport.py`, `ui_theme.py` | explanation text and HTML, PDF case report, interface design system |
| `src/drscreen/report.py` | the research-report builder (Markdown template + results → PDF) |
| `notebooks/kaggle/*.py` | Kaggle entry points: classifier training, lesion training, inference, EyePACS experiment |
| `scripts/` | Kaggle submission/monitoring/download, pre-set comparison rules, report and guide builders, UI screenshots, test summary |
| `app/app.py` | the RetinaLab Gradio interface |
| `sim/screening_simulation.py` | SimPy workflow simulation |
| `tests/` | {{val:test_summary.total|d}} automated tests (pytest) |
| `artifacts/` | checkpoints and calibration files (not in version control) |
| `results/` | `metrics.json`, `predictions.csv`, lesion features, simulation output, test summary |
| `report/` | research report, this guide, their sources and figures |

## Running the project end to end

The steps below were checked against the repository's README and scripts.

**1. Set up Python (once).**

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (use bin/activate elsewhere)
pip install -r requirements.txt
pip install -e .
pytest -q                       # all tests should pass
```

**2. Connect to Kaggle (once).** Run `kaggle auth login` (browser sign-in), accept the APTOS 2019 competition rules on kaggle.com (and the EyePACS rules for that experiment), and phone-verify the account (required for GPU use). Kaggle's sign-in expires after a few hours; `scripts/kaggle_status.py` reports this as `auth_expired`.

**3. Train and evaluate remotely.**

```bash
python scripts/kaggle_submit.py push --mode smoke    # a short sanity run first
python scripts/kaggle_submit.py push --mode full     # EfficientNet-B0 (also: robust, b3, lesion, lesion2, eyepacs, infer)
python scripts/kaggle_status.py --watch --max-minutes 60
python scripts/kaggle_download_outputs.py            # installs results, figures and checkpoints; verifies they load
```

**4. Apply the pre-set comparison rules (CPU, seconds).** `scripts/compare_robust.py`, `scripts/evaluate_ensemble.py`, `scripts/compare_lesion_v2.py` and `scripts/compare_eyepacs.py` read the downloaded outputs, apply the rule written in their docstring, record the decision in `results/metrics.json` and install the chosen artifacts.

**5. Build the documents.**

```bash
python scripts/test_summary.py            # runs the tests, saves results/test_summary.json
python scripts/build_report.py            # report/Final_Report.pdf
python scripts/make_guide_figures.py      # figures for this guide
python scripts/build_guide.py             # report/DR_Learning_Guide.pdf (needs Chrome or Edge)
```

**6. Run the interface.** `.\run_app.ps1` (or `python app/app.py`) starts RetinaLab at http://127.0.0.1:7860/. Upload a photograph, choose a model and press *Analyse image*.

::: warning Manual steps and external services
Kaggle access needs an account, accepted competition rules and a verified phone number; the free GPU quota (about 30 hours per week at the time of the project) limits how many experiments can run. The datasets are never redistributed with the project.
:::

## Design decisions and their trade-offs

The table lists the main decisions. The left column records what was decided, as documented in the code, configuration and project notes; the right column gives the reasoning and what it costs.

{{tab:decisions|Main engineering decisions, with the reasons and the remaining limitations.}}

| Decision | Why, and what it costs |
|---|---|
| Train on a free Kaggle T4 GPU, run the app locally | No local GPU setup is needed and runs are reproducible scripts; the price is a weekly GPU quota, a 12-hour session limit (hence checkpoints and resume) and an upload/download step. |
| EfficientNet-B0 at 384 pixels as the main model | Strong accuracy for its size, fast on a T4 and on a CPU; small lesions lose detail at 384 pixels. B3 at 448 pixels was tried later; the ensemble of both is the default. |
| Ben Graham colour normalisation | Reduces camera and lighting differences cheaply; the images look unnatural and noise is amplified. |
| Class weights with power 0.5 | Counteracts imbalance without letting the rare grades dominate; rare grades (3, 4) remain the weakest. |
| Checkpoint selection on validation QWK | QWK rewards getting close on the ordinal scale, matching the clinical meaning of grades; it is not the screening metric itself. |
| Threshold: highest value with validation sensitivity ≥ 0.90 | Screening values missed cases more than false alarms; the threshold is tuned to APTOS validation data and need not transfer. |
| Temperature scaling | Makes scores better calibrated with one parameter and no change to rankings; it is fitted on one validation set from one source. |
| Duplicate-aware split instead of patient split | The only option without patient identifiers; leaves possible same-patient leakage. |
| Heuristic quality gate | No quality-labelled data were available; it misses JPEG damage and over-exposure. |
| Lesion module trained on IDRiD | The only dataset with masks for all four lesion types; 54 training images from one camera, so it transfers imperfectly. |
| Rules written down before each comparison | Prevents choosing the model that happens to look best on the test set; occasionally rejects a model with appealing properties (the EyePACS model, lesion v2). |
| Gradio interface [39] | A real working application in pure Python; its components limit some visual choices, so the design is built with custom HTML and CSS on top. |
