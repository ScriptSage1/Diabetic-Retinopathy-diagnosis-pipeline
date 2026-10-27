# Project status

_Last updated: 2026-10-25 (evening). Phase 1 (core pipeline) and phase 2 (upgrades, EyePACS experiment, new interface) are complete. Nothing is running on Kaggle._

## State
All deliverables are built, run on real data, tested and documented. `pytest -q` passes (92 tests), the report builds with no unresolved placeholders (`report/Final_Report.pdf`, 35 pages), and the application runs the real checkpoints.

| Item | Status |
|---|---|
| Data pipeline, leakage-aware splits, EfficientNet-B0 training on Kaggle T4 (standard + robustness-augmented) | done |
| EfficientNet-B3 (448 px) and B0+B3 ensemble with flip test-time augmentation | done; ensemble adopted as the app default by a pre-set rule (report 15.2) |
| EyePACS experiment (patient-level split, mixed training then APTOS fine-tune) | done; better external-domain numbers, failed the pre-set in-domain non-inferiority check, **installed as optional model, not default** (report 15.3) |
| Lesion segmentation v1 (IDRiD), locator, ICDR rule engine, fusion, IDRiD external check | done (report 14) |
| Lesion model v2 (IDRiD + e-ophtha, partial labels, acquisition augmentation) | done; **not adopted** by its pre-set rule (IDRiD AUPR drop > 0.03) although it halves false detections and raises rule-grade agreement; weights kept in `artifacts/lesion_v2` (report 14.7) |
| Metrics (sens/spec/PPV/NPV, ROC/PR AUC, QWK, ECE/Brier, bootstrap CIs; thresholds on validation only), baselines, quality gate, Grad-CAM, robustness comparison | done (report 10-15) |
| SimPy workflow simulation | done |
| Interface ("RetinaLab" Gradio workstation: viewer with Original/Preprocessed/Grad-CAM/Lesion views, zoom/pan/expand, threshold chart, evidence tabs, PDF case report) | done; layout checked at 1920/1366/1024/820/390 px (`scripts/ui_screenshots.py`) |
| Report PDF + Markdown + template; README | done |
| Learning guide PDF (`report/DR_Learning_Guide.pdf`, source in `report/guide/`) | done; regenerated from the saved results |
| Notebooks | M02 executed; **M00, M01 generated but not executed** (need the 8 GB image set; deliberately not downloaded) |

## Kaggle (account `candyguy`)
Kernels (all complete): classifier B0 standard/robust, B3 robust, logits (`dr-classifier-logits`), lesion v1/v2 and feature kernels, `dr-eyepacs-experiment`, smoke/probe kernels. Kaggle OAuth tokens expire after a few hours: `scripts/kaggle_status.py` reports `auth_expired`; fix with `kaggle auth login --force`. Large downloads go to `%LOCALAPPDATA%\drscreen\`.

## Headline results
Internal holdout (APTOS test n = 519, threshold chosen on validation), standard B0: sensitivity 0.932 (0.896-0.964), specificity 0.955, ROC-AUC 0.987, QWK 0.910. B0+B3+TTA ensemble (default in the app): sensitivity 0.941, specificity 0.949, ROC-AUC 0.991, QWK 0.921 (gain over robust B0 not statistically distinguishable from zero).
**External IDRiD (n = 100-103): the APTOS-only models reach sensitivity ~0.77-0.81 at the APTOS-derived threshold; the roadmap's external target (> 90%) is NOT met.** The EyePACS-augmented model raises IDRiD ROC-AUC 0.876 -> 0.917 (interval excludes 0) and sensitivity to 0.83, still below 0.90.
Lesion v1 segmentation on IDRiD test, AUPR: disc 0.979, haemorrhage 0.569, soft exudate 0.598, microaneurysm 0.446, hard exudate 0.332. Calibrated rule engine on APTOS: QWK 0.527 (v2: 0.809); fusion with the classifier gives no demonstrated improvement.

## Known limits (stated in the report)
No APTOS patient IDs (EyePACS split is by patient); results are internal except small external checks where performance drops; thresholds do not transfer across datasets; lesion detections on APTOS are an unvalidated domain transfer; venous beading/IRMA/neovascularisation are not detected (rule engine cannot assign grade 4); heuristic quality gate is validated only on synthetic degradations; several variants were compared on the same test/IDRiD data (selection effects); EyePACS labels are noisy; third-party Kaggle copies of IDRiD and e-ophtha; not a medical device.

## Remaining / optional
- Fill in author and institution in `report/meta.json`, then `python scripts/build_report.py` (placeholders were left on purpose).
- Local CUDA torch is not installed on purpose; training and the demo do not need it.
- Execute M00/M01 on Kaggle or locally with the data if wanted.
- If the user prefers the EyePACS-augmented model as the default, change the menu order in `app/app.py` (`MODEL_FILES`); the pre-set rule did not recommend it.
- A git repository exists but nothing is committed (`.gitignore` excludes credentials, data and weights).

## Last verified execution status
92 tests pass; app served and driven end to end (upload, analyse, model switch, zoom/reset, New image, error path); report builds with 0 unresolved placeholders.
