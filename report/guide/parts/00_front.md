## About this guide {-}

<p class="lead">This document is a companion to the project. It teaches the science behind a diabetic retinopathy (DR) screening system, from how the eye works to how a neural network learns, and then shows, with the project's own saved results, what was actually built and how well it works.</p>

It is written for an undergraduate computer-science student who knows some Python but has not studied ophthalmology, image processing, deep learning or clinical statistics. Each concept is introduced when it is needed and answers four questions: *what is it, why do we need it, how does it work, and where does it appear in our project?* Mathematics is introduced intuition-first, with every symbol defined and a small worked example.

**What makes this a report as well as a tutorial.** Every number quoted about the project is read automatically from the project's saved result files (mainly `results/metrics.json`) when the PDF is built, so the text cannot drift from the evidence. Teaching examples use made-up, round numbers and are always marked *illustrative*. Published results from other studies are cited and kept separate from ours.

The guide uses a small set of boxes:

::: define A definition
A new term, defined from scratch.
:::

::: project In our project
Where the idea appears in the code, with file and function names that were checked against the repository.
:::

::: example An illustrative example
Teaching numbers, not project results.
:::

::: warning A caution
A common misunderstanding, a limitation, or a claim the evidence does not support.
:::

### How the guide is organised {-}

<div class="roadmap">
<div><b>Part I</b><span>Biology</span>The eye, the retina, fundus photographs, diabetic retinopathy, lesions and grading, and the screening question.</div>
<div><b>Part II</b><span>Images</span>Pixels and arrays, the preprocessing and augmentation the project really uses, the dataset and its splits.</div>
<div><b>Part III</b><span>Machine learning</span>Supervised learning, convolutional networks, EfficientNet, loss, back-propagation, training, Grad-CAM, lesion segmentation, rules, ensembles.</div>
<div><b>Part IV</b><span>The system</span>Architecture, data flow, repository map, how to run everything, design decisions.</div>
<div><b>Part V</b><span>Evaluation</span>Every metric the project uses, from the confusion matrix to calibration and bootstrap intervals.</div>
<div><b>Part VI</b><span>Results</span>The measured results, internal and external, with an honest interpretation.</div>
<div><b>Part VII</b><span>One image, end to end</span>A real photograph followed through the whole pipeline.</div>
<div><b>Part VIII</b><span>Reference</span>References, glossary and viva questions with answers.</div>
</div>

### The project at a glance {-}

The project builds and evaluates a **screening** system: given one colour photograph of the back of the eye, it estimates the severity of diabetic retinopathy on the five-level international scale and decides whether the eye should be **referred** to an eye specialist (grade 2, *moderate*, or worse). Around that classifier it adds an image-quality check, visual explanations, a lesion detector with a rule engine that states its reasons, calibrated scores, an interface and a PDF case report.

The original plan (`ROADMAP.md`) described a learning path of 19 modules. Not everything in it was built, and some things were built that it did not foresee. The table separates *intended*, *implemented*, *executed* and *verified*, using the repository as evidence.

::: takeaway What the evidence supports, in one paragraph
On held-out images from the same source as the training data (APTOS 2019) the classifier separates referable from non-referable eyes very well: sensitivity {{val:internal_holdout.screening.sensitivity|.3f}} and specificity {{val:internal_holdout.screening.specificity|.3f}} at a threshold fixed on validation data, ROC-AUC {{val:internal_holdout.screening.roc_auc|.3f}}. On a small independent dataset from a different camera and grading protocol (IDRiD, {{val:external_idrid.n_evaluated|d}} images) the sensitivity falls to {{val:external_idrid.rows.0.sensitivity|.3f}}. The system is a well-engineered and carefully evaluated research prototype; it is not clinically validated and should not be used to make decisions about real patients.
:::

{{tab:status|Status of each component, judged from the repository and its saved outputs (not from the roadmap).}}

| Component | Implemented | Executed on real data | Verified by | Status |
|---|---|---|---|---|
| Data loading, preprocessing, duplicate-aware splits | `drscreen/datasets.py`, `preprocess.py` | yes, APTOS 2019 on Kaggle | unit tests, leakage assertions | <span class="status s-done">done</span> |
| EfficientNet-B0 classifier (standard and robustness-augmented), B3, ensemble with flip test-time augmentation | `models.py`, `train.py`, `ensemble.py` | yes, Kaggle T4 GPU | test-split metrics with bootstrap intervals | <span class="status s-done">done</span> |
| Temperature scaling and validation-chosen threshold | `metrics.py`, `pipeline.py` | yes | calibration error before and after | <span class="status s-done">done</span> |
| Heuristic image-quality gate | `quality.py` | yes | synthetic degradations only | <span class="status s-part">heuristic</span> |
| Grad-CAM explanations | `gradcam.py` | yes | weight-randomisation sanity check | <span class="status s-done">done</span> |
| Lesion segmentation (U-Net), disc and fovea locator, ICDR rule engine, fusion, cross-check | `lesion.py`, `rules.py`, `fusion.py` | yes (IDRiD-trained) | IDRiD test AUPR/Dice; APTOS agreement | <span class="status s-done">done</span> |
| Lesion model v2 (extra e-ophtha data) | `lesion_pipeline.py` | yes | pre-set comparison | <span class="status s-opt">not adopted</span> |
| EyePACS-augmented classifier | `notebooks/kaggle/train_eyepacs.py` | yes | pre-set comparison | <span class="status s-opt">optional model</span> |
| External check on IDRiD grading images | `lesion_eval.py` | yes (about 100 images) | bootstrap intervals | <span class="status s-done">done</span> |
| Retinal vessel segmentation (roadmap module 11) | not implemented | no | none | <span class="status s-no">not built</span> |
| Detection of venous beading, IRMA, new vessels | not implemented | no | none | <span class="status s-no">not built</span> |
| Workflow simulation | `sim/screening_simulation.py` | yes | Little's law check | <span class="status s-done">done (assumption-driven)</span> |
| Interface (RetinaLab) and PDF case report | `app/app.py`, `explain.py`, `casereport.py` | yes, locally | end-to-end and layout tests | <span class="status s-done">done</span> |
| Teaching notebooks M00, M01 | generated | **not executed** | none | <span class="status s-part">incomplete</span> |
| Clinical or prospective validation | out of scope | no | none | <span class="status s-no">not done</span> |
