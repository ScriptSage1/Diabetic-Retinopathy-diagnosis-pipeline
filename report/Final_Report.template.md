## Abstract

Diabetic retinopathy (DR) is a leading preventable cause of vision loss among people with diabetes, and screening programmes depend on grading large numbers of retinal photographs. This project is an academic prototype of an automated screening pipeline. An ImageNet-pretrained EfficientNet-B0 [6] is fine-tuned on the APTOS 2019 dataset [11] to predict the five International Clinical DR (ICDR) severity grades [5], from which a referable-DR decision (grade 2 or higher) is derived. The pipeline adds a heuristic image-quality gate, Grad-CAM visual explanations [7], temperature-scaled confidence [8], a lesion-evidence pathway (an IDRiD-trained lesion segmentation network and optic-disc/fovea locator feeding an ICDR rule engine that states its reasons), a discrete-event simulation of a screening workflow, and a Gradio demonstration application. Training and evaluation were run remotely on a Kaggle GPU ({{val:environment.gpu}}). On the internal held-out test split of {{val:internal_holdout.n_samples|d}} images the referable-DR sensitivity was {{val:internal_holdout.screening.sensitivity|.3f}} and specificity {{val:internal_holdout.screening.specificity|.3f}} at a threshold chosen on validation data only, with a ROC-AUC of {{val:internal_holdout.screening.roc_auc|.3f}} and a quadratic weighted kappa of {{val:internal_holdout.grades.qwk|.3f}} on the five grades. These are **internal** results on data from the same source as the training data; they are not independent external validation and not clinical validation. On a small independent test on IDRiD ({{val:external_idrid.n_evaluated|d}} images, a different camera and grading protocol) the same classifier reached a sensitivity of only {{val:external_idrid.rows.0.sensitivity|.3f}} (specificity {{val:external_idrid.rows.0.specificity|.3f}}, ROC-AUC {{val:external_idrid.rows.0.roc_auc|.3f}}), so the internal figures overstate what should be expected on other data. Follow-up experiments, each judged by a decision rule written down before its results were seen, added an EfficientNet-B3 plus flip test-time-augmentation ensemble (adopted as the application default), a robustness-augmented model (adopted for the degraded-image case), a second lesion model trained with an additional dataset (not adopted) and a model trained with additional EyePACS data [24], which raised the IDRiD ROC-AUC from {{val:eyepacs_experiment.models.efficientnet_b0_robust.idrid_test.roc_auc|.3f}} to {{val:eyepacs_experiment.models.efficientnet_b0_eyepacs.idrid_test.roc_auc|.3f}} but failed the pre-set in-domain non-inferiority criterion and is therefore offered only as an optional model.

> This project is an academic research prototype for diabetic retinopathy screening. Its outputs are not a medical diagnosis and must not be used as a substitute for qualified clinical assessment.

# 1. Introduction and problem statement

Diabetic retinopathy is a microvascular complication of diabetes in which long-standing hyperglycaemia damages the small blood vessels of the retina. Early disease is usually asymptomatic, while advanced disease (severe non-proliferative and proliferative DR) threatens sight and is treatable with laser photocoagulation, intravitreal anti-VEGF injection or vitreoretinal surgery when detected in time. Screening therefore aims to sort patients quickly into those who can be re-screened routinely and those who need an ophthalmologist. The international severity scale of Wilkinson et al. [5] defines five levels: no apparent retinopathy (0), mild (1), moderate (2) and severe (3) non-proliferative DR, and proliferative DR (4). In this work **referable DR** means grade 2 or higher, the definition stated in the project roadmap.

The problem addressed here is: given a colour fundus photograph, (a) decide whether the image is of sufficient quality to analyse, (b) estimate the ICDR grade, (c) derive a referable / non-referable screening decision, and (d) present the result with a visual explanation and honest, clinically interpretable performance statistics.

# 2. Motivation: automated screening and the Indian context

India has a very large population living with diabetes. The ICMR-INDIAB national cross-sectional study reported a diabetes prevalence of 11.4% (95% CI 10.2–12.5) in its analysed sample of 107,119 individuals [4]. DR prevalence in the Chennai Urban Rural Epidemiology Study eye study was 17.6% (95% CI 15.8–19.5) among people with diabetes, and 20.8% among those with previously known diabetes [3]. The Chennai study is a single-city study, so it should not be read as a national figure. Because the number of people needing periodic retinal examination far exceeds the number of available eye specialists, particularly outside cities, automated grading that can run on photographs taken in primary-care or camp settings is attractive. In the United States, a pivotal trial of an autonomous system (IDx-DR) in primary-care offices reported a sensitivity of 87.2% and specificity of 90.7% for more-than-mild DR against a reading-centre reference [2]; this shows that regulated autonomous screening is feasible, but it concerns a different system, population and protocol than this prototype and is cited for context only, not as a benchmark that this project is claimed to meet.

# 3. Objectives and scope

1. Build a reproducible, tested end-to-end pipeline from images to a screening decision.
2. Train and evaluate an EfficientNet-B0 classifier on all five grades and derive referable DR from the grade probabilities.
3. Report evaluation metrics in a form a clinician can interpret: sensitivity, specificity, predictive values, ROC/PR, confusion matrices, quadratic weighted kappa, calibration, and bootstrap confidence intervals.
4. Provide a quality gate, visual explanations, a decision-support demo, and a workflow simulation.
5. Give a clear reason for each result: detect and localise lesions with a model trained on lesion annotations (IDRiD), turn the ICDR criteria into an explicit rule engine that lists which criteria are met, not met or **not assessable**, and cross-check it against the classifier.

**In scope and implemented:** items 1–5 above. **Out of scope / not implemented:** detection of venous beading, intraretinal microvascular abnormalities (IRMA) and neovascularisation (so the rule engine can never assign grade 4), patient-level data separation (no identifiers available), and clinical validation. These limits are discussed in Section 14 and Section 19.

# 4. Related work

Gulshan et al. [1] trained a deep convolutional network on retinal photographs graded by ophthalmologists and reported, for referable DR on EyePACS-1 and Messidor-2, sensitivities of 90.3% and 87.0% with specificities of 98.1% and 98.5% at a high-specificity operating point, with ROC-AUCs of 0.991 and 0.990. Their referable definition also included referable macular oedema, and they used far larger training data than APTOS; the numbers are therefore not directly comparable to this project. The IDx-DR pivotal trial [2] evaluated an autonomous system prospectively in primary care. EfficientNet [6] scales depth, width and resolution jointly and is a common backbone for fundus-image transfer learning. Grad-CAM [7] produces coarse class-discriminative localisation maps from the gradients of a convolutional network, and the sanity checks of Adebayo et al. [18] motivate testing whether a saliency map depends on the learned weights. Guo et al. [8] showed that modern networks are often over-confident and that temperature scaling is a simple post-hoc remedy. Ben Graham's colour normalisation, popular in the Kaggle DR competitions [17], subtracts a local Gaussian-blurred average to reduce illumination differences between cameras. The IDRiD dataset [14] provides pixel-level lesion masks, optic-disc and fovea locations and image-level grades, and is used here to train and test the lesion-evidence module.

# 5. Datasets, labels and splits

## 5.1 Data source and access

The primary dataset is APTOS 2019 Blindness Detection [11], released by the Asia Pacific Tele-Ophthalmology Society as a Kaggle competition. Access to the files requires accepting the competition rules on kaggle.com; this was done by the project owner, and the data were attached to the Kaggle notebook as a competition source. Only the labelled training partition (the competition's test labels are not public) is used, and it is re-split by this project.

{{table:splits}}

{{figure:class_distribution.png|Figure 1. Images per ICDR grade in each split. Grades 3 and 4 are rare, so the split is stratified on grade and the loss is class-weighted.}}

## 5.2 Labels

Each image has a single grade from 0 to 4. Referable DR is derived as grade ≥ 2 (positive class). Image-level grades only are available: **no lesion annotations, no patient identifiers, no eye laterality and no image-quality labels** are provided.

## 5.3 Splits and leakage prevention

The labelled images were split into train / validation / test with a stratified (on grade), seeded, group-aware procedure. Images are grouped when they are byte-identical (MD5) or near-duplicates (16×16 difference-hash, Hamming distance ≤ 6), and each group is kept inside one split; an assertion verifies that no MD5 or group spans two splits. {{val:dataset.n_images_in_merged_groups|d}} images were merged into groups with another image. The split manifest (image id, grade, split, group) is saved with the artifacts for exact reproduction.

**Limitation.** {{val:dataset.limitation}}

## 5.4 IDRiD (lesion evidence and external check)

The Indian Diabetic Retinopathy Image Dataset (IDRiD) [14] is released under CC BY 4.0. Its official release has three parts, all used here: **segmentation** (54 training and 27 test images with pixel masks for microaneurysms, haemorrhages, hard exudates, soft exudates and the optic disc), **localization** (optic-disc and fovea centres for 413 training and 103 test images) and **disease grading** (the same 516 images with a DR grade 0–4 and a macular-oedema risk label). The lesion network is trained on the 54 segmentation-training images (10 of them held out for model and threshold selection), the locator on the localization-training images, and both are evaluated only on the official test images. A missing mask file is read as "no lesion of that type in the image", the dataset's convention. The copy used is a third-party re-upload of the official release on Kaggle; I could not verify that it is byte-identical to the original distribution. IDRiD was photographed with a single camera type and is therefore not representative of APTOS's mixture of cameras.

## 5.5 EyePACS (additional training and external-domain data)

For the experiment of Section 15.3 the Kaggle *Diabetic Retinopathy Detection* (EyePACS) competition data [24] were used. Access requires accepting the competition rules, which the project owner did; the images were read inside the Kaggle notebook (never downloaded to a personal computer or redistributed), directly from the split archive. Images carry one grade per eye from 0 to 4 (a single reader per image, from a variety of clinics and cameras; the labels are noisier than those of APTOS and IDRiD). The patient identifier is encoded in the file name, so the split is **by patient**: both eyes of a patient are always in the same split. A training pool took every grade 1-4 image of the training patients plus a random sample of grade 0, and a held-out sample of test patients was kept with its natural grade mix. Every EyePACS image was checked against all APTOS validation and test images and the IDRiD grading test images by exact hash and perceptual (difference-hash) similarity so that no photograph could appear on both sides of an evaluation.

{{table:eyepacs_data}}

# 6. Image preprocessing and augmentation

Each image is read with OpenCV (corrupt, empty, grayscale-only or constant images are rejected and logged, never silently dropped), then: (1) cropped to the bounding box of the non-black retina region, (2) padded to a square, (3) resized to {{val:dataset.img_size|d}}×{{val:dataset.img_size|d}} pixels (area interpolation when shrinking), (4) masked with a circular field-of-view mask. The model input is further normalised with the Ben Graham transform (4·I − 4·G_σ∗I + 128 with σ = size/30 and a slightly tighter circular mask to suppress the edge ring), then ImageNet mean/standard-deviation normalisation. The preprocessed array is cached once per run.

{{figure:preprocessing_examples.png|Figure 2. One example per grade: original photograph, cropped/masked/resized image, Ben Graham-normalised model input, and the green channel (in which red lesions have the highest contrast).}}

Augmentation is applied to **training images only**: random rotation (0–360°), horizontal/vertical flips, scale 0.92–1.08, brightness and contrast jitter, and gamma. Validation and test images receive no augmentation and no test-time augmentation.

# 7. Model architecture and training

The classifier is EfficientNet-B0 [6] (timm implementation [12] on PyTorch [13], ImageNet-pretrained weights) with a five-way output layer and dropout {{val:run.config.drop_rate|g}}. The five-grade softmax is the primary output; the referable probability is P(grade 2) + P(grade 3) + P(grade 4). Training used AdamW [19] (learning rate {{val:run.config.lr|g}}, weight decay {{val:run.config.weight_decay|g}}), a one-epoch linear warm-up followed by cosine decay, batch size {{val:run.config.batch_size|d}}, automatic mixed precision, gradient clipping, and class-weighted cross-entropy with weights proportional to the square root of inverse class frequency. The best checkpoint and early stopping used the validation quadratic weighted kappa (patience {{val:run.config.patience|d}}). A checkpoint containing model, optimiser, scheduler and scaler state was written every epoch.

{{table:environment}}

Training ran for {{val:run.epochs_completed|d}} epochs (best epoch {{val:run.best_epoch|d}}; stopping reason: {{val:run.stop_reason}}) in {{val:run.train_seconds|.0f}} seconds of training time. The checkpoint was reloaded twice and produced identical logits (maximum absolute difference {{val:checkpoint_reload_check.max_abs_diff|.2e}}).

{{figure:learning_curves.png|Figure 3. Training and validation loss, validation quadratic weighted kappa and validation referable ROC-AUC per epoch. A widening gap between training and validation loss indicates over-fitting.}}

**Baselines** trained on the same split for comparison: a majority-class prior, hand-crafted image statistics with gradient boosting, and a small four-block CNN trained from scratch. They quantify how much of the performance comes from transfer learning rather than from trivial image statistics.

# 8. Evaluation protocol

All headline results are computed on the held-out **test split**, which is never used for training, checkpoint selection, temperature fitting, threshold selection, or quality-gate calibration. The temperature, the operating threshold and the quality thresholds are fitted on validation (respectively training) data only.

{{table:threshold_selection}}

The operating threshold on P(referable) was chosen on the validation split as the highest threshold meeting a sensitivity target of 0.90 (a design choice reflecting that missed referable cases are the costly error in screening); test performance at that fixed threshold is then reported without re-tuning. Results at the default threshold of 0.5 are also stored in `results/metrics.json`. The resulting evaluation is an **internal holdout** evaluation: test images come from the same collection, cameras and labelling process as the training images.

Confidence intervals are 95% percentile bootstrap intervals [10] ({{val:internal_holdout.confidence_intervals._method.n_boot|d}} replicates, seed {{val:internal_holdout.confidence_intervals._method.seed|d}}) in which test images are resampled with replacement and every metric is recomputed on the same resample; replicates in which a metric is undefined (for example no positive case drawn) are skipped. Images are treated as independent. Because APTOS has no patient identifiers, the intervals may be too narrow if both eyes of a patient are in the test set.

# 9. Evaluation metrics in plain language

Let TP be referable cases flagged referable, FN referable cases passed as non-referable (**missed cases**), FP non-referable cases flagged referable (**false alarms**), and TN non-referable cases passed.

- **Sensitivity** = TP / (TP + FN). Of the truly referable patients, the share the system flagged. It is the headline screening metric because a missed referable case is the costly error.
- **Specificity** = TN / (TN + FP). Of the truly non-referable patients, the share correctly reassured. Low specificity floods reviewers with false alarms.
- **Precision (PPV)** = TP / (TP + FP). Of the flagged cases, the share that are truly referable. It depends on prevalence.
- **Negative predictive value** = TN / (TN + FN). How reassuring a "no referral" result is.
- **ROC-AUC.** The probability that a random referable case receives a higher score than a random non-referable case; it does not depend on a threshold.
- **PR-AUC (average precision).** Summarises precision against recall; more informative than ROC-AUC when positives are the minority [23].
- **Balanced accuracy** = (sensitivity + specificity) / 2, and **F1** = 2·PPV·sensitivity / (PPV + sensitivity).
- **Quadratic weighted kappa (QWK)** [9]. Agreement between predicted and true grades beyond chance, where a disagreement of d grades is penalised in proportion to d². Calling a grade-4 eye grade 0 costs sixteen times more than a one-grade error, which matches clinical intuition.
- **Dice / IoU / AUPR (lesion segmentation).** Dice (equal to pixel-level F1) = 2·TP / (2·TP + FP + FN) and IoU = TP / (TP + FP + FN) measure overlap between predicted and annotated lesion pixels at a chosen threshold; AUPR is the area under the pixel precision-recall curve over all thresholds (the IDRiD challenge metric). Lesion pixels are extremely rare, so a model can reach 99% pixel accuracy while detecting nothing; accuracy is therefore not reported for segmentation.
- **Calibration.** Expected calibration error (ECE) is the bin-size-weighted average gap between confidence and observed accuracy over 15 equal-width bins; the Brier score [22] is the mean squared error of the referable probability. A softmax output is a model score, not automatically a calibrated probability [8].

A trade-off is unavoidable: lowering the threshold raises sensitivity and lowers specificity. Which operating point is acceptable is a clinical and programme decision that this prototype does not make.

# 10. Quantitative results (internal held-out test split)

{{table:setting}}

{{table:counts}}

{{table:main_metrics}}

{{table:grade_metrics}}

**Reading these results.** At the validation-selected threshold ({{val:internal_holdout.screening.threshold|.3f}}) the system flagged {{val:internal_holdout.screening.sensitivity|.3f}} of the {{val:internal_holdout.screening.n_positive|d}} truly referable test images (so {{val:internal_holdout.screening.fn|d}} referable cases were missed) and correctly passed {{val:internal_holdout.screening.specificity|.3f}} of the {{val:internal_holdout.screening.n_negative|d}} non-referable images ({{val:internal_holdout.screening.fp|d}} false alarms). The sensitivity target of 0.90 was set on validation data; the test sensitivity is not tuned and is reported with its bootstrap interval. These are internal holdout numbers on a single-source dataset. The project roadmap states a target for *external* data (sensitivity above 90% and specificity above 85%); that target is **not met** on the only independent check available (Section 14.5: sensitivity {{val:external_idrid.rows.0.sensitivity|.3f}}, 95% CI {{val:external_idrid.classifier_ci.sensitivity.lo|.3f}} to {{val:external_idrid.classifier_ci.sensitivity.hi|.3f}} on {{val:external_idrid.n_evaluated|d}} IDRiD images), and the internal numbers must not be read as evidence that it would be. Published systems (for example [1] and [2]) used different data, definitions and protocols, so these numbers must not be ranked against them.

Referable-versus-non-referable screening is much stronger than fine-grained grading. The quadratic weighted kappa is {{val:internal_holdout.grades.qwk|.3f}}, but the macro-averaged F1 over the five grades is only {{val:internal_holdout.grades.macro_f1|.3f}} and the balanced five-grade accuracy {{val:internal_holdout.grades.balanced_accuracy_5|.3f}}: the classifier recognises grade 0 almost perfectly (recall {{val:internal_holdout.grades.per_grade.0.recall|.3f}}) but is weaker on mild disease (grade 1 recall {{val:internal_holdout.grades.per_grade.1.recall|.3f}}), severe NPDR (grade 3 recall {{val:internal_holdout.grades.per_grade.3.recall|.3f}}, only {{val:internal_holdout.grades.per_grade.3.support|d}} test images) and proliferative DR (grade 4 recall {{val:internal_holdout.grades.per_grade.4.recall|.3f}}, {{val:internal_holdout.grades.per_grade.4.support|d}} images). The errors are mostly one or two grades apart (Figure 5), which is why QWK stays high, and many proliferative cases are graded as moderate: such an eye is still flagged referable, but the *urgency* would be under-stated. Grades 3 and 4 are rare in the data and the intervals for those rows are wide.

For reference, the validation-split numbers are stored separately in `results/metrics.json` under `validation_split_reference` and are not unbiased because that split was used for model selection.

# 11. Curves, confusion matrices and calibration

{{figure:confusion_referable.png|Figure 4. Referable-DR confusion matrix on the test split. The positive class is referable DR. The bottom-left cell (FN) counts missed referable cases; the top-right cell (FP) counts false alarms.}}

{{figure:confusion_5grade.png|Figure 5. Five-grade confusion matrix: counts (left) and row-normalised recall per true grade (right). Errors concentrated next to the diagonal are one-grade disagreements; errors far from it are the ones QWK penalises most.}}

{{figure:roc_pr.png|Figure 6. ROC curve (left) and precision-recall curve (right) for referable DR on the test split; the dot marks the validation-selected operating point.}}

{{figure:threshold_sweep_validation.png|Figure 7. Sensitivity, specificity and precision against the decision threshold on the validation split, with the chosen threshold marked. This is where the operating point was selected.}}

## Calibration

{{table:calibration}}

Temperature scaling (T = {{val:validation_threshold_selection.temperature|.2f}}, fitted on validation) reduced the five-grade top-label ECE from {{val:internal_holdout.calibration_uncalibrated.ece_top_label_5grade|.3f}} to {{val:internal_holdout.calibration.ece_top_label_5grade|.3f}} and the validation NLL from {{val:validation_threshold_selection.val_nll_before|.3f}} to {{val:validation_threshold_selection.val_nll_after|.3f}}. The referable-probability ECE was already small before scaling ({{val:internal_holdout.calibration_uncalibrated.ece_referable|.3f}}) and improved only slightly ({{val:internal_holdout.calibration.ece_referable|.3f}}). With 519 test images and most predictions near 0 or 1, the middle bins of the reliability diagram hold few images and are noisy. The probabilities are better calibrated on this internal split; they are not clinically validated risks.

{{figure:reliability.png|Figure 8. Reliability diagram for P(referable) on the test split before and after temperature scaling. Points on the dashed diagonal indicate that predicted probabilities match observed frequencies; grey bars show how many images fall in each bin, and sparsely populated bins are noisy.}}

# 12. Grad-CAM explanations

Grad-CAM [7] was implemented from first principles with forward and backward hooks on the last convolutional feature map (before global pooling): the map is the ReLU of the sum of feature maps weighted by their spatially averaged gradients of the predicted-grade score, up-sampled to the input size and masked to the field of view.

{{figure:gradcam_examples.png|Figure 9. Grad-CAM overlays for one correctly and one incorrectly classified test image per grade (where available). Warm colours mark regions that increased the model's score.}}

{{figure:error_examples.png|Figure 10. The most confident false negatives and false positives at the operating threshold, with Grad-CAM overlays.}}

**Interpretation limits.** A Grad-CAM map shows where this particular network's output is sensitive. It is not evidence that a lesion is present, that the model reasons as a clinician does, or that a highlighted region is pathological. As a sanity check, the map of the trained model was compared with that of a weight-randomised copy on three images (Spearman correlation {{val:gradcam.randomization_spearman_mean|.3f}}; values near zero indicate that the map depends on the learned weights); this is a weak, indicative test on a handful of images.

# 13. Image-quality gate

`quality.py` implements a **heuristic** gate: sharpness (variance of the Laplacian of the green channel inside the eroded field of view at a fixed 512-pixel scale), mean illumination, contrast, saturation, and field-of-view completeness. It returns `good`, `enhance` (contrast enhancement with CLAHE [20] is applied before analysis) or `reject`, with a reason. It is **not** a model trained on genuine quality labels such as EyeQ; thresholds are set from training-set percentiles and are therefore distribution-relative, not clinically validated.

{{table:quality_clean}}

{{table:quality_degradation}}

{{figure:quality_degradation.png|Figure 11. Share of test images flagged by the gate after synthetic degradations at three severities. The dashed line is the flag rate on unmodified images.}}

{{figure:quality_examples.png|Figure 12. One test image under each synthetic degradation (severity 2) with the gate's verdict.}}

On the {{val:quality.clean.n|d}} unmodified test images the gate marked {{val:quality.clean.good|d}} good, {{val:quality.clean.enhance|d}} enhance and {{val:quality.clean.reject|d}} reject. The synthetic-degradation chart shows what it can and cannot see: blur, darkness and low contrast are caught at nearly every severity, a severe top-edge crop is caught only at the highest severity, and **over-exposure and JPEG compression are essentially invisible to it**. The gate's response to synthetic degradations shows only that it reacts to those specific, artificial defects; it says nothing about agreement with human graders of real image quality.

The table below shows why the gate matters. It uses a grade-stratified subset (so its absolute numbers are not comparable with the headline test results). Without the gate, a blurred image is silently classified as healthy (the sensitivity collapses), while the gate rejects every one of them. Conversely, heavy JPEG compression makes the classifier call almost every image referable (specificity near zero) and the gate does not notice, so this is a real robustness weakness of the classifier that the gate does not cover.

{{table:quality_perf}}

On clean test images, running the full gated pipeline (contrast enhancement for 'enhance' images, rejected images counted as referred) gave a slightly lower sensitivity and specificity than the classifier alone (Section 15), so the enhancement step did not help on this data; its benefit, if any, is for genuinely poor images and was not demonstrated here.

# 14. Lesion evidence, rule engine and fusion

**Why.** A classifier's grade has no explanation attached. The lesion pathway produces the evidence a clinician looks for: where microaneurysms, haemorrhages and exudates are, how many, and in which part of the retina, and then applies the ICDR criteria to that evidence in code so that the reason for a grade can be read.

## 14.1 Lesion segmentation network

A U-Net with an ImageNet-pretrained EfficientNet-B0 encoder and a full-resolution output stem predicts five masks (microaneurysm MA, haemorrhage HE, hard exudate EX, soft exudate SE, optic disc OD) as independent sigmoid channels. Each image is cropped to the retina, padded and resized to a 1280-pixel canvas (so scale is normalised by retina size, not by camera resolution) and Ben Graham-normalised. Training uses random 512×512 crops, 60% centred on a lesion pixel, flips and 90° rotations, a class-weighted binary cross-entropy plus soft-Dice loss, and mixed precision. Pixel thresholds were chosen on the 10 validation images; **all numbers below are on the 27 official IDRiD test images**, which were never used for training or selection.

{{table:lesion_seg}}

{{figure:lesion_pr_idrid_test.png|Figure 11a. Pixel-level precision-recall curves per lesion type on the IDRiD test images.}}

{{figure:lesion_idrid_test_examples.png|Figure 11b. IDRiD test images: photograph, expert annotation and model prediction (red = MA, orange = HE, yellow = EX, cyan = SE).}}

Small structures (microaneurysms) are the hardest at this resolution; the table shows the measured performance and should be read as such. Only the held-out IDRiD performance measures lesion detection.

## 14.2 Optic-disc and fovea locator, and quadrants

An EfficientNet-B0 regressor predicts the optic-disc and fovea centres. They define the frame for the quadrant analysis: the line from the disc through the fovea is the horizontal axis, a perpendicular through the fovea is the vertical axis, and "superior" is the side that is up in the image. This is an approximation of the clinical quadrants for an upright photograph, not a validated definition. Distances are expressed in optic-disc diameters (measured from the predicted disc mask).

The regressor alone generalised only partly to APTOS: in visual checks of APTOS overlays its optic-disc estimate was often visibly off the disc (APTOS has no disc annotations, so this could not be quantified). The deployed method therefore takes the disc position from the **segmented disc** (Dice {{val:lesion_evidence.segmentation_idrid_test.per_lesion.OD.dice|.3f}} on IDRiD test) and uses the regressor's fovea only when its disc estimate agrees with the segmented disc to within one disc diameter; otherwise the fovea is placed by an anatomical prior (about 2.5 disc diameters from the disc toward the centre of the retina). The prior was needed for {{val:lesion_evidence.locator_final_idrid_test.share_using_anatomical_prior|.1%}} of the IDRiD test images and for {{val:lesion_fusion.aptos_locator_fallback_share|.1%}} of all APTOS images.

{{table:locator}}

## 14.3 ICDR rule engine

`rules.py` turns the detected lesions into a grade with reasons. The rules are: no lesion found, grade 0; microaneurysms only, grade 1; any haemorrhage, hard exudate or soft exudate (more than microaneurysms alone) grade 2; haemorrhage-like regions in all four quadrants at or above a per-quadrant count (the '4' of the 4-2-1 rule), grade 3. A macular-involvement indicator is raised when hard exudates lie within one disc diameter of the estimated fovea. For every image the engine prints which criteria were met, not met, or **not assessed**: venous beading, IRMA and neovascularisation cannot be assessed because no detector for them exists in this project, so **the rule engine can never assign grade 4**, and a rule-derived grade 3 rests on the haemorrhage criterion alone. Two threshold sets are evaluated: the literal ICDR counts (at least 1 region of each type; at least 20 haemorrhages per quadrant) and a data-calibrated set whose counts are fitted on the APTOS training split only.

{{table:rule_params}}

The calibrated thresholds are fitted to this detector's output on APTOS, not to clinical practice (they can look absurd as clinical counts), and the report text produced by the application says so.

## 14.4 Evidence fused with the classifier

The classifier and the lesion path are compared and combined on the **same held-out APTOS test images**. Everything tunable is fitted on training or validation rows only: the calibrated rule counts and a gradient-boosting grader on the lesion features (training split), the fusion weight and thresholds (validation split). The lesion model was trained on IDRiD and is applied to APTOS without any APTOS lesion annotation, so its APTOS detections are an unvalidated domain transfer; their value is measured here only indirectly, through agreement with the APTOS grade.

{{table:lesion_fusion}}

{{table:lesion_paired}}

{{figure:lesion_fusion_auc.png|Figure 11c. Referable-DR ROC-AUC of the classifier, the lesion-based graders and the fusion on the same test images.}}

**Cross-check policy.** When the lesion rules indicate referable disease but the classifier does not, the case is escalated to human review (a possible missed case); when the classifier flags a case but no supporting lesions are found, the flag is kept and the lack of corroboration is stated. The table shows how often the two agree, whether the classifier is wrong more often when they disagree, and what escalation costs and gains in terms of extra referrals and rescued referable cases.

{{table:lesion_cross}}

**What the lesion path adds, measured.** The literal ICDR counts are unusable on APTOS: the detector finds at least one lesion region in almost every image, so the rule calls nearly everything referable. Counts fitted on the training split (a high haemorrhage count per quadrant and tens of exudate regions, reflecting that detected-region counts are far larger than clinical counts) give moderate agreement (QWK {{val:lesion_fusion.rows.2.qwk|.3f}} on the test split), much weaker than the classifier ({{val:lesion_fusion.rows.0.qwk|.3f}}). A gradient-boosting grader on the same lesion features does much better (ROC-AUC {{val:lesion_fusion.rows.3.roc_auc|.3f}}), so the detected lesions do carry real information, but it remains significantly below the classifier (paired AUC difference {{val:lesion_fusion.paired_lesion_gb_minus_classifier.roc_auc.point_diff|+.3f}}, interval {{val:lesion_fusion.paired_lesion_gb_minus_classifier.roc_auc.lo|+.3f}} to {{val:lesion_fusion.paired_lesion_gb_minus_classifier.roc_auc.hi|+.3f}}). **Fusing the two did not improve screening:** the ROC-AUC difference is {{val:lesion_fusion.paired_fusion_minus_classifier.roc_auc.point_diff|+.3f}} (interval {{val:lesion_fusion.paired_fusion_minus_classifier.roc_auc.lo|+.3f}} to {{val:lesion_fusion.paired_fusion_minus_classifier.roc_auc.hi|+.3f}}, including zero) and the five-grade QWK is slightly lower ({{val:lesion_fusion.paired_fusion_minus_classifier.qwk.point_diff|+.3f}}, interval {{val:lesion_fusion.paired_fusion_minus_classifier.qwk.lo|+.3f}} to {{val:lesion_fusion.paired_fusion_minus_classifier.qwk.hi|+.3f}}). The lesion pathway is therefore justified here by what it *explains*, not by better accuracy.

As an uncertainty signal, disagreement between the classifier and the (calibrated) lesion rules is mildly informative: the classifier errs on {{val:lesion_fusion.cross_check.calibrated.classifier_error_rate_when_disagree|.1%}} of the cases where they disagree against {{val:lesion_fusion.cross_check.calibrated.classifier_error_rate_when_agree|.1%}} where they agree. Escalating the {{val:lesion_fusion.cross_check.calibrated.status_counts.lesions_only|d}} cases in which the rules say referable but the classifier does not would rescue only {{val:lesion_fusion.cross_check.calibrated.with_escalation.referable_cases_rescued|d}} of the {{val:lesion_fusion.cross_check.calibrated.classifier_alone.fn|d}} missed referable cases while adding {{val:lesion_fusion.cross_check.calibrated.with_escalation.n_extra_referrals|d}} extra referrals, a poor trade at this accuracy of the rules, so the escalation policy is implemented and reported but not recommended on this evidence.

## 14.5 Independent check on IDRiD grading images

IDRiD's grading test images were never used to fit the classifier, the rule thresholds or the gradient-boosting grader (images that are near-duplicates of the lesion model's training images are excluded to avoid leakage). This is a small external evaluation of the whole pipeline on a different camera and grading protocol, so its confidence intervals are wide and the result should be read as a consistency check, not as validation.

{{table:idrid_external_info}}

{{table:idrid_external}}

**What the independent check shows.** Performance drops substantially away from APTOS. On these {{val:external_idrid.n_evaluated|d}} images (referable prevalence {{val:external_idrid.referable_prevalence|.2f}}, higher than in APTOS) the classifier, using the threshold and temperature fitted on APTOS validation data and nothing tuned on IDRiD, reaches sensitivity {{val:external_idrid.rows.0.sensitivity|.3f}} (95% CI {{val:external_idrid.classifier_ci.sensitivity.lo|.3f}} to {{val:external_idrid.classifier_ci.sensitivity.hi|.3f}}), specificity {{val:external_idrid.rows.0.specificity|.3f}} ({{val:external_idrid.classifier_ci.specificity.lo|.3f}} to {{val:external_idrid.classifier_ci.specificity.hi|.3f}}), ROC-AUC {{val:external_idrid.rows.0.roc_auc|.3f}} ({{val:external_idrid.classifier_ci.roc_auc.lo|.3f}} to {{val:external_idrid.classifier_ci.roc_auc.hi|.3f}}) and QWK {{val:external_idrid.rows.0.qwk|.3f}}. The ROC-AUC shows that the ranking of cases is largely retained, but at the APTOS-derived operating point too many referable cases are missed; the fixed threshold does not transfer to a different camera and population. The lesion-feature grader trained on APTOS has a similar AUC ({{val:external_idrid.rows.3.roc_auc|.3f}}) and the rules fitted on APTOS counts transfer poorly (very high specificity, low sensitivity). The macular-involvement indicator (hard exudates within one disc diameter of the estimated fovea) separates IDRiD's macular-oedema-risk labels with a ROC-AUC of {{val:external_idrid.dme_proxy.roc_auc|.3f}}, modest evidence that it tracks something real. With only {{val:external_idrid.n_evaluated|d}} images the intervals are wide.

Figure 11d shows a systematic failure of the lesion model on APTOS. In several images with uneven illumination (a dark or over-bright sector), a large part of the retina is marked as haemorrhage (the big orange regions): these are false positives caused by domain shift to cameras and exposure conditions that IDRiD does not cover. They add large "haemorrhage" areas but few separate regions, so they distort the region counts used by the rules in unpredictable ways, and they are one reason the rule-derived grade agrees with the reference grade in only 2 of the 10 unselected example images shown. The disc and fovea markers, in contrast, sit plausibly in most of the examples.

{{figure:lesion_examples_aptos.png|Figure 11d. Lesion detections on APTOS images of increasing grade (domain transfer; no APTOS lesion annotations exist to score them). Green cross = estimated fovea.}}

## 14.6 What the screening report now says

The application and `Predictor.predict` return, in addition to the grade and referable decision: the lesion overlay, the per-type lesion counts by quadrant, the ICDR checklist with met / not-met / not-assessed status, the rule-derived grade with its reasons, the macular-involvement indicator, and the cross-check with the classifier. Lesion detections are evidence for a clinician to check, not verified findings.

{{figure:example_report.png|Figure 11e. Worked example (the first grade-3 image of the test split, chosen without looking at the outcome): the photograph, the classifier's Grad-CAM, and the detected lesions with the optic disc, fovea and quadrant axes. The classifier called this eye grade 2 and the lesion rules also gave grade 2, so both under-called the reference grade 3. The complete text produced by the application for this image follows.}}

{{file:example_report.md}}

## 14.7 Lesion model v2 (trained with an additional dataset; not adopted)

Section 14.5 showed false lesion detections on APTOS images with uneven illumination. A second lesion model was therefore trained to address that: IDRiD plus the e-ophtha lesion data [25] (microaneurysm and exudate masks plus healthy images from other clinics and cameras; obtained from a public mirror that declares an MIT licence, whose original distributor terms were not independently verified), a loss that ignores lesion types a source does not annotate (partial labels), augmentation that imitates acquisition differences (illumination, colour and contrast changes), and a filter that discards implausibly large regions. Whether to adopt it was decided by a rule written down beforehand: **adopt v2 only if (a) its mean AUPR on the IDRiD test images is not more than 0.03 below the deployed v1, and (b) the rule engine's agreement with the expert APTOS grade on the validation rows improves by at least 0.03.**

{{table:lesion_v2_aupr}}

{{table:lesion_v2_aptos}}

{{table:lesion_v2_decision}}

**Reading the result honestly.** v2 is a clearly worse segmenter on the IDRiD test images it was originally measured on (mean AUPR {{val:lesion_v2_comparison.idrid_test_mean_aupr.v2|.3f}} against {{val:lesion_v2_comparison.idrid_test_mean_aupr.v1|.3f}}), so criterion (a) fails and by the pre-set rule v1 stays deployed. At the same time, v2 produces far fewer implausible detections on healthy APTOS images (the share of grade-0 images with at least 20 detected haemorrhage regions falls from {{val:lesion_v2_comparison.aptos_v1.grade0_share_with_HE_ge_20|.1%}} to {{val:lesion_v2_comparison.aptos_v2.grade0_share_with_HE_ge_20|.1%}}) and its rule-derived grade agrees much better with the APTOS grade (validation QWK {{val:lesion_v2_comparison.aptos_v1.rules_val_qwk|.3f}} to {{val:lesion_v2_comparison.aptos_v2.rules_val_qwk|.3f}}; test {{val:lesion_v2_comparison.aptos_v1.rules_test_qwk|.3f}} to {{val:lesion_v2_comparison.aptos_v2.rules_test_qwk|.3f}}). The two criteria therefore point in opposite directions. The APTOS figures are an indirect measure (no APTOS lesion annotations exist), the rule thresholds are refitted on APTOS training rows, and the IDRiD AUPR drop measures the actual segmentation task; I therefore did not override the rule. The v2 weights and features are kept in the project's output folder as a documented alternative, and the finding that more diverse training data reduce false detections on a different camera while costing some in-domain segmentation accuracy is itself a result of this work. The gradient-boosting grader on the lesion features is unchanged in quality (ROC-AUC {{val:lesion_v2_comparison.aptos_v1.gb_test_auc|.3f}} against {{val:lesion_v2_comparison.aptos_v2.gb_test_auc|.3f}} on the test rows).

# 15. Ablation and baseline comparison

{{table:baselines}}

{{table:ablation}}

{{table:paired}}

{{figure:ablation_auc.png|Figure 13. Referable-DR ROC-AUC of each model on the same test images.}}

Rows labelled with the quality gate use **conservative accounting**: images that the gate rejects ("recapture") or that fall in the confidence band around the threshold ("human review") are counted as referred, so that every image receives an outcome and nothing is dropped from the denominator. No lesion-evidence or lesion–CNN fusion row exists because that pathway was not implemented. Claims of improvement are made only where the paired-bootstrap interval below excludes zero. Transfer learning with EfficientNet-B0 improved the five-grade agreement over a small CNN trained from scratch (QWK difference {{val:paired_comparison_primary_minus_baseline.Small CNN from scratch (224px).qwk.point_diff|+.3f}}, 95% CI {{val:paired_comparison_primary_minus_baseline.Small CNN from scratch (224px).qwk.lo|+.3f}} to {{val:paired_comparison_primary_minus_baseline.Small CNN from scratch (224px).qwk.hi|+.3f}}), but for the referable ROC-AUC alone the difference ({{val:paired_comparison_primary_minus_baseline.Small CNN from scratch (224px).roc_auc.point_diff|+.3f}}, interval {{val:paired_comparison_primary_minus_baseline.Small CNN from scratch (224px).roc_auc.lo|+.3f}} to {{val:paired_comparison_primary_minus_baseline.Small CNN from scratch (224px).roc_auc.hi|+.3f}}) is not distinguishable from zero on this test set. Hand-crafted image statistics with gradient boosting are clearly weaker. The gated pipeline rows show a trade-off rather than an improvement: counting rejected and borderline cases as referred raises referrals (lower specificity) and, here, did not raise sensitivity.

## 15.1 Robustness-augmented training (comparison)

The degraded-image experiment in Section 13 showed that the standard classifier fails silently under some defects: it calls blurred images healthy, and it calls almost every heavily JPEG-compressed image referable. A second EfficientNet-B0 was therefore trained with exactly the same data split, architecture, schedule and seed, changing only the **training** augmentation: Gaussian blur (25% of images), JPEG re-compression at quality 8-60 (30%), wider darkening / contrast / washed-out illumination changes (30%) and over-exposure (15%), on top of the standard rotations and flips. Validation and test images receive no augmentation. The two models are compared on the same {{val:robustness_comparison.n_test|d}} clean test images with a paired bootstrap, and on the same degraded images through the same code path. The decision rule for recommending the robust model was written down before the results were seen.

{{table:robust_clean}}

{{table:robust_paired}}

{{table:robust_degraded}}

{{figure:robustness_comparison.png|Figure 12a. Sensitivity and specificity of the standard and the robustness-augmented model on clean and synthetically degraded test images (severity 2, no quality gate). The standard model's specificity of 1.00 on blurred images only reflects that it calls every blurred image healthy (its sensitivity there is near zero).}}

{{table:robust_decision}}

**What the comparison shows.** On clean images the two models are statistically indistinguishable (all paired intervals include zero; the specificity at the operating point is {{val:robustness_comparison.paired_robust_minus_standard.specificity.point_diff|+.3f}} for the robust model, interval {{val:robustness_comparison.paired_robust_minus_standard.specificity.lo|+.3f}} to {{val:robustness_comparison.paired_robust_minus_standard.specificity.hi|+.3f}}), and the robust model is better calibrated (five-grade ECE {{val:robustness_comparison.clean_test.robust.ece_5grade|.3f}} versus {{val:robustness_comparison.clean_test.standard.ece_5grade|.3f}}). On degraded images it is much more reliable: sensitivity on blurred images rises from near zero to about 0.9 and specificity on JPEG-compressed images from about 0.05 to about 0.65.

**Limits of this comparison.** (1) The degradations used for testing belong to the same families as the training augmentations, which favours the robust model; gains against other real-world defects are untested. (2) Each degraded condition uses {{val:robustness_comparison.degraded.0.n|d}} images at one severity, and per-image outputs were not stored, so no confidence intervals are given for these rows; the differences are large but not formally tested. (3) JPEG robustness is improved, not solved (specificity remains well below the clean value). (4) The recommendation used test-split images (the clean non-inferiority check and the degraded subset), so the robust model's clean-test numbers are not as free of selection as those of the standard model, whose configuration was fixed before any test result was seen. For that reason the headline results in this report remain those of the standard model; the demonstration application uses the robust checkpoint (as the first member of the ensemble of Section 15.2) because the pre-set rule recommended it, and both models' numbers are reported here in full.

## 15.2 Larger backbone, ensemble and flip test-time augmentation

An EfficientNet-B3 at 448 px with the same robustness augmentation, split, schedule and seed was trained as a second backbone, and candidate ensembles were formed from it and the robust B0, with and without flip test-time augmentation (the average of the plain image and its horizontal and vertical flips). All candidates use the temperature and the threshold fitted on **APTOS validation** data only. The candidate is selected on the **validation** mean of QWK and ROC-AUC, and adopted over the current default (robust B0, no augmentation) only if its validation score is at least 0.003 higher and the paired test-split intervals show it is not worse; the test split plays no part in the selection.

{{table:ensemble_candidates}}

{{table:ensemble_decision}}

The selected candidate is the B0 + B3 ensemble with flip test-time augmentation. Its gain over the previous default is small and not statistically distinguishable from zero on the test split (ROC-AUC difference {{val:ensemble_selection.paired_selected_minus_default_test.roc_auc.point_diff|+.4f}}, interval {{val:ensemble_selection.paired_selected_minus_default_test.roc_auc.lo|+.4f}} to {{val:ensemble_selection.paired_selected_minus_default_test.roc_auc.hi|+.4f}}; QWK difference {{val:ensemble_selection.paired_selected_minus_default_test.qwk.point_diff|+.4f}}, interval {{val:ensemble_selection.paired_selected_minus_default_test.qwk.lo|+.4f}} to {{val:ensemble_selection.paired_selected_minus_default_test.qwk.hi|+.4f}}); it satisfies the pre-set rule, which was designed to require only validation gain and non-inferiority. It costs several times the computation of a single B0 (two networks, three views of the image each; B3 at 448 px is also the larger network). The IDRiD columns show that none of these variants cures the external drop (Section 14.5): the ensemble's IDRiD sensitivity is higher than the single B0's, but with n = 100 the difference is within noise.

## 15.3 Adding EyePACS training data

The external drop suggested that APTOS (one source, one grading protocol) is too narrow a training set. Because the competition rules had been accepted, the much larger EyePACS data (Section 5.5) were added. **Phase A** trained an EfficientNet-B0 (same architecture, resolution and robustness augmentation as the deployed model) on the union of APTOS training images and the EyePACS training pool; **phase B** fine-tuned phase A's best checkpoint for five epochs on APTOS training images alone at a lower learning rate. The APTOS split is the same as for every other model; APTOS validation is the only set used to pick checkpoints, the temperature and the threshold; APTOS test, the IDRiD grading test and a patient-level EyePACS test sample are evaluation only. The decision rule was fixed before the results: the candidate is whichever of phase A and B has the higher APTOS-validation QWK, and it becomes the application default only if (a) it is non-inferior to the current model on the APTOS test split (paired AUC interval lower bound above -0.010 and QWK lower bound above -0.020) **and** (b) its IDRiD ROC-AUC is at least 0.02 higher with no loss of sensitivity.

{{table:eyepacs_models}}

{{table:eyepacs_paired}}

{{table:eyepacs_decision}}

**What the experiment shows.** Training on more diverse data improved performance away from APTOS: on the IDRiD grading images the ROC-AUC rose from {{val:eyepacs_experiment.models.efficientnet_b0_robust.idrid_test.roc_auc|.3f}} to {{val:eyepacs_experiment.models.efficientnet_b0_eyepacs.idrid_test.roc_auc|.3f}} (paired difference {{val:eyepacs_experiment.paired_candidate_minus_robust.idrid_test.roc_auc.point_diff|+.3f}}, interval {{val:eyepacs_experiment.paired_candidate_minus_robust.idrid_test.roc_auc.lo|+.3f}} to {{val:eyepacs_experiment.paired_candidate_minus_robust.idrid_test.roc_auc.hi|+.3f}}, excluding zero), the sensitivity at the APTOS-derived threshold from {{val:eyepacs_experiment.models.efficientnet_b0_robust.idrid_test.sensitivity|.3f}} to {{val:eyepacs_experiment.models.efficientnet_b0_eyepacs.idrid_test.sensitivity|.3f}} and the five-grade QWK from {{val:eyepacs_experiment.models.efficientnet_b0_robust.idrid_test.qwk|.3f}} to {{val:eyepacs_experiment.models.efficientnet_b0_eyepacs.idrid_test.qwk|.3f}}; on the unseen EyePACS patients the ROC-AUC rose from {{val:eyepacs_experiment.models.efficientnet_b0_robust.eyepacs_test.roc_auc|.3f}} to {{val:eyepacs_experiment.models.efficientnet_b0_eyepacs.eyepacs_test.roc_auc|.3f}} (an in-domain gain for the new model there, since it saw EyePACS-like data, and an external-domain result for the APTOS-only model). In exchange, on the APTOS test split the five-grade agreement is slightly lower (QWK difference {{val:eyepacs_experiment.paired_candidate_minus_robust.test.qwk.point_diff|+.3f}}, interval {{val:eyepacs_experiment.paired_candidate_minus_robust.test.qwk.lo|+.3f}} to {{val:eyepacs_experiment.paired_candidate_minus_robust.test.qwk.hi|+.3f}}); the interval's lower bound is below the pre-set tolerance of -0.020, so criterion (a) is **not met** and the rule says not to adopt the model as the default. It is installed as an optional model in the application ("EyePACS-augmented, experimental").

**Caveats.** EyePACS labels are noisier than APTOS and IDRiD labels, and the EyePACS test sample is drawn from the same population as the added training data, so its numbers are an in-domain result for the new model. The IDRiD test set has only {{val:eyepacs_experiment.models.efficientnet_b0_eyepacs.idrid_test.n|d}} images, so criterion (b) is a weak guard and its intervals are wide. The sensitivity on EyePACS images at the APTOS-derived operating point is low for both models ({{val:eyepacs_experiment.models.efficientnet_b0_robust.eyepacs_test.sensitivity|.2f}} and {{val:eyepacs_experiment.models.efficientnet_b0_eyepacs.eyepacs_test.sensitivity|.2f}}): the threshold chosen on APTOS does not transfer to EyePACS's grading and prevalence, which again shows that a deployment would need its own calibration. The trade-off between a small in-domain loss and a better behaviour on other data is a judgement for the user of the system; the rule decided it conservatively.

# 16. Workflow simulation

A discrete-event simulation (SimPy [16]) models patients arriving at a screening site, queueing for fundus cameras, imaging, uploading over a slow link, AI inference, a quality check with possible recapture, and a review queue for flagged or uncertain cases. **Every parameter is an assumption, not observed operational data** (annual volume of 100,000 patients follows the project roadmap; imaging, upload and review times are illustrative). The simulation counts working time only (queues carry over between days), so calendar waiting times are understated. Its results describe how this queueing system behaves under the stated assumptions; they are not evidence of clinical effectiveness. Little's law (L = λW) [21] was used as a consistency check.

{{table:sim_assumptions}}

{{table:sim_sweep}}

{{table:sim_littles}}

# 17. Error analysis

False negatives (missed referable cases) and false positives (false alarms) are listed in `results/predictions.csv` with the true and predicted grades and probabilities, and the most confident errors are visualised in Figure 10. Misclassifications between adjacent grades (for example 1 versus 2 around the referable boundary) are expected given the subjectivity of grading and the rarity of grades 1, 3 and 4 in the data.

# 18. Performance and reproducibility

{{table:latency}}

All randomness is seeded (seed {{val:config.seed|d}}), the split manifest and calibration file are saved, and `pytest` covers preprocessing, dataset loading and split leakage, model shapes, the training loop, checkpoint reload, metric agreement with scikit-learn on controlled examples, the app prediction function, report generation and the simulation. The report is generated from `results/metrics.json` by `scripts/build_report.py`; values that could not be computed are shown as "N/A (not computed)" and listed in `report/missing_values.txt`.

## 18.1 Software testing and verification

The code is covered by an automated test suite that is run before every release of the report. The table below is generated from the output of the last run (`scripts/test_summary.py`), not typed in; all {{val:test_summary.total|d}} tests passed in {{val:test_summary.seconds|.0f}} seconds ({{val:test_summary.run_date}}).

{{table:tests}}

Beyond unit tests, the following verifications were run and are part of the evidence for the results:

- **Reproducible inference.** The best checkpoint was reloaded twice and produced identical logits on the same inputs (maximum absolute difference {{val:checkpoint_reload_check.max_abs_diff|.1e}}; all values finite).
- **Metric correctness.** Each metric was checked against scikit-learn or a hand calculation on controlled examples, including the tie-handling of the precision-recall area, quadratic weighted kappa, expected calibration error and the paired bootstrap.
- **Leakage guards.** An assertion fails if any exact or near-duplicate image group appears in more than one split, and a test checks that both eyes of an EyePACS patient always stay in one split; EyePACS images were also screened against the APTOS and IDRiD evaluation images ({{val:eyepacs_experiment.data.dedup.n_removed_near_duplicates|d}} near-duplicates found).
- **Smoke runs before every long run.** Each Kaggle script was first executed on a small subset (on CPU where possible, so no GPU time was spent) and only then run in full; this caught several of the defects below.
- **End-to-end checks of the application.** The interface was driven in a real browser: upload, analysis, switching between models, zoom and reset, starting a new image, and an analysis request with no image; the layout was checked at five window widths for overflow, nested scrolling and clipped text (Section 18.2).
- **Report integrity.** Every number in this report enters through a placeholder resolved from a saved result file; unresolved placeholders are listed in `report/missing_values.txt` (none in this version).

**Defects found by this testing and fixed** (each has a regression test or a guard where the code could re-introduce it):

1. *Wrong ground-truth file.* The first full training run read the competition's `sample_submission.csv` as labels (5,590 rows with a constant label). The run was stopped and discarded; label discovery now ignores submission files and constant-label tables, and the pipeline refuses to start if the count of labelled images differs from the expected number.
2. *Precision-recall area with tied scores.* An integration order gave a perfect classifier an area of 0.098; it was replaced by the standard descending-threshold sum and cross-checked against scikit-learn.
3. *IDRiD name collision.* Training and test images share file names, which silently mismatched labels in a merge; the merge is now positional within each split.
4. *Optic-disc and fovea locator.* A coordinate regressor trained on IDRiD located the disc poorly on APTOS photographs; the deployed locator uses the segmented disc and accepts the regressor's fovea only when it is consistent (median disc error fell from 79 px to 23 px).
5. *Silent failures in the Kaggle submission tooling.* A push that the service rejected was reported as success; error text from the command-line client is now treated as failure, and expired sign-in is reported as such.
6. *Dataset mounts.* Kaggle mounts some dataset folders with names such as `A.%20Segmentation`; the inference and EyePACS kernels failed on the first smoke run and now locate data by content. A slice that ran past the end of the first dataset when merging two image caches was found the same way.
7. *Interface defects.* Dark-theme text was invisible on the result cards, the PDF case report printed literal `<b>` markup, and the launcher script mishandled its list of Python candidates; all were fixed, and the interface was then rebuilt with a light-only design system.

## 18.2 Demonstration application

A Gradio [15] interface (`app/app.py`, started with `run_app.ps1` or `python app/app.py`; the product name defaults to "RetinaLab" and is a setting) wraps a single reusable prediction function (`drscreen.predict.Predictor.predict`) and presents the result for a non-specialist reader. Its design (a light, restrained clinical style with one set of colour, type, spacing and radius tokens, a fluid layout that uses the whole window, and short fades only) lives in `src/drscreen/ui_theme.py`. For an uploaded photograph it shows:

- on the left, a **retinal image viewer** with a segmented switch between the original, the pre-processed image, the Grad-CAM overlay and the detected-lesion overlay (a view that was not produced is disabled, never faked), zoom, pan, reset and an expanded inspection view;
- on the right, the **screening result**: the recommendation (referral recommended, no referral indicated, needs a human reviewer, or please retake the photograph), a threshold chart that shows the referable-DR score, the operating threshold, the optional human-review band and the side of the threshold on which the score falls (position, shape and text carry the meaning, not colour alone), then the predicted grade, the score, the threshold, the image-quality verdict, a **heuristic confidence level** (the distance of the score from the threshold, lowered when image quality is poor or the lesion evidence disagrees; explicitly not a calibrated probability of being right) and the model used;
- below, tabs for the **interpretation** (a plain-language explanation generated from the computed values only, and the stated limits, for example that new-vessel growth cannot be assessed), **Grad-CAM**, the **lesion evidence** (counts of detected regions, a per-quadrant grid, a macular indicator and the **ICDR criteria checklist** with each criterion marked met, not met or not assessed, and the cross-check with the classifier), **calibration** (the five grade probabilities and how to read them) and the full text report;
- a downloadable **PDF case report** for that image (Figure 14), so a reviewer can keep or forward the reasoning.

It loads real trained checkpoints. A menu offers the validated B0 + B3 ensemble (the default, Section 15.2), the robustness-augmented model, the standard model and the optional EyePACS-augmented model (Section 15.3). Invalid files are reported without a prediction; a rejected image is returned with a request to retake it; when the lesion models are absent the interface states that no lesion evidence was computed. The interface runs wherever it is started; it is not a continuously hosted service, and a Kaggle notebook is a batch compute environment rather than an inference server. Figures 14a and 14b were captured from the running application with `scripts/ui_screenshots.py`, which also checks the layout at five window widths for horizontal overflow, nested scroll containers and clipped text.

{{figure:ui_upload_state.png|Figure 14a. The interface before an image is loaded (upload state).}}

{{figure:ui_results_first_screen.png|Figure 14b. The interface after analysis (a public APTOS test image): the retinal image viewer on the left, the screening result and threshold chart on the right, the evidence tabs below.}}

{{figure:case_report_example.png|Figure 14. First page of the PDF case report produced by the interface for the worked example of Section 14.6: recommendation, key numbers, the four supporting images and the plain-language reasoning (the second page holds the ICDR checklist and the intended-use statement).}}

# 19. Limitations, risks and ethics

- **Mostly internal evaluation, and the external check is worse.** The headline results are internal holdout results. The only independent check, the small IDRiD grading test in Section 14.5 (one camera type, a different grading protocol, {{val:external_idrid.n_evaluated|d}} images), gave a sensitivity of {{val:external_idrid.rows.0.sensitivity|.3f}} and ROC-AUC {{val:external_idrid.rows.0.roc_auc|.3f}}, against {{val:internal_holdout.screening.sensitivity|.3f}} and {{val:internal_holdout.screening.roc_auc|.3f}} internally. The operating threshold does not transfer across datasets; any deployment would need local calibration and validation.
- **No patient-level separation.** APTOS has no patient identifiers; the split guards only against duplicate and near-duplicate images.
- **Label noise and single-source grading.** Grades are subjective and the grading protocol of the source is not under this project's control.
- **Prevalence.** APTOS is enriched for disease relative to a screening population, so predictive values here do not transfer to a general screening setting.
- **Lesion evidence is limited.** Venous beading, IRMA and neovascularisation are not detected, so grade 4 and two of the three 4-2-1 criteria cannot be assessed. Lesion detections on APTOS are an unvalidated transfer from IDRiD (a single camera type, 54 training images; a second model trained with extra e-ophtha data reduced false detections but segmented IDRiD less accurately and was not adopted, Section 14.7); microaneurysm detection in particular is hard. Quadrants rely on a model-estimated fovea and assume an upright photograph. Counts of detected regions are not clinical lesion counts. There is no direct macular-oedema assessment, only an exudate-proximity indicator; referable DR is defined by grade ≥ 2.
- **Selection effects.** Several models and variants were compared. Every comparison used a rule fixed before its results, a single APTOS validation split for tuning, and paired intervals on the test split, but the test split and the 100-image IDRiD set were looked at repeatedly across experiments, so small differences between variants should not be over-read. The EyePACS model improved external numbers at a small in-domain cost and was not adopted by its rule; EyePACS labels are noisier than the other sources.
- **Quality gate** is heuristic and validated only on synthetic degradations.
- **Probabilities** are model scores post-scaled on one validation split, not clinically validated risks.
- **Not a medical device.** The system has no regulatory approval, no prospective validation, and no safety case. A false negative can mean a missed sight-threatening disease, and automation bias is a real risk if outputs are over-trusted. It must be used only for research and teaching, with qualified clinicians responsible for any real decision. Fundus images are personal health data and any real deployment would require consent, de-identification and governance that are outside this project.

# 20. Conclusion and future work

This project delivers a reproducible, tested pipeline for five-grade DR classification and referable-DR screening with an explicit evaluation protocol that selects every tunable quantity on validation data only. The measured internal-holdout performance is reported with bootstrap confidence intervals in Section 10; on the one independent test available it did not hold up (Section 14.5), which is the most important caveat of this work. The lesion pathway provides explicit, checkable reasons for a grade, but on this evidence it does not make the classifier more accurate. Later experiments improved the demonstrated system in a measured way: an ensemble with test-time augmentation was adopted by a pre-set rule, and adding EyePACS data improved external-domain performance at a small in-domain cost (Section 15.3), though the pre-set rule left it as an optional model; none of them removed the external-validation caveat. Future work: external validation on Messidor-2 and further datasets and larger, multi-source training data; patient-level splits; detectors for venous beading, IRMA and neovascularisation (and lesion data from more cameras such as DDR or FGADR) so the rule engine can assess all of the ICDR criteria; a quality model trained on genuine quality labels; prospective evaluation; and a hosted deployment.

# 21. References

1. Gulshan V, Peng L, Coram M, et al. Development and validation of a deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs. *JAMA*. 2016;316(22):2402–2410. doi:10.1001/jama.2016.17216
2. Abràmoff MD, Lavin PT, Birch M, Shah N, Folk JC. Pivotal trial of an autonomous AI-based diagnostic system for detection of diabetic retinopathy in primary care offices. *npj Digital Medicine*. 2018;1:39.
3. Rema M, Premkumar S, Anitha B, Deepa R, Pradeepa R, Mohan V. Prevalence of diabetic retinopathy in urban India: the Chennai Urban Rural Epidemiology Study (CURES) eye study, I. *Investigative Ophthalmology & Visual Science*. 2005;46(7):2328–2333. doi:10.1167/iovs.05-0019
4. Anjana RM, Unnikrishnan R, Deepa M, et al. Metabolic non-communicable disease health report of India: the ICMR-INDIAB cross-sectional study (ICMR-INDIAB-17). *The Lancet Diabetes & Endocrinology*. 2023;11(7).
5. Wilkinson CP, Ferris FL, Klein RE, et al. Proposed international clinical diabetic retinopathy and diabetic macular edema disease severity scales. *Ophthalmology*. 2003;110(9):1677–1682.
6. Tan M, Le QV. EfficientNet: rethinking model scaling for convolutional neural networks. *Proceedings of the 36th International Conference on Machine Learning (PMLR 97)*. 2019:6105–6114.
7. Selvaraju RR, Cogswell M, Das A, Vedantam R, Parikh D, Batra D. Grad-CAM: visual explanations from deep networks via gradient-based localization. *Proceedings of the IEEE International Conference on Computer Vision*. 2017:618–626.
8. Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. *Proceedings of the 34th International Conference on Machine Learning (PMLR 70)*. 2017:1321–1330.
9. Cohen J. Weighted kappa: nominal scale agreement with provision for scaled disagreement or partial credit. *Psychological Bulletin*. 1968;70(4):213–220.
10. Efron B, Tibshirani RJ. *An Introduction to the Bootstrap*. Chapman & Hall; 1993.
11. Asia Pacific Tele-Ophthalmology Society. APTOS 2019 Blindness Detection (Kaggle competition). https://www.kaggle.com/c/aptos2019-blindness-detection
12. Wightman R. PyTorch Image Models (timm). 2019. doi:10.5281/zenodo.4414861
13. Paszke A, Gross S, Massa F, et al. PyTorch: an imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems 32*. 2019.
14. Porwal P, Pachade S, Kamble R, et al. Indian Diabetic Retinopathy Image Dataset (IDRiD): a database for diabetic retinopathy screening research. *Data*. 2018;3(3):25.
15. Abid A, Abdalla A, Abid A, Khan D, Alfozan A, Zou J. Gradio: hassle-free sharing and testing of ML models in the wild. arXiv:1906.02569. 2019.
16. SimPy: discrete event simulation for Python. https://simpy.readthedocs.io
17. Graham B. Kaggle Diabetic Retinopathy Detection competition report. University of Warwick; 2015.
18. Adebayo J, Gilmer J, Muelly M, Goodfellow I, Hardt M, Kim B. Sanity checks for saliency maps. *Advances in Neural Information Processing Systems 31*. 2018.
19. Loshchilov I, Hutter F. Decoupled weight decay regularization. *International Conference on Learning Representations*. 2019.
20. Zuiderveld K. Contrast limited adaptive histogram equalization. In: *Graphics Gems IV*. Academic Press; 1994:474–485.
21. Little JDC. A proof for the queuing formula: L = λW. *Operations Research*. 1961;9(3):383–387.
22. Brier GW. Verification of forecasts expressed in terms of probability. *Monthly Weather Review*. 1950;78(1):1–3.
23. Davis J, Goadrich M. The relationship between precision-recall and ROC curves. *Proceedings of the 23rd International Conference on Machine Learning*. 2006:233–240.
24. Kaggle and EyePACS. Diabetic Retinopathy Detection (Kaggle competition, 2015). https://www.kaggle.com/c/diabetic-retinopathy-detection
25. Decencière E, Cazuguel G, Zhang X, et al. TeleOphta: machine learning and image processing methods for teleophthalmology. *IRBM*. 2013;34(2):196–203.
