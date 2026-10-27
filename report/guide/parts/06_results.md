# Results and what they mean

The measured outcomes of the real project, read from its saved result files. Every result is labelled with the data it comes from: the **internal test split** (held-out APTOS images, same source as training), or an **external** dataset (different source, never used for training or selection).

## Experimental setup

{{tab:setup|The evaluated configuration of the headline model.}}

| Item | Value |
|---|---|
| training data | APTOS 2019, {{val:dataset.splits.train.n|d}} training and {{val:dataset.splits.val.n|d}} validation images |
| evaluation data | APTOS internal test split, {{val:internal_holdout.n_samples|d}} images ({{val:internal_holdout.screening.n_positive|d}} referable, {{val:internal_holdout.screening.n_negative|d}} not) |
| model | EfficientNet-B0, 384 px, ImageNet pre-trained, fine-tuned (`artifacts/efficientnet_b0/best.pt`, best epoch {{val:run.best_epoch|d}}) |
| calibration | temperature {{val:validation_threshold_selection.temperature|.2f}}, fitted on validation |
| decision threshold | {{val:validation_threshold_selection.threshold|.3f}} on the referable score, chosen on validation (sensitivity ≥ 0.90) |
| uncertainty | percentile bootstrap, 1,000 resamples, 95% intervals |
| hardware | Kaggle, {{val:environment.gpu}} GPU; PyTorch {{val:environment.torch}}, timm {{val:environment.timm}} |

The headline results are those of the **standard** model, whose configuration was fixed before any test result was seen. Later models (robust, B3, ensemble, EyePACS) were compared with it afterwards and are reported in their own sections.

## Referable-DR screening on the internal test split

{{tab:screening|Referable-DR screening, APTOS internal test split (n = {{val:internal_holdout.screening.n|d}}, prevalence {{val:internal_holdout.screening.prevalence|.3f}}), standard EfficientNet-B0 at the validation-chosen threshold.}}

| Metric | Value | 95% CI | Plain meaning |
|---|---|---|---|
| sensitivity | {{val:internal_holdout.screening.sensitivity|.3f}} | {{val:internal_holdout.confidence_intervals.sensitivity.lo|.3f}}–{{val:internal_holdout.confidence_intervals.sensitivity.hi|.3f}} | {{val:internal_holdout.screening.tp|d}} of {{val:internal_holdout.screening.n_positive|d}} referable eyes flagged |
| specificity | {{val:internal_holdout.screening.specificity|.3f}} | {{val:internal_holdout.confidence_intervals.specificity.lo|.3f}}–{{val:internal_holdout.confidence_intervals.specificity.hi|.3f}} | {{val:internal_holdout.screening.tn|d}} of {{val:internal_holdout.screening.n_negative|d}} non-referable eyes passed |
| PPV (precision) | {{val:internal_holdout.screening.precision|.3f}} | {{val:internal_holdout.confidence_intervals.precision.lo|.3f}}–{{val:internal_holdout.confidence_intervals.precision.hi|.3f}} | share of referrals that were correct |
| NPV | {{val:internal_holdout.screening.npv|.3f}} | {{val:internal_holdout.confidence_intervals.npv.lo|.3f}}–{{val:internal_holdout.confidence_intervals.npv.hi|.3f}} | share of cleared eyes that were correct |
| ROC-AUC | {{val:internal_holdout.screening.roc_auc|.3f}} | {{val:internal_holdout.confidence_intervals.roc_auc.lo|.3f}}–{{val:internal_holdout.confidence_intervals.roc_auc.hi|.3f}} | separation, independent of threshold |
| PR-AUC | {{val:internal_holdout.screening.pr_auc|.3f}} | {{val:internal_holdout.confidence_intervals.pr_auc.lo|.3f}}–{{val:internal_holdout.confidence_intervals.pr_auc.hi|.3f}} | precision across recall levels |
| balanced accuracy | {{val:internal_holdout.screening.balanced_accuracy|.3f}} | {{val:internal_holdout.confidence_intervals.balanced_accuracy.lo|.3f}}–{{val:internal_holdout.confidence_intervals.balanced_accuracy.hi|.3f}} | mean of sensitivity and specificity |
| F1 | {{val:internal_holdout.screening.f1|.3f}} | {{val:internal_holdout.confidence_intervals.f1.lo|.3f}}–{{val:internal_holdout.confidence_intervals.f1.hi|.3f}} | balance of precision and sensitivity |
| Brier score | {{val:internal_holdout.calibration.brier_referable|.3f}} | {{val:internal_holdout.confidence_intervals.brier.lo|.3f}}–{{val:internal_holdout.confidence_intervals.brier.hi|.3f}} | mean squared error of the score (lower is better) |

{{fig:confusion_referable.png|Screening confusion matrix on the internal test split (figure from the training run).|50|cmref}}

{{fig:roc_pr.png|ROC and precision-recall curves on the internal test split, with the operating point marked (figure from the training run).|86|rocpr}}

**Interpretation.** At the chosen threshold the model flagged {{val:internal_holdout.screening.tp|d}} of the {{val:internal_holdout.screening.n_positive|d}} referable test eyes and missed {{val:internal_holdout.screening.fn|d}}; it raised {{val:internal_holdout.screening.fp|d}} false alarms among {{val:internal_holdout.screening.n_negative|d}} non-referable eyes. The test sensitivity is slightly above the 0.90 targeted on validation, which is consistent with sampling variation (the interval spans {{val:internal_holdout.confidence_intervals.sensitivity.lo|.3f}} to {{val:internal_holdout.confidence_intervals.sensitivity.hi|.3f}}). The ROC-AUC near 0.99 means the scores rank referable eyes above non-referable ones almost perfectly *on images from this source*.

{{fig:threshold_sweep_validation.png|Sensitivity and specificity on the validation set as the threshold moves; the dashed line is the chosen threshold (figure from the training run).|74|sweep}}

## Five-grade classification

{{tab:grades|Per-grade results on the internal test split (standard EfficientNet-B0).}}

| Grade | Images | Precision | Recall | F1 |
|---|---|---|---|---|
| 0 No DR | {{val:internal_holdout.grades.per_grade.0.support|d}} | {{val:internal_holdout.grades.per_grade.0.precision|.3f}} | {{val:internal_holdout.grades.per_grade.0.recall|.3f}} | {{val:internal_holdout.grades.per_grade.0.f1|.3f}} |
| 1 Mild NPDR | {{val:internal_holdout.grades.per_grade.1.support|d}} | {{val:internal_holdout.grades.per_grade.1.precision|.3f}} | {{val:internal_holdout.grades.per_grade.1.recall|.3f}} | {{val:internal_holdout.grades.per_grade.1.f1|.3f}} |
| 2 Moderate NPDR | {{val:internal_holdout.grades.per_grade.2.support|d}} | {{val:internal_holdout.grades.per_grade.2.precision|.3f}} | {{val:internal_holdout.grades.per_grade.2.recall|.3f}} | {{val:internal_holdout.grades.per_grade.2.f1|.3f}} |
| 3 Severe NPDR | {{val:internal_holdout.grades.per_grade.3.support|d}} | {{val:internal_holdout.grades.per_grade.3.precision|.3f}} | {{val:internal_holdout.grades.per_grade.3.recall|.3f}} | {{val:internal_holdout.grades.per_grade.3.f1|.3f}} |
| 4 Proliferative DR | {{val:internal_holdout.grades.per_grade.4.support|d}} | {{val:internal_holdout.grades.per_grade.4.precision|.3f}} | {{val:internal_holdout.grades.per_grade.4.recall|.3f}} | {{val:internal_holdout.grades.per_grade.4.f1|.3f}} |
| **overall** | {{val:internal_holdout.n_samples|d}} | | | macro-F1 {{val:internal_holdout.grades.macro_f1|.3f}} · QWK {{val:internal_holdout.grades.qwk|.3f}} · exact accuracy {{val:internal_holdout.grades.accuracy_5|.3f}} |

{{fig:confusion_5grade.png|Five-grade confusion matrix on the internal test split (rows: dataset grade; columns: predicted grade).|92|cm5}}

**Interpretation.** Grade 0 is recognised almost perfectly. The difficulty lies in the middle and severe grades: grade 3 (severe NPDR) is recognised in only {{val:internal_holdout.grades.per_grade.3.recall|.0%}} of cases and is mostly confused with grade 2, and grade 4 is often predicted as grade 2. The quadratic weighted kappa of {{val:internal_holdout.grades.qwk|.3f}} is high because almost all errors are off by one grade. For screening the important point is that most of these errors stay on the referable side (2, 3 and 4 are all referable); for *severity grading* the model is much weaker than its QWK suggests.

## Calibration

{{fig:reliability.png|Reliability diagrams before and after temperature scaling on the internal test split (figure from the training run).|86|rel}}

Temperature scaling ($T$ = {{val:validation_threshold_selection.temperature|.2f}}) reduced the five-grade expected calibration error from {{val:internal_holdout.calibration_uncalibrated.ece_top_label_5grade|.3f}} to {{val:internal_holdout.calibration.ece_top_label_5grade|.3f}}, and the negative log-likelihood from {{val:internal_holdout.calibration_uncalibrated.nll_5grade|.3f}} to {{val:internal_holdout.calibration.nll_5grade|.3f}}, on test images it had never seen. The uncalibrated network was over-confident, as expected [22]. Calibration was fitted and checked only on APTOS images; it says nothing about calibration on other cameras.

## Baselines: is the deep network worth it?

{{tab:baselines|Baselines on the same test images, each with its own validation-chosen threshold.}}

| Model | Sensitivity | Specificity | ROC-AUC | QWK |
|---|---|---|---|---|
| majority-class prior (no image information) | {{val:baselines.Majority-class prior (no image information).sensitivity|.3f}} | {{val:baselines.Majority-class prior (no image information).specificity|.3f}} | {{val:baselines.Majority-class prior (no image information).roc_auc|.3f}} | {{val:baselines.Majority-class prior (no image information).qwk|.3f}} |
| hand-crafted image statistics + gradient boosting | {{val:baselines.Hand-crafted features + gradient boosting.sensitivity|.3f}} | {{val:baselines.Hand-crafted features + gradient boosting.specificity|.3f}} | {{val:baselines.Hand-crafted features + gradient boosting.roc_auc|.3f}} | {{val:baselines.Hand-crafted features + gradient boosting.qwk|.3f}} |
| small CNN trained from scratch (224 px) | {{val:baselines.Small CNN from scratch (224px).sensitivity|.3f}} | {{val:baselines.Small CNN from scratch (224px).specificity|.3f}} | {{val:baselines.Small CNN from scratch (224px).roc_auc|.3f}} | {{val:baselines.Small CNN from scratch (224px).qwk|.3f}} |
| **EfficientNet-B0, transfer learning (384 px)** | {{val:baselines.EfficientNet-B0 transfer learning (384px).sensitivity|.3f}} | {{val:baselines.EfficientNet-B0 transfer learning (384px).specificity|.3f}} | {{val:baselines.EfficientNet-B0 transfer learning (384px).roc_auc|.3f}} | {{val:baselines.EfficientNet-B0 transfer learning (384px).qwk|.3f}} |

The paired bootstrap shows that transfer learning improved five-grade agreement over the scratch CNN (QWK difference {{val:paired_comparison_primary_minus_baseline.Small CNN from scratch (224px).qwk.point_diff|+.3f}}, interval {{val:paired_comparison_primary_minus_baseline.Small CNN from scratch (224px).qwk.lo|+.3f}} to {{val:paired_comparison_primary_minus_baseline.Small CNN from scratch (224px).qwk.hi|+.3f}}). For the referable ROC-AUC alone the difference ({{val:paired_comparison_primary_minus_baseline.Small CNN from scratch (224px).roc_auc.point_diff|+.3f}}, interval {{val:paired_comparison_primary_minus_baseline.Small CNN from scratch (224px).roc_auc.lo|+.3f}} to {{val:paired_comparison_primary_minus_baseline.Small CNN from scratch (224px).roc_auc.hi|+.3f}}) cannot be distinguished from zero on this test set. Note that the scratch CNN's higher sensitivity comes with lower specificity at its own threshold, so the two rows are different operating points, not a ranking. Hand-crafted features are clearly weaker.

## Image quality and robustness

On the unmodified test images the gate judged {{val:quality.clean.good|d}} images good, {{val:quality.clean.enhance|d}} in need of enhancement and {{val:quality.clean.reject|d}} unusable. On synthetic degradations it rejected blurred images reliably, but it let JPEG-compressed and over-exposed images through ({{ref:qdeg}}). Worse, the standard classifier failed **silently** on some of them: on blurred images it called almost every eye healthy, and on heavily compressed images it called almost every eye referable.

{{fig:quality_degradation.png|The quality gate's verdict on synthetically degraded test images at three severities (figure from the training run).|86|qdeg}}

The robustness-augmented model (same split, architecture and schedule; only the training augmentation differs) was compared with the standard model by a rule fixed beforehand: adopt it if it is not meaningfully worse on clean images *and* improves mean balanced accuracy on degraded images by at least 0.05.

{{tab:robust|Standard versus robustness-augmented model on degraded test images (grade-stratified subset of {{val:robustness_comparison.degraded.0.n|d}} images, severity 2, no quality gate).}}

| Condition | Standard sens. | Standard spec. | Robust sens. | Robust spec. |
|---|---|---|---|---|
| clean | {{val:robustness_comparison.degraded.0.standard.ungated_sensitivity|.2f}} | {{val:robustness_comparison.degraded.0.standard.ungated_specificity|.2f}} | {{val:robustness_comparison.degraded.0.robust.ungated_sensitivity|.2f}} | {{val:robustness_comparison.degraded.0.robust.ungated_specificity|.2f}} |
| blur | {{val:robustness_comparison.degraded.1.standard.ungated_sensitivity|.2f}} | {{val:robustness_comparison.degraded.1.standard.ungated_specificity|.2f}} | {{val:robustness_comparison.degraded.1.robust.ungated_sensitivity|.2f}} | {{val:robustness_comparison.degraded.1.robust.ungated_specificity|.2f}} |
| over-exposed | {{val:robustness_comparison.degraded.3.standard.ungated_sensitivity|.2f}} | {{val:robustness_comparison.degraded.3.standard.ungated_specificity|.2f}} | {{val:robustness_comparison.degraded.3.robust.ungated_sensitivity|.2f}} | {{val:robustness_comparison.degraded.3.robust.ungated_specificity|.2f}} |
| JPEG | {{val:robustness_comparison.degraded.5.standard.ungated_sensitivity|.2f}} | {{val:robustness_comparison.degraded.5.standard.ungated_specificity|.2f}} | {{val:robustness_comparison.degraded.5.robust.ungated_sensitivity|.2f}} | {{val:robustness_comparison.degraded.5.robust.ungated_specificity|.2f}} |

The robust model was statistically indistinguishable on clean images and much more reliable on degraded ones (mean balanced-accuracy gain {{val:robustness_comparison.decision.mean_balanced_accuracy_gain_on_degraded|+.3f}}), so the rule adopted it for the application. Two caveats: the test degradations come from the same families as the training augmentations, which favours the robust model; and per-image outputs were not stored for these rows, so they carry no confidence intervals.

## Explanations: Grad-CAM and lesion evidence

Grad-CAM maps (shown in Part III) depend on the learned weights (mean Spearman correlation with weight-randomised maps {{val:gradcam.randomization_spearman_mean|.2f}}), but they are coarse and are not validated against lesion annotations.

{{tab:seg|Lesion segmentation on the IDRiD test images (pixel level).}}

| Structure | AUPR | Dice | IoU |
|---|---|---|---|
| microaneurysms | {{val:lesion_evidence.segmentation_idrid_test.per_lesion.MA.aupr|.3f}} | {{val:lesion_evidence.segmentation_idrid_test.per_lesion.MA.dice|.3f}} | {{val:lesion_evidence.segmentation_idrid_test.per_lesion.MA.iou|.3f}} |
| haemorrhages | {{val:lesion_evidence.segmentation_idrid_test.per_lesion.HE.aupr|.3f}} | {{val:lesion_evidence.segmentation_idrid_test.per_lesion.HE.dice|.3f}} | {{val:lesion_evidence.segmentation_idrid_test.per_lesion.HE.iou|.3f}} |
| hard exudates | {{val:lesion_evidence.segmentation_idrid_test.per_lesion.EX.aupr|.3f}} | {{val:lesion_evidence.segmentation_idrid_test.per_lesion.EX.dice|.3f}} | {{val:lesion_evidence.segmentation_idrid_test.per_lesion.EX.iou|.3f}} |
| cotton-wool spots | {{val:lesion_evidence.segmentation_idrid_test.per_lesion.SE.aupr|.3f}} | {{val:lesion_evidence.segmentation_idrid_test.per_lesion.SE.dice|.3f}} | {{val:lesion_evidence.segmentation_idrid_test.per_lesion.SE.iou|.3f}} |
| optic disc | {{val:lesion_evidence.segmentation_idrid_test.per_lesion.OD.aupr|.3f}} | {{val:lesion_evidence.segmentation_idrid_test.per_lesion.OD.dice|.3f}} | {{val:lesion_evidence.segmentation_idrid_test.per_lesion.OD.iou|.3f}} |

The optic disc is segmented almost perfectly; the lesions only moderately, which is typical for tiny, low-contrast targets learned from 54 training images. The deployed locator placed the optic-disc centre with a median error of {{val:lesion_evidence.locator_final_idrid_test.od_error_px_median|.0f}} pixels (on IDRiD images about {{val:lesion_evidence.locator_final_idrid_test.disc_diameter_px_approx|.0f}} pixels per disc diameter).

On APTOS, where no lesion annotations exist, the lesion path can only be judged by how well it agrees with the expert grade. The data-calibrated rule engine reached QWK {{val:lesion_fusion.rows.2.qwk|.3f}} against the classifier's {{val:lesion_fusion.rows.0.qwk|.3f}}; a gradient-boosting grader on the lesion features reached ROC-AUC {{val:lesion_fusion.rows.3.roc_auc|.3f}}, showing that the detections carry real information. **Fusing the lesion evidence with the classifier did not improve screening**: the paired ROC-AUC difference was {{val:lesion_fusion.paired_fusion_minus_classifier.roc_auc.point_diff|+.3f}} (interval {{val:lesion_fusion.paired_fusion_minus_classifier.roc_auc.lo|+.3f}} to {{val:lesion_fusion.paired_fusion_minus_classifier.roc_auc.hi|+.3f}}). The lesion path is therefore justified by what it *explains*, not by extra accuracy.

{{fig:lesion_examples_aptos.png|Lesion detections on APTOS test images of increasing grade (an unvalidated domain transfer). Several images with uneven illumination show large false "haemorrhage" regions, a known failure of the IDRiD-trained detector.|92|lesaptos}}

## The external check: performance on a different dataset

The most important result in this report comes from the **IDRiD grading test images**: a different hospital, camera, population and grading protocol, never used for training, threshold choice or calibration.

{{tab:external|The standard classifier on IDRiD (n = {{val:external_idrid.n_evaluated|d}}, referable prevalence {{val:external_idrid.referable_prevalence|.2f}}), with the APTOS-fitted temperature and threshold.}}

| Metric | IDRiD (external) | APTOS test (internal) |
|---|---|---|
| sensitivity | {{val:external_idrid.rows.0.sensitivity|.3f}} ({{val:external_idrid.classifier_ci.sensitivity.lo|.3f}}–{{val:external_idrid.classifier_ci.sensitivity.hi|.3f}}) | {{val:internal_holdout.screening.sensitivity|.3f}} |
| specificity | {{val:external_idrid.rows.0.specificity|.3f}} ({{val:external_idrid.classifier_ci.specificity.lo|.3f}}–{{val:external_idrid.classifier_ci.specificity.hi|.3f}}) | {{val:internal_holdout.screening.specificity|.3f}} |
| ROC-AUC | {{val:external_idrid.rows.0.roc_auc|.3f}} ({{val:external_idrid.classifier_ci.roc_auc.lo|.3f}}–{{val:external_idrid.classifier_ci.roc_auc.hi|.3f}}) | {{val:internal_holdout.screening.roc_auc|.3f}} |
| QWK | {{val:external_idrid.rows.0.qwk|.3f}} | {{val:internal_holdout.grades.qwk|.3f}} |

Away from APTOS, sensitivity dropped to about three in four referable eyes. The ROC-AUC shows that the *ranking* of eyes is still fairly good, but the threshold learned on APTOS no longer sits in the right place for this camera, population and grading protocol. With about 100 images the intervals are wide, but the drop is too large to be explained by chance alone. **The roadmap's target of more than 90% external sensitivity was not met.**

::: warning Why internal results flatter
Internal test images share the source, camera, grading habits and patient population of the training images, and possibly even patients. The model can exploit regularities that do not exist elsewhere. Only external data reveal how much of the performance transfers; here, a substantial part did not.
:::

## Later experiments, each decided by a pre-set rule

### Larger backbone, ensemble and test-time augmentation

{{tab:ens|Candidate models, selected on APTOS validation only (validation score = mean of QWK and ROC-AUC).}}

| Candidate | Validation score | Test sens. | Test spec. | Test AUC | Test QWK | IDRiD sens. |
|---|---|---|---|---|---|---|
| robust B0 | {{val:ensemble_selection.candidates.0.val_score|.4f}} | {{val:ensemble_selection.candidates.0.test_sensitivity|.3f}} | {{val:ensemble_selection.candidates.0.test_specificity|.3f}} | {{val:ensemble_selection.candidates.0.test_auc|.3f}} | {{val:ensemble_selection.candidates.0.test_qwk|.3f}} | {{val:ensemble_selection.candidates.0.idrid_sensitivity|.3f}} |
| robust B0 + flips | {{val:ensemble_selection.candidates.1.val_score|.4f}} | {{val:ensemble_selection.candidates.1.test_sensitivity|.3f}} | {{val:ensemble_selection.candidates.1.test_specificity|.3f}} | {{val:ensemble_selection.candidates.1.test_auc|.3f}} | {{val:ensemble_selection.candidates.1.test_qwk|.3f}} | {{val:ensemble_selection.candidates.1.idrid_sensitivity|.3f}} |
| robust B3 | {{val:ensemble_selection.candidates.2.val_score|.4f}} | {{val:ensemble_selection.candidates.2.test_sensitivity|.3f}} | {{val:ensemble_selection.candidates.2.test_specificity|.3f}} | {{val:ensemble_selection.candidates.2.test_auc|.3f}} | {{val:ensemble_selection.candidates.2.test_qwk|.3f}} | {{val:ensemble_selection.candidates.2.idrid_sensitivity|.3f}} |
| robust B3 + flips | {{val:ensemble_selection.candidates.3.val_score|.4f}} | {{val:ensemble_selection.candidates.3.test_sensitivity|.3f}} | {{val:ensemble_selection.candidates.3.test_specificity|.3f}} | {{val:ensemble_selection.candidates.3.test_auc|.3f}} | {{val:ensemble_selection.candidates.3.test_qwk|.3f}} | {{val:ensemble_selection.candidates.3.idrid_sensitivity|.3f}} |
| B0 + B3 | {{val:ensemble_selection.candidates.4.val_score|.4f}} | {{val:ensemble_selection.candidates.4.test_sensitivity|.3f}} | {{val:ensemble_selection.candidates.4.test_specificity|.3f}} | {{val:ensemble_selection.candidates.4.test_auc|.3f}} | {{val:ensemble_selection.candidates.4.test_qwk|.3f}} | {{val:ensemble_selection.candidates.4.idrid_sensitivity|.3f}} |
| **B0 + B3 + flips (selected)** | {{val:ensemble_selection.candidates.5.val_score|.4f}} | {{val:ensemble_selection.candidates.5.test_sensitivity|.3f}} | {{val:ensemble_selection.candidates.5.test_specificity|.3f}} | {{val:ensemble_selection.candidates.5.test_auc|.3f}} | {{val:ensemble_selection.candidates.5.test_qwk|.3f}} | {{val:ensemble_selection.candidates.5.idrid_sensitivity|.3f}} |

The ensemble had the best validation score and passed the rule (at least 0.003 better on validation, not worse on the test split), so it is the application's default. Its gain over the robust B0 is small and not statistically distinguishable from zero on the test split (ROC-AUC difference {{val:ensemble_selection.paired_selected_minus_default_test.roc_auc.point_diff|+.4f}}, interval {{val:ensemble_selection.paired_selected_minus_default_test.roc_auc.lo|+.4f}} to {{val:ensemble_selection.paired_selected_minus_default_test.roc_auc.hi|+.4f}}), and it costs several times the computation.

### Adding EyePACS training data

To attack the external drop, about 16,000 EyePACS images (split by patient, screened for duplicates of every evaluation image) were added to training, followed by a short fine-tune on APTOS.

{{tab:eyepacs|EyePACS experiment: the current robust B0 versus the EyePACS-augmented model (fine-tuned candidate). Temperature and threshold fitted on APTOS validation for both.}}

| Evaluation set | Metric | Robust B0 (APTOS only) | EyePACS-augmented |
|---|---|---|---|
| APTOS test (internal) | ROC-AUC | {{val:eyepacs_experiment.models.efficientnet_b0_robust.test.roc_auc|.3f}} | {{val:eyepacs_experiment.models.efficientnet_b0_eyepacs.test.roc_auc|.3f}} |
| APTOS test (internal) | QWK | {{val:eyepacs_experiment.models.efficientnet_b0_robust.test.qwk|.3f}} | {{val:eyepacs_experiment.models.efficientnet_b0_eyepacs.test.qwk|.3f}} |
| IDRiD (external) | sensitivity | {{val:eyepacs_experiment.models.efficientnet_b0_robust.idrid_test.sensitivity|.3f}} | {{val:eyepacs_experiment.models.efficientnet_b0_eyepacs.idrid_test.sensitivity|.3f}} |
| IDRiD (external) | ROC-AUC | {{val:eyepacs_experiment.models.efficientnet_b0_robust.idrid_test.roc_auc|.3f}} | {{val:eyepacs_experiment.models.efficientnet_b0_eyepacs.idrid_test.roc_auc|.3f}} |
| IDRiD (external) | QWK | {{val:eyepacs_experiment.models.efficientnet_b0_robust.idrid_test.qwk|.3f}} | {{val:eyepacs_experiment.models.efficientnet_b0_eyepacs.idrid_test.qwk|.3f}} |
| EyePACS held-out patients | ROC-AUC | {{val:eyepacs_experiment.models.efficientnet_b0_robust.eyepacs_test.roc_auc|.3f}} | {{val:eyepacs_experiment.models.efficientnet_b0_eyepacs.eyepacs_test.roc_auc|.3f}} |

More diverse training data improved external performance: the IDRiD ROC-AUC rose by {{val:eyepacs_experiment.paired_candidate_minus_robust.idrid_test.roc_auc.point_diff|+.3f}} (interval {{val:eyepacs_experiment.paired_candidate_minus_robust.idrid_test.roc_auc.lo|+.3f}} to {{val:eyepacs_experiment.paired_candidate_minus_robust.idrid_test.roc_auc.hi|+.3f}}, excluding zero). On the APTOS test split the QWK fell slightly ({{val:eyepacs_experiment.paired_candidate_minus_robust.test.qwk.point_diff|+.3f}}, interval {{val:eyepacs_experiment.paired_candidate_minus_robust.test.qwk.lo|+.3f}} to {{val:eyepacs_experiment.paired_candidate_minus_robust.test.qwk.hi|+.3f}}), and because the lower end of that interval passed the pre-set tolerance of −0.020, the rule did **not** adopt the model as the default. It is available in the application as an optional, experimental model. IDRiD sensitivity rose but still did not reach 0.90.

### Lesion model v2

A second lesion model, trained with extra e-ophtha data and acquisition augmentation, segmented IDRiD less accurately (mean AUPR {{val:lesion_v2_comparison.idrid_test_mean_aupr.v2|.3f}} against {{val:lesion_v2_comparison.idrid_test_mean_aupr.v1|.3f}}) but produced far fewer false detections on healthy APTOS images (grade-0 images with at least 20 detected haemorrhage regions: {{val:lesion_v2_comparison.aptos_v1.grade0_share_with_HE_ge_20|.1%}} → {{val:lesion_v2_comparison.aptos_v2.grade0_share_with_HE_ge_20|.1%}}) and a rule-derived grade that agreed much better with the APTOS grade (validation QWK {{val:lesion_v2_comparison.aptos_v1.rules_val_qwk|.3f}} → {{val:lesion_v2_comparison.aptos_v2.rules_val_qwk|.3f}}). The pre-set rule required both a small IDRiD loss and an APTOS gain; the IDRiD loss was too large, so v1 stays deployed and v2 is kept as a documented alternative.

## The workflow simulation

Under the simulation's assumptions (100,000 patients a year, 4 cameras, 6-minute imaging, 2 reviewers; prevalence 15%), patients arrived at about {{val:simulation.baseline_run.arrived_per_day|.0f}} per day but only about {{val:simulation.baseline_run.completed_per_day|.0f}} could be imaged: the **cameras** were the bottleneck (utilisation {{val:simulation.baseline_run.utilisation.cameras|.0%}}), while the AI step ({{val:simulation.baseline_run.utilisation.inference|.0%}}) and the reviewers ({{val:simulation.baseline_run.utilisation.reviewers|.0%}}) were lightly used. The lesson is that in such a service the AI is unlikely to be the slow part; imaging capacity is. These are consequences of assumptions, not observations of a real programme.

## What we learned

**Strengths, supported by the evidence.**

- A reproducible, tested pipeline in which every tunable quantity was chosen on validation data and every comparison followed a rule fixed in advance.
- Strong internal screening performance with honest uncertainty estimates and good calibration after temperature scaling.
- An interface that explains each result with checkable evidence (heat map, lesions, criteria checklist) and states its limits.
- Experiments that measured real problems instead of hiding them: silent failure on degraded images (partly fixed by robust training), weak external sensitivity (partly improved by more diverse data).

**Weaknesses and limitations.**

- **External validity is the main gap.** External sensitivity was 0.77–0.83 against an internal 0.93, on only about 100 external images.
- Rare severe grades are poorly separated; the system screens better than it grades.
- No patient-level split for APTOS; possible correlation between test and training images.
- The quality gate is heuristic; the lesion detector transfers imperfectly and cannot see venous beading, IRMA or new vessels.
- Many variants were compared on the same test and IDRiD images, which creates mild selection effects for small differences.
- No clinical, prospective or regulatory validation.

**Engineering completion is not scientific validation.** The software is finished and works; whether it would help real patients is unanswered.

**Most valuable next steps,** in order: (1) external validation on larger, independent datasets such as Messidor-2 and multi-site data, with local re-calibration of the threshold; (2) more diverse training data (the EyePACS result points this way) and patient-level splits; (3) a quality model trained on real quality labels; (4) detectors for venous beading, IRMA and neovascularisation so the rule engine can assess all ICDR criteria; (5) a prospective study with clinicians, which is the only route to clinical claims.
