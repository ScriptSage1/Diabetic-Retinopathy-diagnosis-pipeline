# Evaluation metrics from scratch

How do we know whether a screening model is good? Not from one accuracy figure. This part builds every measure the project uses from a single confusion matrix, then adds threshold choice, calibration, segmentation overlap and statistical uncertainty. Teaching examples are illustrative; the project's measured values are given alongside and analysed in Part VI.

## Why one accuracy number is not enough

Imagine a screening programme in which 10% of eyes are referable. A "model" that always answers *not referable* is right 90% of the time, so its accuracy is 90%, and it misses every patient who needs a specialist. Accuracy hides *which* mistakes are made, and in screening the two kinds of mistake have very different costs. We therefore start from the four possible outcomes.

## The confusion matrix

For a binary question (referable or not), every image ends up in one of four cells:

::: define The four outcomes
- **True positive (TP)**: referable, and the model said referable.
- **False negative (FN)**: referable, but the model said not referable. This is a **missed case**.
- **False positive (FP)**: not referable, but the model said referable. This is a **false alarm**.
- **True negative (TN)**: not referable, and the model said not referable.

Here "positive" means the condition we are looking for (referable DR), not "good news".
:::

{{fig:confusion_teaching.png|An illustrative screening of 100 eyes, 20 of them referable. All worked examples in this chapter use these counts.|58|cmteach}}

The illustrative counts are TP = 18, FN = 2, FP = 8, TN = 72. The project's real test-set counts are TP = {{val:internal_holdout.screening.tp|d}}, FN = {{val:internal_holdout.screening.fn|d}}, FP = {{val:internal_holdout.screening.fp|d}} and TN = {{val:internal_holdout.screening.tn|d}}, out of {{val:internal_holdout.screening.n|d}} images.

## Metrics for the screening decision

### Accuracy

$$ \mathrm{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} $$

The share of all images classified correctly. Example: $(18 + 72)/100 = 0.90$. High is good, but as shown above it can be high for a useless model when one class dominates. The project reports it for the five-grade task only (exact-grade accuracy {{val:internal_holdout.grades.accuracy_5|.3f}}).

### Sensitivity (recall, true-positive rate)

$$ \mathrm{Sensitivity} = \frac{TP}{TP + FN} $$

Of the eyes that truly need referral, the share the system flags [29]. Example: $18/20 = 0.90$; 2 of 20 referable eyes were missed. This is the most important number in screening, because a missed case means a patient with sight-threatening disease is told to come back next year. Its limitation is that a system can reach 100% sensitivity by referring everyone, so it must always be read together with specificity. **Project: {{val:internal_holdout.screening.sensitivity|.3f}}** (95% CI {{val:internal_holdout.confidence_intervals.sensitivity.lo|.3f}} to {{val:internal_holdout.confidence_intervals.sensitivity.hi|.3f}}).

### Specificity (true-negative rate)

$$ \mathrm{Specificity} = \frac{TN}{TN + FP} $$

Of the eyes that do not need referral, the share correctly passed. Example: $72/80 = 0.90$; 8 healthy eyes were sent to a specialist unnecessarily. Low specificity floods clinics with false alarms, costs money and causes anxiety. **Project: {{val:internal_holdout.screening.specificity|.3f}}** ({{val:internal_holdout.confidence_intervals.specificity.lo|.3f}} to {{val:internal_holdout.confidence_intervals.specificity.hi|.3f}}).

### Positive predictive value (PPV, precision)

$$ \mathrm{PPV} = \frac{TP}{TP + FP} $$

Of the eyes the system refers, the share that really need it. Example: $18/26 = 0.69$, so almost a third of referrals are false alarms even though sensitivity and specificity are both 0.90. PPV depends strongly on how common the disease is in the screened population (see below). **Project: {{val:internal_holdout.screening.precision|.3f}}.**

### Negative predictive value (NPV)

$$ \mathrm{NPV} = \frac{TN}{TN + FN} $$

Of the eyes the system clears, the share that really are fine. Example: $72/74 = 0.97$. A high NPV is what makes a "no referral" result reassuring. **Project: {{val:internal_holdout.screening.npv|.3f}}.**

### F1 score and balanced accuracy

$$ F_1 = \frac{2 \cdot \mathrm{PPV} \cdot \mathrm{Sensitivity}}{\mathrm{PPV} + \mathrm{Sensitivity}}, \qquad \mathrm{Balanced\ accuracy} = \frac{\mathrm{Sensitivity} + \mathrm{Specificity}}{2} $$

F1 is the harmonic mean of PPV and sensitivity; it is high only if both are. Example: $2 \cdot 0.692 \cdot 0.90 / (0.692 + 0.90) = 0.78$. Balanced accuracy averages the two rates and is not distorted by class imbalance; example: $0.90$. F1 ignores true negatives, which matter in screening, so it is reported but not used to choose anything. **Project: F1 {{val:internal_holdout.screening.f1|.3f}}, balanced accuracy {{val:internal_holdout.screening.balanced_accuracy|.3f}}.**

::: takeaway Screening metrics travel in pairs
Sensitivity and specificity describe the test; PPV and NPV describe what a result means for a patient in a particular population. Always report the pair, the threshold that produced it, and the counts.
:::

## Thresholds, ROC and precision-recall curves

### The threshold creates the trade-off

The model outputs a score between 0 and 1; a **threshold** turns it into a decision. Lowering the threshold refers more eyes: sensitivity rises and specificity falls. Raising it does the opposite ({{ref:tradeoff}}). There is no threshold that is "correct" in itself; it encodes how much a missed case costs compared with a false alarm.

{{fig:threshold_tradeoff.png|Why a threshold trades sensitivity against specificity (illustrative score distributions).|100|tradeoff}}

### The ROC curve and its area

The **receiver operating characteristic (ROC) curve** plots sensitivity (true-positive rate) against 1 − specificity (false-positive rate) for every possible threshold [27] ({{ref:rocteach}}). A perfect model hugs the top-left corner; random guessing follows the diagonal. The **area under the curve (ROC-AUC)** summarises it in one number with a useful meaning: *the probability that a randomly chosen referable eye receives a higher score than a randomly chosen non-referable eye*. An AUC of 0.5 is chance; 1.0 is perfect separation.

{{fig:roc_teaching.png|An illustrative ROC curve with three thresholds marked.|46|rocteach}}

ROC-AUC does not depend on the threshold, which makes it good for comparing models, but it says nothing about the operating point actually used, and with few positive cases it can look flattering. **Project: ROC-AUC {{val:internal_holdout.screening.roc_auc|.3f}}** ({{val:internal_holdout.confidence_intervals.roc_auc.lo|.3f}} to {{val:internal_holdout.confidence_intervals.roc_auc.hi|.3f}}).

### Precision-recall curve and average precision

The **precision-recall (PR) curve** plots PPV against sensitivity across thresholds. When positives are a minority it is more informative than the ROC curve, because it reacts to false positives in proportion to the number of true positives [28]. Its area (**PR-AUC**, computed as *average precision*) equals the prevalence for a random model and 1 for a perfect one. **Project: PR-AUC {{val:internal_holdout.screening.pr_auc|.3f}}** (test-set prevalence {{val:internal_holdout.screening.prevalence|.3f}}).

## Prevalence and predictive values

Sensitivity and specificity are properties of the test; PPV and NPV also depend on the **prevalence**, the share of referable eyes in the population screened. By Bayes' rule [30]:

$$ \mathrm{PPV} = \frac{\mathrm{Se} \cdot \pi}{\mathrm{Se} \cdot \pi + (1 - \mathrm{Sp})(1 - \pi)} $$

where $\mathrm{Se}$ is sensitivity, $\mathrm{Sp}$ specificity and $\pi$ the prevalence. The APTOS test set is enriched for disease (prevalence {{val:internal_holdout.screening.prevalence|.2f}}). Keeping the project's measured sensitivity and specificity but moving to a population where 10% of eyes are referable, the PPV falls to about 0.70 and the NPV rises to about 0.99 ({{ref:ppv}}). In other words, roughly three in ten referrals would be false alarms there, even with unchanged model performance.

{{fig:ppv_prevalence.png|PPV and NPV as functions of prevalence, computed from the project's measured test-set sensitivity and specificity (the 10% population is illustrative).|66|ppv}}

## Choosing the threshold in this project

The threshold is chosen on the **validation** set by a rule fixed in advance (`metrics.select_threshold`): *the highest threshold at which validation sensitivity is at least 0.90*, which gives the best specificity subject to that sensitivity floor. If no threshold reached 0.90, Youden's index (sensitivity + specificity − 1) would be used instead. The chosen value is then frozen and applied unchanged to the test set and to every new image.

{{tab:thresholds|The operating points actually used, all fitted on APTOS validation data.}}

| Model | Temperature | Threshold on the referable score | Human-review band |
|---|---|---|---|
| Standard EfficientNet-B0 (headline results) | {{val:validation_threshold_selection.temperature|.2f}} | {{val:validation_threshold_selection.threshold|.3f}} | ± {{val:validation_threshold_selection.review_margin|.3f}} |
| B0 + B3 ensemble with flips (application default) | per member | {{val:ensemble_selection.selected.threshold|.3f}} | as stored in `artifacts/ensemble/ensemble.json` |

The **human-review band** is the interval around the threshold within which the application asks for a human reviewer instead of deciding automatically. Its width was chosen on validation data so that at most a fixed share of images falls inside it (`fusion.choose_review_margin`).

::: warning Limits of the threshold choice
The 0.90 target is a design choice for this prototype, not a clinical standard. The threshold is tuned to the APTOS validation images; on other cameras and populations the same score can mean something different. Part VI shows exactly this happening on IDRiD and EyePACS. A real deployment would have to re-calibrate and re-validate locally.
:::

## Metrics for the five grades

### Per-class precision, recall and F1, macro and weighted averages

For multiclass problems each grade gets its own precision (of images predicted as this grade, the share that are) and recall (of images of this grade, the share found), computed one-versus-rest. The **macro average** treats all grades equally, so rare grades count as much as common ones; the **weighted average** weights grades by their frequency and is dominated by grade 0. The project reports per-grade values and the macro-F1 ({{val:internal_holdout.grades.macro_f1|.3f}}), because the rare grades are exactly the clinically severe ones.

### Quadratic weighted kappa (QWK)

Grades are **ordered**: mistaking grade 3 for grade 2 is less serious than mistaking grade 4 for grade 0. Plain accuracy treats both errors alike. **Cohen's weighted kappa** with quadratic weights [31] measures agreement between two raters (here, the dataset labels and the model), penalises disagreements by the square of their distance and corrects for agreement expected by chance:

$$ \kappa_w = 1 - \frac{\sum_{i,j} w_{ij}\, O_{ij}}{\sum_{i,j} w_{ij}\, E_{ij}}, \qquad w_{ij} = \frac{(i-j)^2}{(K-1)^2} $$

$O_{ij}$ is the observed number of images with true grade $i$ and predicted grade $j$ (the confusion matrix), $E_{ij}$ is the number expected by chance from the two sets of marginal totals, and $K = 5$ is the number of grades. Off-by-one errors cost $w = 1/16$, errors of two grades cost $4/16$, and a 0-versus-4 error costs the full 1. A kappa of 1 is perfect agreement, 0 is chance level; values above 0.8 are usually described as very good agreement. QWK was the metric of the APTOS competition and is the project's checkpoint-selection metric. **Project: QWK {{val:internal_holdout.grades.qwk|.3f}}** ({{val:internal_holdout.confidence_intervals.qwk.lo|.3f}} to {{val:internal_holdout.confidence_intervals.qwk.hi|.3f}}).

::: warning A high QWK can hide weak rare grades
Because most images are grade 0 or 2 and most errors are off by one, QWK can be high while grades 3 and 4 are often wrong. That is the case here (Part VI), which is why the per-grade table matters.
:::

## Calibration: can the scores be taken at face value?

A model is **calibrated** if, among all images it scores at 0.8, about 80% really are positive. Calibration matters whenever a score is shown to a person or used with a threshold chosen elsewhere [22].

- A **reliability diagram** groups predictions into bins by score and plots the observed frequency in each bin against the average score; a calibrated model lies on the diagonal.
- **Expected calibration error (ECE)** [34] summarises the gaps:

$$ \mathrm{ECE} = \sum_{b=1}^{B} \frac{n_b}{N}\, \left| \mathrm{acc}(b) - \mathrm{conf}(b) \right| $$

where $B$ is the number of bins (15 here), $n_b$ the number of predictions in bin $b$, $N$ the total, $\mathrm{acc}(b)$ the observed accuracy and $\mathrm{conf}(b)$ the mean confidence in the bin. Example: if 50 predictions with mean confidence 0.9 are 80% correct, and 50 with mean confidence 0.6 are 60% correct, ECE $= 0.5 \cdot 0.1 + 0.5 \cdot 0 = 0.05$.

- The **Brier score** [33] is the mean squared difference between the score $p_i$ and the outcome $y_i$ (1 or 0):

$$ \mathrm{Brier} = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2 $$

A score of 0.9 for a referable eye contributes $0.01$; a score of 0.3 for a referable eye contributes $0.49$. Lower is better; it rewards both discrimination and calibration.

**Project:** temperature scaling reduced the five-grade ECE from {{val:internal_holdout.calibration_uncalibrated.ece_top_label_5grade|.3f}} to {{val:internal_holdout.calibration.ece_top_label_5grade|.3f}} on the test set; the referable-score Brier score is {{val:internal_holdout.calibration.brier_referable|.3f}}.

## Segmentation metrics

For the lesion network, every pixel is a small classification. Overlap metrics compare the predicted mask $B$ with the expert mask $A$ ({{ref:dice}}):

$$ \mathrm{Dice} = \frac{2\,|A \cap B|}{|A| + |B|}, \qquad \mathrm{IoU} = \frac{|A \cap B|}{|A \cup B|} $$

where $|A|$ is the number of pixels in $A$. Both are 1 for perfect overlap and 0 for none; Dice [26] is always at least as large as IoU. For tiny lesions such as microaneurysms a few pixels of misalignment destroy the overlap, so the project also reports the **area under the pixel-level precision-recall curve (AUPR)**, which does not depend on a threshold. Results are in Part VI.

{{fig:dice_iou.png|Dice and IoU compare a predicted mask with an expert mask (illustrative).|56|dice}}

## Confidence intervals and the bootstrap

A metric computed on 519 test images is an *estimate*: a different set of 519 images from the same source would give a slightly different number. A **95% confidence interval** expresses this uncertainty: intervals constructed this way contain the true value in about 95% of repeated studies.

The project uses the **percentile bootstrap** [32]: draw 519 images *with replacement* from the 519 test images (some appear twice, some not at all), recompute the metric, repeat 1,000 times, and take the 2.5th and 97.5th percentiles of the results as the interval ({{ref:boot}}). It needs no formula for each metric and works for AUC and kappa alike. To compare two models fairly, the **paired bootstrap** resamples the *same* images for both models and looks at the interval of their *difference*; if it excludes zero, the difference is unlikely to be resampling noise.

{{fig:bootstrap.png|The bootstrap made visible: sensitivity recomputed on 2,000 resamples of the project's saved test predictions (`results/predictions.csv`). The 2.5th–97.5th percentile range closely matches the interval stored in `metrics.json` from the original 1,000-resample run.|66|boot}}

::: warning What the intervals do not cover
The bootstrap treats images as independent. APTOS has no patient identifiers, so two images of the same patient may be correlated and the true uncertainty may be larger. The intervals also say nothing about how the model would perform on a different population or camera; that is a question of external validity, not sampling noise.
:::

## Speed

Inference **latency** was measured on the Kaggle T4 GPU: the network's forward pass took {{val:latency.model_forward_ms_per_image_bs1|.1f}} ms for one image and {{val:latency.model_forward_ms_per_image_bs16|.2f}} ms per image in batches of 16. Preprocessing plus the quality check took {{val:latency.preprocess_plus_quality_ms_per_image_cpu|.0f}} ms per image on the CPU. The forward pass was timed on random input tensors, and the measurements exclude reading files and Grad-CAM. On a laptop CPU a complete analysis by the interface, including the ensemble, Grad-CAM and lesion module, takes a few seconds.
